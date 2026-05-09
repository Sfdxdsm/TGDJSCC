import math
from functools import partial
from random import choice
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import PatchEmbed
from torch.nn.init import trunc_normal_

from MAE_FS_v2.util.distortion import info_nce_loss
from util.modules import projector_v2v, DownSample, UpSample
from util.modules import Block, AdaptiveModulator
from util.pos_embed import get_2d_sincos_pos_embed
from util.channel import Channel


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, img_size=32, patch_size=2, in_chans=3,
                 embed_dim=256, depth=6, num_heads=8,
                 decoder_embed_dim=192, decoder_depth=4, decoder_num_heads=6,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, selected_nodes_num=16,
                 downsample=2, C=64, chan_type='awgn', multiple_snr='1,4,7,10,13', mask_ratio=0.75, window_size=4):
        super().__init__()
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.num_patches = self.patch_embed.num_patches
        self.embed_dim = embed_dim
        self.mask_ratio = mask_ratio
        self.N_vis = int(self.num_patches * (1 - self.mask_ratio))
        self.selected_nodes_num = selected_nodes_num
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim), requires_grad=True)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        num_merge = math.floor(math.log(self.num_patches, window_size))
        temp = self.num_patches
        length = 0
        for i in range(num_merge):
            temp = temp // window_size
            length += temp
        self.length = length
        self.latent_tokens_num = length
        self.latent_pos_embed = nn.Parameter(torch.zeros(1, self.latent_tokens_num, embed_dim), requires_grad=True)
        self.latent_token = nn.Parameter(torch.randn(1, self.length, embed_dim))
        self.latent_norm = nn.LayerNorm(embed_dim)
        self.projector = projector_v2v(dim_in=(self.length, self.embed_dim),
                                       dim_out=(self.length, self.embed_dim))
        self.map = nn.Linear(C, embed_dim)

        # feature extract
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.DownSample = DownSample(downsample=downsample, dim=embed_dim, out_dim=C)
        self.proj = nn.Linear(embed_dim, C)
        self.fc_norm = norm_layer(C)

        # snr and task prior generation
        self.layer_num = layer_num = 3
        self.snr_proj_list = nn.ModuleList()
        self.feat_proj_list = nn.ModuleList()
        hidden_dim = embed_dim
        self.feat_proj_list.append(nn.Linear(hidden_dim, hidden_dim))
        for i in range(layer_num):
            self.snr_proj_list.append(AdaptiveModulator(hidden_dim))
            self.feat_proj_list.append(nn.Linear(hidden_dim, hidden_dim))
        self.sigmoid = nn.Sigmoid()
        self.prior_proj = nn.Linear(hidden_dim, hidden_dim)

        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.UpSample = UpSample(upsample=downsample, dim=C, out_dim=decoder_embed_dim)
        self.projReverse = nn.Linear(C, decoder_embed_dim)
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_pos_embed = nn.Parameter(torch.randn(1, self.num_patches +1, decoder_embed_dim), requires_grad=True)
        self.mask_token = nn.Parameter(torch.randn(1, 1, decoder_embed_dim))
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)
        ])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_fc_norm = norm_layer(decoder_embed_dim)
        self.masked_recon_head = nn.Linear(decoder_embed_dim, patch_size ** 2 * 3)

        self.pred_loss_fn = nn.MSELoss()
        # --------------------------------------------------------------------------
        self.norm_pix_loss = norm_pix_loss
        self.initialize_weights()

        self.multiple_snr = multiple_snr.split(",")
        for i in range(len(self.multiple_snr)):
            self.multiple_snr[i] = int(self.multiple_snr[i])
        self.channel = Channel(chan_type, multiple_snr)

        self.depth = depth
        self.memory = {}
        self.fs_recoder = {}

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                            cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                            cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.latent_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        trunc_normal_(self.latent_pos_embed, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x):
        B, _, H, W = x.size()
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        x, mask, ids_restore = self.random_masking(x, self.mask_ratio)
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        latent_tokens = self.latent_token.repeat(x.shape[0], 1, 1) + self.latent_pos_embed
        x = torch.cat((cls_tokens, latent_tokens, x), dim=1)

        #----------------------stage 1: feature extract------------------
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        latent_tokens = self.latent_norm(x[:, 1:self.latent_tokens_num + 1, :])
        cls_tokens = x[:, :1, :]
        img_tokens = x[:, self.latent_tokens_num + 1:, :]
        x = torch.cat([self.proj(cls_tokens), self.DownSample(img_tokens)], dim=1)
        x = self.fc_norm(x)
        return x, latent_tokens, mask, ids_restore

    def feature_pass_channel(self, feature, chan_param, avg_pwr=False):
        noisy_feature = self.channel.forward(feature, chan_param, avg_pwr)
        return noisy_feature

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = torch.cat([self.projReverse(x[:, :1, :]), self.UpSample(x[:, 1:, :])], dim=1)
        x = self.decoder_fc_norm(x)
        self.memory['f_d'] = x

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.masked_recon_head(x)

        # remove cls token
        x = x[:, 1:, :]
        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, aug_samples, momentum_encoder=None, target_moel=None, chan_param=None, alpha=1, beta=1):
        if chan_param is None:
            chan_param = choice(self.multiple_snr)

        features, latent_tokens, mask, ids_restore = self.forward_encoder(imgs)
        noisy_features = self.feature_pass_channel(features, chan_param)
        pred = self.forward_decoder(noisy_features, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)

        with torch.no_grad():
            features_momentum = momentum_encoder(aug_samples).detach()

        feats = torch.cat([self.map(features).mean(dim=1), features_momentum.mean(dim=1)])
        loss_info = info_nce_loss(feats, temperature=0.2)

        with torch.no_grad():
            _, latents_target = target_moel(imgs, chan_param = chan_param)
            latents_target = latents_target.detach()

        node_feats = F.normalize(self.projector(latent_tokens), dim=-1)
        node_feats_target = F.normalize(latents_target, dim=-1)
        loss_align  = torch.nn.MSELoss()(node_feats / 0.05, node_feats_target / 0.05)

        return loss + alpha * loss_info + beta * loss_align

    def get_feature(self, key):
        return self.memory[key]

def mae_vit_base_size32_patch4_enc256_dec192(**kwargs):
    model = MaskedAutoencoderViT(
        img_size=32, patch_size=4, embed_dim=256, depth=1, num_heads=8,
        decoder_embed_dim=192, decoder_depth=2, decoder_num_heads=6,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_base_size32_patch2_enc256_dec192(**kwargs):
    model = MaskedAutoencoderViT(
        img_size=32, patch_size=2, embed_dim=256, depth=4, num_heads=8,
        decoder_embed_dim=192, decoder_depth=4, decoder_num_heads=6,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model



mae_vit_base_patch4 = mae_vit_base_size32_patch4_enc256_dec192
mae_vit_base_patch2 = mae_vit_base_size32_patch2_enc256_dec192












