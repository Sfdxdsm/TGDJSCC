import math
from functools import partial
from random import choice
import torch
import torch.nn as nn
import timm.models.vision_transformer
from util.modules import DownSample, projector_v2v
from util.distortion import Distortion, SetCriterion
from util.channel import Channel
from feature_selection.feature_selector import DifferentiableFeatureSelector
from util.modules import UpSample, Block, AdaptiveModulator, TransformerDecoder
import torch.nn.functional as F
from util.matcher import HungarianMatcher


class Encoder(timm.models.vision_transformer.VisionTransformer):
    """
    JSCC encoder with VisionTransformer backbone
    """

    def __init__(self, C, selected_nodes_num=16, window_size=4, multiple_snr='1,4,7,10,13', chan_type='awgn', latent_tokens_num=85, **kwargs):
        super().__init__(**kwargs)
        # param
        norm_layer = kwargs['norm_layer']
        embed_dim = kwargs['embed_dim']
        num_heads = kwargs['num_heads']
        mlp_ratio = kwargs['mlp_ratio']
        depth = kwargs['depth']
        self.img_size = img_size = kwargs['img_size']

        self.patch_size = kwargs['patch_size']
        self.num_patches = int(img_size // self.patch_size) **2
        self.selected_nodes_num = selected_nodes_num

        # Node Merge
        num_merge = math.floor(math.log(self.num_patches, window_size))
        temp = self.num_patches
        length = 0
        for i in range(num_merge):
            temp = temp // window_size
            length += temp
        self.length = length
        self.latent_tokens_num = latent_tokens_num
        self.latent_pos_embed = nn.Parameter(torch.zeros(1, self.latent_tokens_num, embed_dim), requires_grad=True)
        self.latent_token = nn.Parameter(torch.randn(1, self.latent_tokens_num, embed_dim))
        torch.nn.init.normal_(self.latent_token, std=.02)
        self.latent_norm = nn.LayerNorm(embed_dim)

        self.p = projector_v2v(
            dim_in=(self.latent_tokens_num, self.embed_dim),
            dim_out=(self.length, self.embed_dim)
        )
        self.norm_p = nn.LayerNorm(embed_dim)

        L = int(selected_nodes_num ** 0.5)
        assert L**2 == selected_nodes_num, 'selected_nodes_num must be divisible by L'
        self.downsample_num = img_size // self.patch_size // L // 2
        self.DownSample = DownSample(
            downsample=self.downsample_num,
            dim=embed_dim,
            out_dim=C,
        )
        self.fc_norm = nn.LayerNorm(C)

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

        #feature selector
        self.feature_selector = DifferentiableFeatureSelector(feature_dim=hidden_dim, prior_dim=hidden_dim,
                                                              hidden_dim=hidden_dim, k=self.selected_nodes_num)
        self.select_blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)]
        )

        # channel projection
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, C),
            norm_layer(C)
        )

        # channel param
        self.multiple_snr = multiple_snr.split(",")
        for i in range(len(self.multiple_snr)):
            self.multiple_snr[i] = int(self.multiple_snr[i])
        self.channel = Channel(chan_type, multiple_snr)

        self.memory = {}
        del self.head

    def forward(self, x, chan_param=1, task_id=0):
        B, _, H, W = x.size()
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)

        latent_tokens = self.latent_token.repeat(x.shape[0], 1, 1) + self.latent_pos_embed[:, :, :]

        x = torch.cat((cls_tokens, latent_tokens, x), dim=1)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        cls_tokens = x[:, :1, :]
        latent_tokens = x[:, 1: self.latent_tokens_num + 1, :]
        img_tokens = x[:, self.latent_tokens_num + 1:, :]
        node_feats = torch.cat((self.norm_p(self.p(latent_tokens)) , img_tokens), dim=1)

        snr_cuda = torch.tensor(chan_param, dtype=torch.float).to(x.device)
        snr_batch = snr_cuda.unsqueeze(0).expand(B, -1)
        for i in range(self.layer_num):
            if i == 0:
                temp = self.feat_proj_list[i](node_feats.detach())
            else:
                temp = self.feat_proj_list[i](temp)
            snr_weight = self.snr_proj_list[i](snr_batch).unsqueeze(1).expand(-1, node_feats.shape[1], -1)
            temp = temp * snr_weight
        combine_weight = self.sigmoid(self.feat_proj_list[-1](temp))
        prior = node_feats * combine_weight
        node_feats = node_feats + node_feats * combine_weight

        # ----------------------stage 3: feature selection------------------
        # prior info
        prior = self.prior_proj(prior)

        selected_feats, indicators, scores = self.feature_selector(node_feats, prior)
        selected_feats = torch.cat([cls_tokens,  selected_feats], dim=1)

        # ----------------------stage 4: channel projection------------------
        selected_feats = self.proj(selected_feats)[:, 1:, :]  + self.DownSample(img_tokens)
        selected_feats = self.fc_norm(selected_feats)
        self.memory["feature"] = selected_feats
        self.memory['node_feats'] = node_feats
        return selected_feats, node_feats

    def get_feature(self, key):
        return self.memory[key]

class Decoder(nn.Module):
    """ JSCC decoder with VisionTransformer backbone
    """
    def __init__(self, C, selected_nodes_num=64, patch_size=2, img_size=32, embed_dim=256, num_heads=8, num_queries=20, mlp_ratio=4, depth=4, num_classes=10, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        L = int(selected_nodes_num ** 0.5)
        assert  L ** 2 == selected_nodes_num, 'selected_nodes_num must be divisible by L'

        self.num_patches = int(img_size // patch_size) **2
        self.decoder_embed_dim = embed_dim
        self.upsample_num = img_size // patch_size // L // 2
        if self.upsample_num>0:
            self.Upsample = UpSample(
                upsample=self.upsample_num,
                dim=C,
                out_dim=embed_dim,
            )
        else:
            self.Upsample = nn.Linear(C, embed_dim)

        self.pos_embed = nn.Parameter(torch.zeros((1, self.num_patches, embed_dim)))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.query_embed = nn.Embedding(num_queries, embed_dim)
        self.decoder_layers = nn.ModuleList([
            TransformerDecoder(embed_dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.decoder_norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes+1)
        self.bbox_predictor = nn.Sequential(
            nn.Linear(embed_dim, mlp_ratio * embed_dim),
            nn.LayerNorm(mlp_ratio * embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_ratio * embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 4),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, N_vis, _ = x.shape
        x = self.Upsample(x)
        x = x + self.pos_embed
        query = self.query_embed.weight
        query = query.unsqueeze(0).expand(B,-1,-1)

        for blk in self.decoder_layers:
            query = blk(query, x)
        query = self.decoder_norm(query)
        pred_logits = self.classifier(query)
        pred_boxes = self.bbox_predictor(query)

        return pred_logits, pred_boxes


class Net(nn.Module):
    def __init__(self, encoder, decoder, bbox_loss_coef,
                 giou_loss_coef, set_cost_class, set_cost_box, set_cost_giou, eos_coef,
                 is_channel=True, multiple_snr='1,4,7,10,13', chan_type='awgn'):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.patch_size = encoder.patch_size
        self.is_channel = is_channel
        self.multiple_snr = multiple_snr.split(",")
        for i in range(len(self.multiple_snr)):
            self.multiple_snr[i] = int(self.multiple_snr[i])
        self.channel = Channel(chan_type, multiple_snr)

        embed_dim = encoder.embed_dim
        self.decoder_pred = nn.Linear(embed_dim, self.patch_size ** 2 * 3)
        self.matcher = HungarianMatcher(cost_class=set_cost_class,cost_bbox=set_cost_box,cost_giou=set_cost_giou)
        self.weight_dict = {'loss_ce':1, 'loss_bbox':bbox_loss_coef}
        self.weight_dict['loss_giou'] = giou_loss_coef
        self.Criterion = SetCriterion(decoder.num_classes, matcher=self.matcher, weight_dict=self.weight_dict,
                                      eos_coef=eos_coef, losses=['labels', 'boxes', 'cardinality'])

    def feature_pass_channel(self, feature, chan_param, avg_pwr=False):
        noisy_feature = self.channel.forward(feature, chan_param, avg_pwr)
        return noisy_feature

    def forward(self, input_image, chan_param=None, targets=None):
        B, _, H, W = input_image.size()
        if chan_param is None:
            SNR = choice(self.multiple_snr)
            chan_param = SNR

        selected_feats, node_feats = self.encoder(input_image, chan_param, task_id=0)

        if self.is_channel:
            noisy_feature = self.feature_pass_channel(selected_feats, chan_param)
        else:
            noisy_feature = selected_feats

        CBR = selected_feats.numel() / 2 / input_image.numel()
        pred_logits, pred_boxes = self.decoder(noisy_feature)

        outputs = {
            'pred_logits':pred_logits,
            'pred_boxes':pred_boxes
        }
        loss_dict = self.Criterion(outputs, targets)
        losses = sum(loss_dict[k]*self.weight_dict[k] for k in loss_dict.keys() if k in self.weight_dict)
        loss_dict['total_loss'] = losses
        loss_dict['pred_logits'] = pred_logits
        loss_dict['pred_boxes'] = pred_boxes
        return loss_dict, CBR, chan_param

def enc_base_patch2_embed256(**kwargs):
    model = Encoder(
        img_size=32, patch_size=2, embed_dim=256, depth=4, num_heads=8, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def dec_base_patch2_embed256(**kwargs):
    model = Decoder(
        img_size=32, patch_size=2, embed_dim=1024, depth=2, num_heads=8, mlp_ratio=4, **kwargs)
    return model

