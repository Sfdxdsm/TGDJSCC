import math
from functools import partial
from random import choice
import torch
import torch.nn as nn
import timm.models.vision_transformer

from util.modules import DownSample, projector_v2v
from util.distortion import Distortion
from util.channel import Channel
from feature_selection.feature_selector import DifferentiableFeatureSelector
from util.modules import UpSample, Block, AdaptiveModulator
import torch.nn.functional as F


class Classifier(timm.models.vision_transformer.VisionTransformer):
    """
    classifier with VisionTransformer backbone for classification task
    """


    def __init__(self, C, n_classes, selected_nodes_num=16, window_size=4, multiple_snr='1,4,7,10,13', chan_type='awgn',
                 latent_tokens_num = 85,
                 **kwargs):
        super().__init__(**kwargs)
        # param
        norm_layer = kwargs['norm_layer']
        embed_dim = kwargs['embed_dim']
        img_size = kwargs['img_size']
        self.patch_size = kwargs['patch_size']
        self.num_patches = int(img_size // self.patch_size) ** 2
        self.window_size = window_size
        self.selected_nodes_num = selected_nodes_num

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim), requires_grad=True)
        self.latent_tokens_num = latent_tokens_num
        self.latent_pos_embed = nn.Parameter(torch.zeros(1, self.latent_tokens_num, embed_dim), requires_grad=True)
        self.latent_token = nn.Parameter(torch.randn(1, self.latent_tokens_num, embed_dim))
        self.latent_norm = nn.LayerNorm(embed_dim)
        torch.nn.init.normal_(self.latent_token, std=.02)

        num_merge = math.floor(math.log(self.num_patches, window_size))
        temp = self.num_patches
        length = 0
        for i in range(num_merge):
            temp = temp // window_size
            length += temp
        self.length = length #= latent_tokens_num in the pretraining phase
        self.p = projector_v2v(
            dim_in=(self.latent_tokens_num, self.embed_dim),
            dim_out=(self.length, self.embed_dim)
        )
        self.norm_p = nn.LayerNorm(embed_dim)

        self.DownSample = DownSample(
            downsample=2,
            dim=embed_dim,
            out_dim=C,
        )
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

        # feature selector
        self.feature_selector = DifferentiableFeatureSelector(feature_dim=hidden_dim, prior_dim=hidden_dim,
                                                              hidden_dim=hidden_dim, k=self.selected_nodes_num)

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
        self.head = nn.Linear(selected_nodes_num * C, n_classes) if n_classes > 0 else nn.Identity()
        self.memory = {}

    def feature_pass_channel(self, feature, chan_param, avg_pwr=False):
        noisy_feature = self.channel.forward(feature, chan_param, avg_pwr)
        return noisy_feature

    def forward_encoder(self, x, chan_param=1):
        B, _, H, W = x.size()
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)

        latent_tokens = self.latent_token.repeat(x.shape[0], 1, 1) + self.latent_pos_embed[:, :, :]
        latent_tokens = self.latent_norm(latent_tokens)
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
        selected_feats = self.proj(selected_feats)[:, 1:, :] + self.DownSample(img_tokens)
        selected_feats = self.fc_norm(selected_feats)
        self.memory["feature"] = selected_feats
        return selected_feats

    def forward(self, x, chan_param=None):
        if chan_param is None:
            SNR = choice(self.multiple_snr)
            chan_param = SNR

        feature = self.forward_encoder(x, chan_param)

        noisy_feature = self.feature_pass_channel(feature, chan_param)

        self.memory["noisy features"] = noisy_feature

        out = noisy_feature.view(x.shape[0], -1)
        outcome = self.head(out)
        return outcome

    def feature_hook(self, key):
        return self.memory[key]


def classifier_base_patch4_embed256(**kwargs):
    model = Classifier(
        img_size=32, patch_size=4, embed_dim=256, depth=4, num_heads=8, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def classifier_base_patch2_embed256(**kwargs):
    model = Classifier(
        img_size=32, patch_size=2, embed_dim=256, depth=4, num_heads=8, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model





















