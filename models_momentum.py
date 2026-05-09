from functools import partial
from random import choice

import torch
import torch.nn as nn

import timm.models.vision_transformer

from util.channel import Channel


class Momentum(timm.models.vision_transformer.VisionTransformer):
    """
    classifier with VisionTransformer backbone for classification task
    """

    def __init__(self, **kwargs):
        super(Momentum, self).__init__(**kwargs)

        norm_layer = kwargs['norm_layer']
        embed_dim = kwargs['embed_dim']

        self.norm = norm_layer(embed_dim)
        self.memory = {}
        del self.head

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        feature = x[:, 1:, :]
        self.memory["feature"] = feature

        return feature

    def feature_hook(self, key):
        return self.memory[key]


def Momentum_base_patch2_embed256(**kwargs):
    model = Momentum(
        img_size=32, patch_size=2, embed_dim=256, depth=4, num_heads=8, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
