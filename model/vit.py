# Reference: https://towardsdatascience.com/implementing-visualttransformer-in-pytorch-184f9f16f632

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat

from einops.layers.torch import Rearrange, Reduce

from torchsummary import summary

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels:int = 3, patch_size=16, emb_size=768, img_size=224):
        super().__init__()
        self.patch_sizes = patch_size
        self.projections = nn.Sequential(
            Rearrange('b c (h h1) (w w1) -> b (h w) (h1 w1 c)', h1=patch_size, w1=patch_size),
            nn.Linear(in_channels*patch_size*patch_size, emb_size)
            )
        # CLASS TOKEN for global class information, average each sequence token 
        self.cls_token = nn.Parameter(torch.randn(1,1, emb_size))
        self.position = nn.Parameter(torch.randn((img_size//patch_size)**2+1, emb_size))
    def forward(self, x):
        b, _, _, _ = x.shape
        x = self.projections(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b = b)
        x = torch.cat([cls_tokens, x], dim=1)
        
        x += self.position
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 768, num_heads: int = 8, dropout: float = 0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.qkv = nn.Linear(emb_size, emb_size*3)
        self.att_dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
    
    def forward(self, x : Tensor, mask: Tensor = None):
        qkv = rearrange(self.qkv(x), 'b n (h d qkv) -> (qkv) b h n d', h=self.num_heads, qkv=3) # n is length number
        query, key, value = qkv[0], qkv[1], qkv[2]
        energy = torch.einsum('bhqd, bhkd -> bhkq', query, key) # every head got global information but partial information for signal token
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
            
        scaling = self.emb_size**(1/2)
        att = F.softmax(energy)/scaling
        att = self.att_dropout(att)
        out = torch.einsum('bhal, bhlv -> bhav', att, value)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.projection(out)
        return out
    
class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        
    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size:int, expansion: int = 4, drop_p: float = 0.):
        super().__init__(
            nn.Linear(emb_size, expansion*emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size: int = 768,
                 drop_p: float = 0.,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0.,
                 ** kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))

class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int = 12, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])

class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size: int = 768, n_class: int = 1000):
        super().__init__(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_class)
        )

class ViT(nn.Sequential):
    def __init__(self,
                 in_channels: int = 3,
                 patch_size: int = 16,
                 emb_size: int = 768,
                 img_size: int = 224,
                 depth: int = 12,
                 n_class: int = 1000,
                 **kwargs):
        super().__init__(
            PatchEmbedding(in_channels=in_channels, patch_size=patch_size, emb_size=emb_size, img_size=img_size),
            TransformerEncoder(depth=depth, emb_size=emb_size, **kwargs),
            ClassificationHead(emb_size=emb_size, n_class=n_class)
        )