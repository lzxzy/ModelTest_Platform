from audioop import bias
import imp
from sqlite3 import OptimizedUnicode
from tkinter.messagebox import NO
from turtle import forward, width
import warnings
from functools import partial
from typing import Any, Callable, List, Optional
from xmlrpc.client import TRANSPORT_ERROR
from sympy import Inverse
import math

import torch
from torch import nn, Tensor


__all__ = ["MobileNetV2", "MobileNet_V2_Weights", "mobilenet_v2"]


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v+divisor/2)//divisor*divisor)
    if new_v <0.9*v:
        new_v += divisor
    return new_v
    

def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )
    
def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

class InvertedResidual(nn.Module):
    def __init__(self, inp:int, oup:int, stride:int, expand_ratio:int, norm_layer:Optional[Callable[..., nn.Module]] = None) -> None:
        super().__init__()
        self.stride = stride
        if stride not in [1, 2]:
            raise ValueError(f"stride should be 1 or 2 insted of {stride}")
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
            
        hidden_dim = int(round(inp*expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup
        
        if expand_ratio == 1:
            self.conv = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup)
            )
        
    def forward(self, x:Tensor)->Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)
        
class MobileNetV2(nn.Module):
    def __init__(
        self,
        num_classes: int=1000,
        width_mult:float = 1.0,
        inverted_residual_setting:Optional[List[List[int]]] = None,
        round_nearest: int = 8,
        dropout:float = 0.2
        ) -> None:
        super(MobileNetV2, self).__init__()
        
        self.cfgs = [
            # t, c, n, s
            [1,  16, 1, 1],
            [6,  24, 2, 1],
            [6,  32, 3, 1],
            [6,  64, 4, 2],
            [6,  96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]


        input_channel = _make_divisible(32*width_mult, 4 if width_mult==0.1 else 8)
        layers = [conv_3x3_bn(3, input_channel, 1)]
        
        block = InvertedResidual
        
        for t, c, n, s in self.cfgs:
            output_channel = _make_divisible(c*width_mult, 4 if width_mult == 0.1 else 8)
            for i in range(n):
                layers.append(block(input_channel, output_channel, s if i==0 else 1, t))
                input_channel = output_channel
            
        self.features = nn.Sequential(*layers)
                
        output_channel = _make_divisible(1280*width_mult, 4 if width_mult==0.1 else 8) if width_mult>1.0 else 1280
        self.conv = conv_1x1_bn(input_channel, output_channel)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Linear(output_channel, num_classes)
        
        self._initialize_weights()

    def forward(self, x:Tensor) -> Tensor:
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def mobilenetv2(**kwargs):
    return MobileNetV2(**kwargs)