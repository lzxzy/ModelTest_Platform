import modulefinder
from operator import mod
import random
from turtle import forward
from typing import OrderedDict
from xmlrpc.client import TRANSPORT_ERROR
from numpy import isin
import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb

def conv_3x3_bn(inp, oup, stride):
    
    conv_bn = nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )
    return conv_bn

def conv_1x1_bn(inp, oup):
    conv_bn = nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )
    return conv_bn

def channel_shuffle(x, groups):
    batchsize, num_channels, h, w = x.data.size()
    
    channel_per_group = num_channels // groups
    
    x = x.view(batchsize, groups, channel_per_group, h, w)
    x = torch.transpose(x, 1, 2).contiguous()
    
    x = x.view(batchsize, -1, h, w)
    return x

class ShuffleNet_Unit(nn.Module):
    
    def __init__(self, inp, stride, split_ratio=0.5, downsample=True, use_SE=False, red_ratio=8):
        super(ShuffleNet_Unit, self).__init__()
        
        self.use_SE = use_SE
        self.stride = stride
        if self.stride == 2:
            self.inp = inp
            self.oup = inp
            
            self.res_depthwise_conv = nn.Sequential(
                nn.Conv2d(self.inp, self.inp, 3, self.stride if downsample else 1, 1, groups=self.inp, bias=True),
                nn.BatchNorm2d(self.inp)
            )
            
            self.res_pointwise_conv = nn.Sequential(
                nn.Conv2d(self.inp, self.inp, 1, 1, 0, bias=True),
                nn.BatchNorm2d(self.inp),
                nn.ReLU6(inplace=True)
            )
            
        elif self.stride==1:
            self.inp = int(inp * split_ratio)
            self.oup = int(inp * split_ratio)
            self.res_ch_num = inp - self.inp
        
        self.pre_pointwise_shuffle_conv= nn.Sequential(
            nn.Conv2d(self.inp, self.oup, 1, 1, 0, bias=True),
            nn.BatchNorm2d(self.oup),
            nn.ReLU6(inplace=True)
        )
        
        self.depthwise_conv_layer=nn.Sequential(
            nn.Conv2d(self.oup, self.oup, 3, self.stride if downsample else 1, 1, groups=self.oup, bias=True),
            nn.BatchNorm2d(self.oup)
        )
        
        
        self.after_pointwise_conv=nn.Sequential(
            nn.Conv2d(self.oup, self.oup, 1, 1, 0, bias=True),
            nn.BatchNorm2d(self.oup),
            nn.ReLU6(inplace=True)
        )
        
        if self.use_SE:
            self.SE_block = nn.Sequential(
                nn.Linear(inp, inp//red_ratio, bias=False),
                nn.ReLU6(inplace=True),
                nn.Linear(inp//red_ratio, inp, bias=False),
            )
        
    def forward(self, x):
        
        if self.stride == 1:
            residula = x[:,:self.res_ch_num, :, :]
                   
            x = self.pre_pointwise_shuffle_conv(x[:,self.res_ch_num:, :, :])

            x = self.depthwise_conv_layer(x)
            x = self.after_pointwise_conv(x)
        
            x = torch.cat((residula, x), 1)
            x = channel_shuffle(x, 2)
        elif self.stride == 2:
            residula = self.res_depthwise_conv(x)
            residula = self.res_pointwise_conv(residula)
            
            x = self.pre_pointwise_shuffle_conv(x)

            x = self.depthwise_conv_layer(x)
            x = self.after_pointwise_conv(x)

            x = torch.cat((residula, x), 1)
            x = channel_shuffle(x, 2)       
        else:
            raise ValueError("Invaliade stride number {}".format(self.stride))   
        
        if self.use_SE:
            # pdb.set_trace()
            b, c, _, _ = x.size()
            w_scale = F.adaptive_avg_pool2d(x, (1,1))
            w_scale = w_scale.view(w_scale.size(0), -1)
            w_scale = self.SE_block(w_scale)
            w_scale = F.sigmoid(w_scale).view(b, c, 1, 1)
            x = x * w_scale.expand_as(x)
        
        return x
    
class ShuffleNetV2(nn.Module):
    def __init__(self, inp=3, num_classes=1000):
        super(ShuffleNetV2, self).__init__()
        # pdb.set_trace()
        self.inp = inp
        self.num_classes = num_classes
        self.repeat = [3, 7, 3]
        
        self.oup = [24, 48, 96, 192]
        
        
        block = ShuffleNet_Unit
        
        self.head_bn = nn.Sequential(
            conv_3x3_bn(inp=self.inp, oup=24, stride=1),
            nn.MaxPool2d(3, 2, 1)
        )
        
        self.stage_2 = self._make_stage(self.oup[0], self.oup[1], 3, block, stride=2, downsample=False, use_SE=True)
        self.stage_3 = self._make_stage(self.oup[1], self.oup[2], 7, block, stride=2, downsample=False, use_SE=True)
        self.stage_4 = self._make_stage(self.oup[2], self.oup[3], 3, block, stride=2, use_SE=True)
        
        self.bottom_bn = nn.Sequential(
            conv_1x1_bn(inp=self.oup[-1], oup=1024),
            nn.AvgPool2d(8)
        )
        # self.global_pooling = nn.MaxPool2d(8)
        # self.global_pooling = 
        self.fc = nn.Linear(1024, num_classes, bias=True)
        
        self._initial_weights()
        
    def _make_stage(self, inp, oup, rep, block, stride=1, split_ratio=0.5, downsample=True, use_SE=False):
        layers = [block(inp, stride=stride, downsample=downsample)]
        
        for i in range(rep):
            layers.append(block(oup, stride=1, split_ratio=split_ratio, downsample=downsample, use_SE=use_SE))
            
        return nn.Sequential(*layers)
        
    def _initial_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant(m.weight, 1)
                nn.init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant(m.bias, 0)
                    
    def forward(self, x):
        # import pdb
        # pdb.set_trace()
        x = self.head_bn(x)
        x = self.stage_2(x)
        x = self.stage_3(x)
        x = self.stage_4(x)
        x = self.bottom_bn(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x