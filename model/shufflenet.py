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
    
    def __init__(self, inp, oup, stride, group):
        super(ShuffleNet_Unit, self).__init__()
        
        self.inp = inp
        self.oup = oup
        self.bottleneck_channels = oup//4
        self.stride = stride
        self.group = group
        
        self.pre_pointwise_shuffle_gconv= nn.Sequential(
            nn.Conv2d(inp, self.bottleneck_channels, 1, 1, 0, groups=self.group, bias=True),
            nn.BatchNorm2d(self.bottleneck_channels),
            nn.ReLU6(inplace=True)
        )
        
        self.depthwise_conv_layer=nn.Sequential(
            nn.Conv2d(self.bottleneck_channels, self.bottleneck_channels, 3, self.stride, 1, groups=self.bottleneck_channels, bias=True),
            nn.BatchNorm2d(self.bottleneck_channels)
        )
        
        if self.stride == 2:
            self.oup -= self.inp
        self.after_pointwise_gconv=nn.Sequential(
            nn.Conv2d(self.bottleneck_channels, self.oup, 1, 1, 0, groups=self.group, bias=True),
            nn.BatchNorm2d(self.oup)
        )
        
    def forward(self, x):
        residula = x
        
        x = self.pre_pointwise_shuffle_gconv(x)
        x = channel_shuffle(x, self.group)
        x = self.depthwise_conv_layer(x)
        x = self.after_pointwise_gconv(x)
        
        if self.stride == 1:
            x = x + residula
        elif self.stride == 2:
            x = torch.cat((x, F.avg_pool2d(residula, kernel_size=3, stride=2, padding=1)), 1)
        else:
            raise ValueError("Invaliade stride number {}".format(self.stride))   
        
        return F.relu6(x)
    
class ShuffleNet(nn.Module):
    def __init__(self, inp=3, num_classes=1000, groups=8):
        super(ShuffleNet, self).__init__()
        
        self.inp = inp
        self.num_classes = num_classes
        self.group = groups
        self.repeat = [3, 7, 3]
        
        if self.group == 1:
            self.oup = [24, 144, 288, 576]
        elif self.group == 2:
            self.oup = [24, 200, 400, 800]
        elif self.group == 3:
            self.oup = [24, 240, 480, 960]
        elif self.group == 4:
            self.oup = [24, 272, 544, 1088]
        elif self.group == 8:
            self.oup = [24, 384, 768, 1536]
        
        block = ShuffleNet_Unit
        
        self.head_bn = nn.Sequential(
            conv_3x3_bn(inp=self.inp, oup=24, stride=1),
            # nn.MaxPool2d(3, 2, 1)
        )
        
        # modules = OrderedDict()
        # for s, r in enumerate(self.repeat):
        #     layers = [block(self.oup[s], self.oup[s+1], stride=2, group=self.group)]
        
        #     for i in range(r):
        #         layers.append(block(self.oup[s+1], self.oup[s+1], stride=1, group=self.group))
            
        #     modules[f'stage_{s+1}'] = nn.Sequential(*layers)
            
        # self.stages = nn.Sequential(modules)
        
        self.stage_2 = self._make_stage(self.oup[0], self.oup[1], 3, block, stride=2)
        self.stage_3 = self._make_stage(self.oup[1], self.oup[2], 7, block, stride=2)
        self.stage_4 = self._make_stage(self.oup[2], self.oup[3], 3, block)
        
        
        # self.global_pooling = nn.MaxPool2d(8)
        self.global_pooling = nn.AvgPool2d(4)
        self.fc = nn.Linear(self.oup[-1], num_classes, bias=True)
        
        self._initial_weights()
        
    def _make_stage(self, inp, oup, rep, block, stride=2):
        layers = [block(inp, oup, stride=stride, group=self.group)]
        
        for i in range(rep):
            layers.append(block(oup, oup, stride=1, group=self.group))
            
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
        x = self.global_pooling(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x