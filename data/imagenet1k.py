import imp
import torch.nn as nn
import os
import json

from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

from class_names_imagenet import lab_dict

class ImageNet1k(ImageFolder):
    def __init__(self, root, train=False, transform=None, target_transform=None, category='name', loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        
        self.root = root