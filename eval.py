import argparse
from ast import parse
import csv
import enum
import modulefinder
from operator import mod
import os
import random
from sched import scheduler
from statistics import mode
import sys
from datetime import datetime
from time import time
import typing
from unittest import TestLoader
from xml.etree.ElementInclude import default_loader
from nbformat import write
import os

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.utils.tensorboard as tb

import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import trange
import copy

from uvloop import EventLoopPolicy

from model.mobilenetv2 import MobileNetV2
from model.shufflenet import ShuffleNet

torch.manual_seed(0)
random.seed(0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def evaluation(criterion, device, model, test_dataloader):
    model.eval()
    
    eval_corrects = 0
    eval_loss = 0.0
    total = 0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            
            eval_loss += criterion(outputs, labels)
            total += labels.size(0)
            eval_corrects += torch.sum(labels.data == preds)
    eval_loss = eval_loss / len(test_dataloader)
    eval_acc = eval_corrects.double() / total
    return eval_loss, eval_acc

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    
    parser = argparse.ArgumentParser(description='PyTorch implementation of MobileNetV2')
    parser.add_argument('--data-dir', type=str, default='./datasets')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-class', type=int, default=100)
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.045)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--print-freq', type=int, default=50)
    parser.add_argument('--eval-epoch', type=str, default='best_model_wts.pth')
    parser.add_argument('--save-path', type=str, default='./result/cifar10_shufflenet_no_maxpool')
    parser.add_argument('--resume', type=str, default='', help='For training from one checkpoint')
    parser.add_argument('--start-epoch', type=int, default=0, help='Corresponding to the epoch of resume')
    args = parser.parse_args()
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
    
    testset = torchvision.datasets.CIFAR10(root='./datasets', train=False, download=True, transform=transform_test)

    testdataloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    
    
    use_gpu = torch.cuda.is_available()
    print("use_gpu:{}".format(use_gpu))
    device = torch.device('cuda' if use_gpu else 'cpu')
    
    # model = MobileNetV2(num_classes=10)
    # import pdb
    # pdb.set_trace()
    model = ShuffleNet(num_classes=10)
    model.load_state_dict(torch.load(os.path.join(args.save_path, args.eval_epoch)))
    if use_gpu:
        model.to(torch.device('cuda'))
    else:
        model.to(torch.device('cpu'))
        
    
    criterion = nn.CrossEntropyLoss()
    
    eval_loss, eval_acc = evaluation(criterion=criterion,
                                    device=device,
                                    model=model,
                                    test_dataloader=testdataloader,
                                    )
    print('=============== Evaluating ... ================')        
    print('Evaluating >> Loss: {:.4f} Acc: {:.4f}'.format(eval_loss, eval_acc))
    print('===============================================')
        
        