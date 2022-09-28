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
from time import time, strftime, gmtime
import typing
from unittest import TestLoader
from xml.etree.ElementInclude import default_loader
from cv2 import log
from nbformat import write

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
import pdb

from uvloop import EventLoopPolicy

from model.mobilenetv2 import MobileNetV2
from model.shufflenet import ShuffleNet
from model.shufflenetv2 import ShuffleNetV2

torch.manual_seed(0)
random.seed(0)


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def make_tb_writer(log_dir):
    writer = tb.SummaryWriter(log_dir=os.path.join('./runs', log_dir+'_'+strftime("%Y-%m-%d_%H-%M",gmtime())))
    return writer

def train_model(args, model, criterion, optimizer, scheduler, num_epochs, train_dataloader, test_dataloader, use_gpu):
    since = time()
    # import pdb
    # pdb.set_trace()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    device = torch.device('cuda' if use_gpu else 'cpu')
    
    writer = make_tb_writer(args.save_path.split('/')[-1])
    for epoch in range(args.start_epoch, num_epochs):
        
        scheduler.step(epoch)
        model.train()
                
        total = 0
        running_loss = 0.0
        running_corrects = 0
            
        tic_batch = time()
            
        for i, (inputs, labels) in enumerate(train_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
                
            optimizer.zero_grad()
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            loss = criterion(outputs, labels)
            loss.backward()   
            optimizer.step()
            
            total += labels.size(0)
            running_loss += loss.item()*inputs.size(0)
            running_corrects += torch.sum(labels.data == preds) 
            
            batch_loss = running_loss / ((i+1)*args.batch_size)
            batch_acc = running_corrects.double() / ((i + 1) * args.batch_size)
                
            
            
            if (i + 1) % args.print_freq == 0:
                print('[Epoch {}/{}]-[batch:{}/{}] lr:{:.4f} Loss: {:.6f}  Acc: {:.4f}  Time: {:.4f} sec/batch'.format(
                    epoch + 1, num_epochs, i + 1, round(len(train_dataloader)), scheduler.get_lr()[0], batch_loss, batch_acc, (time()-tic_batch)/args.print_freq))
                tic_batch = time()
                
        epoch_loss = running_loss / len(train_dataloader)
        epoch_acc = running_corrects.double() / total
        print('Epoch:{}/{} Loss: {:.4f} Acc: {:.4f} \n'.format(epoch + 1, num_epochs, epoch_loss, epoch_acc))
        with open(os.path.join(args.save_path, 'result.txt'), 'a')  as f:
            f.write('Epoch:{}/{} Loss: {:.4f} Acc: {:.4f} \n'.format(epoch + 1, num_epochs, epoch_loss, epoch_acc))
        
        # Evaluation
        print('=============== Evaluating ... ================')        
        eval_loss, eval_acc = evaluation(args=args, epoch=epoch, criterion=criterion, device=device, model=model, test_dataloader=test_dataloader)  
        print('Evaluating >> Loss: {:.4f} Acc: {:.4f}'.format(eval_loss, eval_acc))
        print('===============================================')
        
        
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('Acc/train', epoch_acc, epoch)
        writer.add_scalar('Loss/test', eval_loss, epoch)
        writer.add_scalar('Acc/test', eval_acc, epoch)
        
            
        if (epoch+1)%args.save_epoch_freq == 0:
            if not os.path.exists(args.save_path):
                os.makedirs(args.save_path)
            torch.save(model.state_dict(), os.path.join(args.save_path, "epoch_"+str(epoch+1)+".pth"))
        
        if eval_acc > best_acc:
            best_acc = eval_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            
    time_elapsed = time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Accuracy: {:4f}'.format(best_acc))

    # # 载入最佳模型参数(load best model weights)
    model.load_state_dict(best_model_wts)
    return model

def evaluation(args, epoch, criterion, device, model, test_dataloader, tb_writer=None):
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
    parser.add_argument('--num-epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.045)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--print-freq', type=int, default=50)
    parser.add_argument('--save-epoch-freq', type=int, default=1)
    parser.add_argument('--save-path', type=str, default='./result/cifar10_shufflenetv2_no_downsample')
    parser.add_argument('--resume', type=str, default='', help='For training from one checkpoint')
    parser.add_argument('--start-epoch', type=int, default=0, help='Corresponding to the epoch of resume')
    args = parser.parse_args()
    
    
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding = 4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    
    trainset = torchvision.datasets.CIFAR10(root='./datasets', train=True, download=True, transform=transform_train)

    traindataloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)


    testset = torchvision.datasets.CIFAR10(root='./datasets', train=False, download=True, transform=transform_test)

    testdataloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    
    
    use_gpu = torch.cuda.is_available()
    print("use_gpu:{}".format(use_gpu))
    
    # model = MobileNetV2(num_classes=10)
    # import pdb
    # pdb.set_trace()
    # model = ShuffleNet(num_classes=10)
    model = ShuffleNetV2(num_classes=10)
    
    if use_gpu:
        model.to(torch.device('cuda'))
    else:
        model.to(torch.device('cpu'))
        
    criterion = nn.CrossEntropyLoss()
    
    optimizer_ft = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.00004)
    
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=1, gamma=0.98)
    
    model = train_model(args=args,
                        model=model,
                        criterion=criterion,
                        optimizer=optimizer_ft,
                        scheduler=exp_lr_scheduler,
                        num_epochs=args.num_epochs,
                        train_dataloader=traindataloader,
                        test_dataloader=testdataloader,
                        use_gpu=use_gpu)
    torch.save(model.state_dict(), os.path.join(args.save_path, 'best_model_wts.pth'))