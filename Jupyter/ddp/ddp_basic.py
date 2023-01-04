import enum
import os
from statistics import mode
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from math import ceil
from torch.nn.parallel import DistributedDataParallel as DDP
from dataset import partition_dataset
from model_CNN import Net


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    
def cleanup():
    dist.destroy_process_group()
    
    
def demo_basic(rank, world_size):
    print("Running basice DDP example on rank {rank}.")
    
    
    torch.manual_seed(1234)
    train_set, bsz = partition_dataset()
    # device = torch.device("cuda:{}".format(rank))
    
    model = Net().to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01, momentum=0.5)
    loss_fn = nn.NLLLoss()
    
    num_batches = ceil(len(train_set.dataset) / float(bsz))
    
    
    
    for ep in range(10):
        epoch_loss = 0.0
        for ite, (data, target) in enumerate(train_set):
            data = data.to(rank)
            target = target.to(rank)
            
            optimizer.zero_grad()
            output = ddp_model(data)
            loss = loss_fn(output, target)
            epoch_loss += loss
            loss.backward()
            optimizer.step()
            
    cleanup
    
def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)
    
if __name__=="__main__":
    run_demo(demo_basic, 2)