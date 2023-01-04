from cProfile import label
from mimetypes import init
import os
from statistics import mode
from turtle import forward
from wsgiref.simple_server import demo_app
from bleach import clean
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.parallel import DistributedDataParallel as DDP
import tempfile
from math import ceil
from toyModel import ToyMpModel

from dataset import partition_dataset
from model_CNN import Net

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)
        
    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
   
def demo_basic(rank, world_size):
    print(f"Runnig basic DDP example on rank {rank}.")
    setup(rank, world_size)
    
    model = ToyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    
    loss_fn = nn.MSELoss()
    optimzier = optim.SGD(ddp_model.parameters(), lr=0.001)
    
    optimzier.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(rank)
    loss_fn(outputs, labels).backward()
    optimzier.step()
    
    cleanup()
    
def demo_checkpoint(rank, world_size):
    print(f"Running DDP checkpoint example on rank {rank}")
    setup(rank, world_size)
    
    model = ToyModel().to(rank)
    ddp_model = DDP(mode,device_ids=[rank])
    
    CHECKPOINT_PATH = tempfile.gettempdir() + '/model.checkpoint'
    if rank == 0:
        torch.save(ddp_model.state_dict(), CHECKPOINT_PATH)
    dist.barrier()
    
    map_location = {'cuda:%d'%0, 'cuda:%d'%rank}
    ddp_model.load_state_dict(
        torch.load(CHECKPOINT_PATH, map_location=map_location)
    )
    
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr = 0.001)
    
    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(rank)
    
    loss_fn(outputs, labels).backward()
    optimizer.step()
    if rank==0:
        os.remove(CHECKPOINT_PATH)
    cleanup()

def demo_model_parallel(rank, world_size):
    print(f"Running DDP with model parallel example on rank {rank}.")
    setup(rank, world_size)
    
    # setup mp_Model and devices for this process
    dev0 = (rank *2) % world_size
    dev1 = (rank * 2 + 1) % world_size
    
    mp_model = ToyMpModel(dev0, dev1)
    ddp_mp_model = DDP(mp_model)
    
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_mp_model.parameters(), lr=0.001)
    
    #ouptut on dev1
    optimizer.zero_grad()
    outputs = ddp_mp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(dev1)
    loss_fn(outputs, labels).backward()
    optimizer.step()
    
    cleanup()
    
        
def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)


if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = n_gpus
    run_demo(demo_model_parallel, world_size)    