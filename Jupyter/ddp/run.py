from mimetypes import init
import os
from sympy import Ne
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from math import ceil

from dataset import partition_dataset
from model_CNN import Net
# def run(rank, size):
#     pass

# def run(rank, size):
#     '''
#         Blocking point-to-point communication.
#     '''
#     tensor = torch.zeros(1)
#     if rank == 0:
#         tensor += 1
#         # send tensor to process 1
#         dist.send(tensor=tensor, dst=1)
#     else:
#         dist.recv(tensor=tensor, src=0)
        
#     print('Rank', rank, 'has data', tensor[0])

# def run(rank, size):
#     '''
#         Un-Blocking point-to-point communication.
#     '''
#     tensor = torch.zeros(1)
#     if rank == 0:
#         tensor += 1
#         # send tensor to process 1
#         req = dist.isend(tensor=tensor, dst=1)
#         print('Rank 0 started sending')    
#     else:
#         req = dist.irecv(tensor=tensor, src=0)
#         print('Rank 1 started receiving')

#     req.wait()
#     print('Rank', rank, 'has data', tensor[0])

# def run(rank, size):
#     '''
#     All-Reduce example.
#     Simple collective communication.
#     '''
#     group = dist.new_group([0, 1])
#     tensor = torch.ones(1)
#     dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
#     print('Rank', rank, 'has data', tensor[0])

def run(rank, size):
    torch.manual_seed(1234)
    train_set, bsz = partition_dataset()
    device = torch.device("cuda:{}".format(rank))
    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    
    num_batches = ceil(len(train_set.dataset) / float(bsz))
    for epoch in range(10):
        epoch_loss = 0.0
        for data, target in train_set:
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            epoch_loss += loss
            loss.backward()
            average_gradients(model)
            optimizer.step()
        print ('Rank', dist.get_rank(), ' epoch', epoch, ' : ', epoch_loss/num_batches)
            

def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size
        

def init_process(rank, size, fn, backend='gloo'):
    '''
    Initialize the distributed environment. 
    '''
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    
    dist.init_process_group(backend=backend, rank=rank, world_size=size)
    fn(rank, size)
    
if __name__ == "__main__":
    size = 2
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()