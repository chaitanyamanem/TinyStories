
import os
import sys
import tempfile
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP

# On Windows platform, the torch.distributed package only
# supports Gloo backend, FileStore and TcpStore.
# For FileStore, set init_method parameter in init_process_group
# to a local file. Example as follow:
# init_method="file:///f:/libtmp/some_file"
# dist.init_process_group(
#    "gloo",
#    rank=rank,
#    init_method=init_method,
#    world_size=world_size)
# For TcpStore, same way as on Linux.


## Dataset

class Dataset:
    def __init__(self, batch_size=32):        
        self.X_train = torch.rand(128, 10)
        self.y_train = torch.rand(128, 5)        
        self.X_test = torch.rand(128, 10)
        self.y_test = torch.rand(128, 5)
        self.batch_size = batch_size
        
    def get_dataloader(self, dtype="train"):
        if dtype == "train":
            sampler = DistributedSampler(self.X_train)
            train_loader = DataLoader(TensorDataset(self.X_train,self.y_train), batch_size=self.batch_size, shuffle=False, sampler=sampler)
            return train_loader
        if dtype == "test":
            test_loader = DataLoader(TensorDataset(self.X_test,self.y_test), batch_size=self.batch_size, shuffle=False)
            return test_loader

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
    
class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))

def val_loop(model, data_loader, loss_fn, rank):
    loss = torch.zeros(len(data_loader))
    model = model.module.to(rank)
    test_bar = tqdm(total=len(data_loader), desc='Test Step', position=0)
    for s,(x,y) in enumerate(data_loader):
        x, y = x.to(rank), y.to(rank)
        outputs = model(x)        
        loss[s] = loss_fn(outputs, y)
        test_bar.update(1)
    return loss.mean()        
    
    

def demo_basic(rank, world_size, dataset, grad_acc_steps):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)
    
    
    
    
    # create model and move it to GPU with id rank
    model = ToyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    loss_fn = nn.MSELoss()    
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
    ## get the dataloader
    train_dataloader = dataset.get_dataloader(dtype="train")
    test_dataloader = dataset.get_dataloader(dtype="test")
    print(f"Total number of steps in the datset on the GPU:{rank} is {len(train_dataloader)}.")
    ## set the tqdm bar
    #train_bar = tqdm(total=len(train_dataloader), desc='Train Step', position=0, disable = not rank == 0)
    
    for s,(x,y) in enumerate(train_dataloader):
        ddp_model.require_backward_grad_sync = (s + 1) % grad_acc_steps == 0
        print(f"GPU{rank}: At step{s+1} weights of w11 and w12 is {model.net1.weight[0,:2]}")
        x,y = x.to(rank), y.to(rank) 
        print(f"GPU{rank}: At step{s+1} inputs shape:{x.shape} inputs{x[:2,:4]}")       
        outputs = ddp_model(x)        
        loss = loss_fn(outputs, y)
        #print(f"Loss on GPU{rank} for STEP{s} is: {loss}")
        loss.backward()
        print(f"GPU{rank}: At step{s+1} gradients for w11 and w12 is {model.net1.weight.grad[0,:2]}")
        if (s + 1) % grad_acc_steps == 0:
            optimizer.step()
            print(f"GPU{rank}: At step{s+1} weights of w11 and w12 is {model.net1.weight[0,:2]}")
            optimizer.zero_grad()
            
        # if rank == 0  and s % 200 == 0:
        #     val_loss = val_loop(ddp_model, test_dataloader, loss_fn, rank)
        #     print(f"***validation*** loss on GPU{rank} for STEP{s} is: {val_loss}")
        #dist.barrier()
            
        ## updatign tqdm progress bar
        #train_bar.update(1) 

    cleanup()


# def run_demo(demo_fn, world_size):
#     mp.spawn(demo_fn,
#              args=(world_size,),
#              nprocs=world_size,
#              join=True)
    
if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = n_gpus
    dataset = Dataset()
    mp.spawn(demo_basic,
             args=(world_size,dataset,2),
             nprocs=world_size,
             join=True)
    #run_demo(demo_basic, world_size)    