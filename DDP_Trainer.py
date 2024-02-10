
import os
import sys
import argparse
import tempfile
import warnings
import logging
from datetime import datetime
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from tinystories_dataset import TinyStories
from model import Model



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
    
    

def train(rank, world_size, dataset, config):
    print(f"Running on GPU{rank}.")
    setup(rank, world_size)

    # create model and move it to GPU with id rank
    model = Model(config).to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    loss_fn = nn.CrossEntropyLoss()  
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr = config.learning_rate, betas=(0.9, 0.95), weight_decay=0.1)
    ## get the dataloader
    train_dataloader = dataset.getTrainDataLoader(ddp=True)
    test_dataloader = dataset.getValDataLoader()
    print(f"Total number of steps in the datset on the GPU:{rank} is {len(train_dataloader)}.")
    ## set the tqdm bar
    train_bar = tqdm(total=len(train_dataloader), desc='Train Step', position=0, disable = not rank == 0)    
    for s,batch in enumerate(train_dataloader):        
        if s + 1 > config.max_iters: break ## check the maxiters condition \
        ddp_model.require_backward_grad_sync = (s + 1) % config.grad_accumulation_steps == 0
        #print(f"GPU{rank}: At step{s+1} weights of w11 and w12 is {model.net1.weight[0,:2]}")
        x,y = batch["inputs"].to(rank), batch["targets"].to(rank) 
        #print(f"GPU{rank}: At step{s+1} inputs shape:{x.shape} inputs{x[:2,:4]}")       
        outputs = ddp_model(x)        
        loss = loss_fn(outputs, y)
        print(f"GPU{rank}, step{s+1}: loss {loss}")
        loss = loss / config.grad_accumulation_steps ## normalizing
        #print(f"Loss on GPU{rank} for STEP{s} is: {loss}")
        loss.backward()
        #print(f"GPU{rank}: At step{s+1} gradients for w11 and w12 is {model.net1.weight.grad[0,:2]}")
        if (s + 1) % config.grad_accumulation_steps == 0:
            optimizer.step()
            #print(f"GPU{rank}: At step{s+1} weights of w11 and w12 is {model.net1.weight[0,:2]}")
            optimizer.zero_grad()
            
        # if rank == 0  and s % 200 == 0:
        #     val_loss = val_loop(ddp_model, test_dataloader, loss_fn, rank)
        #     print(f"***validation*** loss on GPU{rank} for STEP{s} is: {val_loss}")
        #dist.barrier()
            
        ## updatign tqdm progress bar
        train_bar.update(1) 

    cleanup()

def getArguments():
    ### Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_save_path', dest="model_save_path", type=str, required=True,
                        help="enter the the path to save the model with .pt extension")
    parser.add_argument('--max_iters', dest="max_iters", type=int, required=False,
                        help="maximum iteration of a model")
    parser.add_argument('--base_model_path', dest="base_model_path", type=str, required=False, default='NA',
                        help="existign model to train on")
    parser.add_argument('--loss_check_steps', dest="loss_check_steps", type=int, required=False, default=5000,
                        help="Numbers of steps after vlaidation metrcis calculated")
    parser.add_argument('--n_train_examples', dest="n_train_examples", type=int, required=False, default=-1,
                        help="number of training examples for subset")
    parser.add_argument('--n_val_examples', dest="n_val_examples", type=int, required=False, default=-1,
                        help="number of validation examples for subset")
    parser.add_argument('--tokenizer_path', dest="tokenizer_path", type=str, required=False, default="saved_artifacts/tokenizers",
                        help="path of the tokenizer")
    parser.add_argument('--wandb_project', dest="wandb_project", type=str, required=False, default="test_project",
                        help="wandb project name")
    parser.add_argument('--batch_size', dest="batch_size", type=int, required=False, default=16,
                        help="batch size (per process)")
    
    args = parser.parse_args()
    
    ## configuration settings
    class Config:
        vocab_size = 4096
        dim = 552
        n_heads = 12
        head_size = dim // n_heads
        n_layers = 12
        n_kv_heads = 3
        seq_len = 1024
        multiple_of = 256                
        batch_size = args.batch_size 
        global_batch_size = 150_000 # number of tokens per update
        world_size = torch.cuda.device_count()
        grad_accumulation_steps = int(global_batch_size/(seq_len * batch_size * world_size))
        learning_rate=5e-4
        total_params = 0
        tokenizer_path = args.tokenizer_path
        n_train_examples = args.n_train_examples
        n_val_examples = args.n_val_examples
        max_iters = args.max_iters
            
    config = Config()

    return config

    
if __name__ == "__main__":
    start_time = datetime.now()
    warnings.filterwarnings("ignore")
    ## setup logging
    file_name = os.path.join("saved_artifacts","logs",datetime.now().strftime('log_%Y_%m_%d_%H_%M_%S.log'))
    logging.basicConfig(level=logging.DEBUG, filename=file_name, filemode='w', format='%(name)s - %(levelname)s - %(message)s')
    ## read number of GPUS
    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"

    ## Read commandline arguments
    config = getArguments()
    
    ## Iintialize the datset
    dataset = TinyStories(config)
    ## Run the distributed training
    mp.spawn(train,
             args=(n_gpus,dataset,config),
             nprocs=n_gpus,
             join=True)
    ## log the run time
    run_time = datetime.now() - start_time
    logging.info(f"Total Execution duration: {run_time}")
      