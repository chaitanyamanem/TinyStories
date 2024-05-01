"""
This script is to inference the LLM models using single GPU or multiple-gpu or multi-node and multi-gpu
generate function is decop
"""

import os
import sys
import math
import argparse
import warnings
import logging
import wandb
import csv
from datetime import datetime
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from inference_dataset import TinyStories
from model import Model
from automodel import AutoModel
import sentencepiece as spm



## configuration settings
class Config:
    def __init__(self, args, world_size):
        self.vocab_size = 4096
        self.dim = 768
        self.n_heads = 8
        self.head_size = self.dim // self.n_heads
        self.n_layers = 8
        self.n_kv_heads = 8
        self.seq_len = 1024
        self.multiple_of = 256                
        self.batch_size = args.batch_size 
        self.global_batch_size = 150_000 # number of tokens per update
        self.world_size = world_size
        self.total_params = 0
        self.saved_checkpoint_path = args.saved_checkpoint_path
        self.tokenizer_path = args.tokenizer_path
        self.n_test_examples = args.n_test_examples        
        self.steps_to_serialize = args.steps_to_serialize
        self.rank = 0

class AverageMeter:
    def __init__(self, name):
        self.name = name
        self.reset()

    @torch.no_grad
    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0
        self.avg = 0

    @torch.no_grad
    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = int(self.sum / self.count)

    def __str__(self):
        return f"{self.name}: {self.avg}"
    
    @torch.no_grad
    def value(self):
        return self.avg  

def get_tokenizer(config):
    tokenizer_model_path = os.path.join(config.tokenizer_path,f"tok_{config.vocab_size}.model")
    tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_model_path)
    return tokenizer

def get_model(config, local_rank:int):
    config.rank = local_rank
    model = Model(config)
    model = AutoModel(model, config, device=local_rank)
    #checkpoint = torch.load(os.path.join(config.saved_checkpoint_path,"checkpoint.pt"))    
    # model = DDP(model, device_ids=[local_rank])
    if os.name != 'nt':
        model = torch.compile(model)
    return model

@torch.no_grad
def inference(dataloader, config):
    local_rank = int(os.environ["LOCAL_RANK"])
    rank =  int(os.environ["RANK"])    
    model = get_model(config, local_rank)
    # model.require_backward_grad_sync = False ## Just inferencing
    tokenizer = get_tokenizer(config)
    generation_config = {
        'padding_token': tokenizer.eos_id(),
        'bos_id': tokenizer.bos_id(),
        'max_new_tokens': 200,
        'temperature': 0.1
    }
    generated_tokens = []
    prompts = []
    
    token_rate = AverageMeter(name="token_rate")
    #tqdm setup
    total_steps = len(dataloader)
    inference_bar = tqdm(total=total_steps, desc='Inference Step', position=0, disable = not rank == 0) 
    

    for i, data in enumerate(dataloader):
        x = data["prompt"].to(local_rank)

        start_time = datetime.now()        
        inputs, outputs, tokens_count = model.generate(x,generation_config)
        duration = (datetime.now() - start_time).total_seconds()
        token_rate.update(tokens_count, duration)
        generated_tokens += outputs
        prompts += inputs
        
        
        inference_bar.write(f"Tokens generation speed GPU{local_rank}: {token_rate.value()} / second and per process")

    

        if i % config.steps_to_serialize == 0 or i == total_steps-1:
            prompts = tokenizer.decode(prompts)
            generations = tokenizer.decode(generated_tokens)
            result = [{'prompt': prompt, 'generation': generation} for prompt, generation in zip(prompts,generations)]
            keys = result[0].keys()
            file_name = 'saved_artifacts/generations/generations_{}_{}.csv'.format(str(i),str(local_rank))
            with open(file_name, 'w', newline='') as output_file:
                dict_writer = csv.DictWriter(output_file, keys)
                dict_writer.writeheader()
                dict_writer.writerows(result)

            ## reset token generation rate counters, and others
            generated_tokens = []
            prompts = []           

        inference_bar.update(1)


def setup():
    # initialize the process group
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup():
    dist.destroy_process_group()        

def main(config):
    setup()
    dataloader = TinyStories(config).getTestDataLoader(ddp=True)
    inference(dataloader, config)
    cleanup()
    
if __name__ == "__main__":
    start_time = datetime.now()
    warnings.filterwarnings("ignore")
    ## setup logging
    file_name = os.path.join("saved_artifacts","logs",datetime.now().strftime('log_%Y_%m_%d_%H_%M_%S.log'))
    logging.basicConfig(level=logging.DEBUG, filename=file_name, filemode='w', format='%(name)s - %(levelname)s - %(message)s')
    
    ## read number of GPUS
    world_size = int(os.environ["WORLD_SIZE"])

    ## Read commandline arguments
    ### Command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--saved_checkpoint_path', dest="saved_checkpoint_path", type=str, required=False,
                        default="saved_artifacts/models/model240221", help="existign model to inference with")
    parser.add_argument('--steps_to_serialize', dest="steps_to_serialize", type=int, required=False, default=5,
                        help="Numbers of steps after generated text should seriaize")    
    parser.add_argument('--n_test_examples', dest="n_test_examples", type=int, required=False, default=-1,
                        help="number of testing examples for subset")
    parser.add_argument('--tokenizer_path', dest="tokenizer_path", type=str, required=False, default="saved_artifacts/tokenizers",
                        help="path of the tokenizer")
    parser.add_argument('--wandb_project', dest="wandb_project", type=str, required=False, default="test_project",
                        help="wandb project name")
    parser.add_argument('--batch_size', dest="batch_size", type=int, required=False, default=16,
                        help="batch size (per process)")
    
    
    args = parser.parse_args()            
    config = Config(args, world_size)
    
    ## Run the distributed inference
    main(config)
    
    ## log the run time
    run_time = datetime.now() - start_time
    logging.info(f"Total Execution duration: {run_time}")
      
