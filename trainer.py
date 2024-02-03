from model import Model
import torch
from torch import nn
#from dataset import CharDataset
from tinystories_dataset import TinyStories
from torch.utils.data.dataloader import DataLoader
import argparse
from tqdm import tqdm
import pickle
import os
import wandb


class AverageMeter:
    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0
        self.avg = 0
        
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    def __str__(self):
        return f"{self.name}: {self.avg}"
    
    def value(self):
        return self.avg

class EarlyStopping:
    def __init__(self, patience=0, delta=0, mode='min'):
        self.best_val = float("inf")
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.counter = 0
        self.msg = None

    def check(self, metric_val:float):
        if self.mode == 'min':
            if metric_val <= self.best_val:
                self.best_val = metric_val
                self.counter = 0
                self.msg = "Training improved"
            else:
                self.counter += 1
                self.msg = f"Training Not improved for the conjugutive step {self.counter}"
        else:
            raise Exception(f"given mode: `{self.mode}` not valid")
        return True if self.counter > self.patience else False, self.msg
        


class Dataset:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = iter(self.dataloader)
    def getBatch(self):
        try:
            batch = next(self.iterator)
        except:
            self.iterator = iter(self.dataloader)
            batch = next(self.iterator)
        return batch




## training function
def train_loop(dataloader, model, loss_fn, optimizer, grad_accumulation_steps = 1, val_steps = 5000,
               max_iters = None, val_data = None, callback=None, model_save_path=None):
    
    model.train()
    dataset = Dataset(dataloader)
    total_steps = max_iters if max_iters is not None else len(dataloader)
    train_bar = tqdm(total=total_steps, desc='Train Step', position=0)
    train_loss = AverageMeter(name="train_loss")
    val_loss = AverageMeter(name="val_loss")
    history = {'train_loss':[], 'val_loss':[]}
    stop_training = False

    for batch in range(total_steps):
        if batch > total_steps or stop_training:
            break
        data = dataset.getBatch()
        X, y = data["inputs"].to(device), data["targets"].to(device)
        pred = model(X)
        loss = loss_fn(torch.permute(pred, (0,-1,-2)), y)       

        

        ## Train and test evaluation
        if (batch) % val_steps == 0:            
            # Training loss
            train_loss.update(loss.item(), X.shape[0])
            history['train_loss'].append(train_loss.value())
            # Validation loss
            test_loop(val_data, model, loss_fn, val_loss)
            history['val_loss'].append(val_loss.value())
            train_bar.write(f"Step:{batch} {train_loss}, {val_loss}")            
            stop_training, msg = callback.check(val_loss.value())            
            # save the model
            torch.save(model, model_save_path)
            ## Log on wandb
            wandb.log({'train/loss':train_loss.value(), 'val/loss':val_loss.value()})
                
        #backpropagation
        loss.backward()
        if ((batch + 1) % grad_accumulation_steps == 0) or (batch + 1) == total_steps:            
            optimizer.step()
            optimizer.zero_grad()

        train_bar.update(1)

    return history


## Test function
@torch.no_grad
def test_loop(dataloader, model, loss_fn, val_loss):
    model.eval()
    val_loss.reset()    
    num_batches = len(dataloader)    
    
    test_bar = tqdm(total=num_batches, desc='val loss step', position=1, leave=False) 
    for i, data in enumerate(dataloader):        
        X,y = data["inputs"].to(device), data["targets"].to(device)
        pred = model(X)
        loss = loss_fn(torch.permute(pred,(0,-1,-2)), y)
        val_loss.update(loss.item(), X.shape[0])
        test_bar.update(1) # for tqdm progress bar
    model.train()
    return


@torch.no_grad
def eval_loss(model, dataloader, loss_fn, eval_iters=5):
    model.eval()
    losses = torch.ones(eval_iters)
    for i in range(eval_iters):
        data = next(iter(dataloader))
        X,y = data["inputs"].to(device), data["targets"].to(device)
        pred = model(X)
        loss = loss_fn(torch.permute(pred,(0,-1,-2)), y)
        losses[i] = loss
    model.train()
    return losses.mean().item()



#######  Main funciton ###############
if __name__ == '__main__':
    ### Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_save_path', dest="model_save_path", type=str, required=True,
                        help="enter the the path to save the model with .pt extension")
    parser.add_argument('--max_iters', dest="max_iters", type=int, required=False,
                        help="maximum iteration of a model")
    parser.add_argument('--base_model_path', dest="base_model_path", type=str, required=False, default='NA',
                        help="existign model to train on")
    parser.add_argument('--device', dest="device", type=str, required=False, default='cuda',
                        help="model to run on the device")
    parser.add_argument('--val_steps', dest="val_steps", type=int, required=False, default=5000,
                        help="Numbers of steps after vlaidation metrcis calculated")
    parser.add_argument('--n_train_examples', dest="n_train_examples", type=int, required=False, default=-1,
                        help="number of training examples for subset")
    parser.add_argument('--n_val_examples', dest="n_val_examples", type=int, required=False, default=-1,
                        help="number of validation examples for subset")
    parser.add_argument('--tokenizer_path', dest="tokenizer_path", type=str, required=False, default="saved_artifacts/tokenizers",
                        help="path of the tokenizer")
    parser.add_argument('--wandb_project', dest="wandb_project", type=str, required=False, default="test_project",
                        help="wandb project name")
    parser.add_argument('--batch_size', dest="batch_size", type=int, required=False, default=4,
                        help="batch_size")
    
    args = parser.parse_args()
    model_save_path = args.model_save_path
    max_iters = args.max_iters
    base_model_path = args.base_model_path
    device = args.device
    val_steps = args.val_steps
    n_train_examples = args.n_train_examples
    n_val_examples = args.n_val_examples
    tokenizer_path = args.tokenizer_path
    wandb_project = args.wandb_project
    batch_size = args.batch_size

    



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
        batch_size = batch_size  
        global_batch_size = 100000
        grad_accumulation_steps = int(global_batch_size / (batch_size * seq_len))
        learning_rate=5e-4
        total_params = 0
        tokenizer_path = tokenizer_path
            
    config = Config()




    ### Prepare and load data

    # _, train_data_loader = getTrainDataLoader(batch_size=config.batch_size,from_disk=True, 
    #                                           path="saved_artifacts/datasets/train_data")
    # _, val_data_loader = getValDataLoader(batch_size=config.batch_size, from_disk=True,
    #                                       path="saved_artifacts/datasets/val_data")
    print("------Beginning the data preparation----")
    tinystories = TinyStories(config.vocab_size, config.seq_len, config.tokenizer_path)
    _, train_data_loader = tinystories.getTrainDataLoader(batch_size=config.batch_size, subset_size=n_train_examples)
    _, val_data_loader = tinystories.getValDataLoader(batch_size=config.batch_size, subset_size=n_val_examples) 
    print(f"\n#########Length of the training data:{len(train_data_loader)}, validation data:{len(val_data_loader)}")    
    print("------End of data preparation----")    

    if base_model_path == 'NA':
        ## create the model
        model = Model(config)
    else:
        model = torch.load(base_model_path)
    model = model.to(device)
    
    ## Log Trainable parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    config.total_params = total_params
    print(f"\n#######Total parameter of the model: {total_params * 1e-6}")

    ## intiate the wandb logging
    run = wandb.init(
        project = wandb_project,
        config = config.__dict__
    )

    ## Trainign configuration
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr = config.learning_rate, betas=(0.9, 0.95), weight_decay=0.1)
    early_stopping = EarlyStopping(patience=3, mode='min')

    ## Load the optimizer state of the training is resumed
    if base_model_path != 'NA':
        PATH = base_model_path[0:base_model_path.rfind('/')]
        checkpoint = torch.load(os.path.join(PATH,'checkpoint.pth'))
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    ## Train the model
    
    # for t in range(config.epochs):
    print(f"\n Training \n-------------------------------")
    history = train_loop(train_data_loader, model, loss_fn, optimizer, config.grad_accumulation_steps, 
                         val_steps=val_steps, max_iters = max_iters, val_data = val_data_loader, callback = early_stopping,
                         model_save_path=model_save_path)
    #test_loop(val_data_loader, model, loss_fn)
    print("Done!")
    ## Save model and optimizer state dict
    root_save_path =  model_save_path[:model_save_path.rfind('/')]    
    torch.save(model, model_save_path) ## savign after last update
    torch.save({'optimizer_state_dict': optimizer.state_dict()},os.path.join(root_save_path,'checkpoint.pth'))
    
    with open(os.path.join(root_save_path,"loss_data.pkl"), 'wb') as f:
        pickle.dump(history, f)
    # save the model
    