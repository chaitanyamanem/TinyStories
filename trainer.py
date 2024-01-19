from model import Model
import torch
from torch import nn
from dataset import CharDataset
from tinystories_dataset import getTrainDataLoader, getValDataLoader, getVocabSize
from torch.utils.data.dataloader import DataLoader
import argparse
from tqdm import tqdm
import pickle
import os


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
    def __init__(self, patience=3, delta=0, mode='min'):
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
                self.msg = "Training improved"
            else:
                self.counter += 1
                self.msg = f"Training Not improved for the conjugutive step {self.counter}"
        else:
            raise Exception(f"given mode: `{self.mode}` not valid")
        return True if self.counter >= self.patience else False, self.msg
        






## training function
def train_loop(dataloader, model, loss_fn, optimizer, grad_accumulation_steps = 1, 
               max_iters = None, val_data = None, callback=None, model_save_path=None):
    model.train()
    total_steps = max_iters if max_iters is not None else len(dataloader)
    train_bar = tqdm(total=total_steps, desc='Train Step', position=0)
    train_loss = AverageMeter(name="train_loss")
    val_loss = AverageMeter(name="val_loss")
    history = {'train_loss':[], 'val_loss':[]}
    stop_training = False

    for batch, data in enumerate(dataloader):
        if (max_iters is not None and batch > max_iters) or stop_training:
            break        
        X, y = data["inputs"].to(device), data["targets"].to(device)
        pred = model(X)
        loss = loss_fn(torch.permute(pred, (0,-1,-2)), y)       

        

        ## Train and test evaluation
        if (batch) % 100 == 0:            
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
    parser.add_argument('--device', dest="device", type=str, required=False, default='cuda',
                        help="model to run on the device")      
    args = parser.parse_args()
    model_save_path = args.model_save_path
    max_iters = args.max_iters
    device = args.device



    ## configuration settings
    class Config:
        def __init__(self):
            self.embedding_dim = 512            
            self.head_size = 0
            self.block_size = 512 #context length
            self.vocab_size = 0
            self.epochs = 2
            self.n_heads = 8
            self.n_blocks = 8 #number of layers
            self.batch_size = 4
            self.grad_accumulation_steps = 16

    config = Config()




    ### Prepare and load data

    # _, train_data_loader = getTrainDataLoader(batch_size=config.batch_size,from_disk=True, 
    #                                           path="saved_artifacts/datasets/train_data")
    # _, val_data_loader = getValDataLoader(batch_size=config.batch_size, from_disk=True,
    #                                       path="saved_artifacts/datasets/val_data")
    _, train_data_loader = getTrainDataLoader(batch_size=config.batch_size, block_size=512, subset_size=2000)
    _, val_data_loader = getValDataLoader(batch_size=config.batch_size, block_size=512, subset_size=1000)     

    ## change required configuration settings
    config.vocab_size = getVocabSize()
    config.head_size = config.embedding_dim // config.n_heads

    ## create the model
    model = Model(config.vocab_size, config.embedding_dim, config.block_size, 
                  config.head_size, config.n_heads, config.n_blocks)
    model = model.to(device)
    ## Trainign loop
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    early_stopping = EarlyStopping(patience=5, mode='min')
    ## Train the data
    
    # for t in range(config.epochs):
    print(f"\n Training \n-------------------------------")
    history = train_loop(train_data_loader, model, loss_fn, optimizer, config.grad_accumulation_steps, 
                         max_iters = max_iters, val_data = val_data_loader, callback = early_stopping,
                         model_save_path=model_save_path)
    #test_loop(val_data_loader, model, loss_fn)
    print("Done!")
    #torch.save(model, model_save_path)
    loss_data_save_path =  model_save_path[:model_save_path.rfind('/')]    
    with open(os.path.join(loss_data_save_path,"loss_data.pkl"), 'wb') as f:
        pickle.dump(history, f)
    # save the model
    