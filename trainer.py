from model import Model
import torch
from torch import nn
from dataset import CharDataset
from tinystories_dataset import getTrainDataLoader, getValDataLoader, getVocabSize
from torch.utils.data.dataloader import DataLoader
import argparse
from tqdm import tqdm


## training function
def train_loop(dataloader, model, loss_fn, optimizer, max_iters = None, val_data = None):
    model.train()
    total_steps = max_iters if max_iters is not None else len(dataloader)
    train_bar = tqdm(total=total_steps, desc='Train Step', position=0)
    for batch, data in enumerate(dataloader):
        if max_iters is not None and batch > max_iters:
            break        
        X, y = data["inputs"].to(device), data["targets"].to(device)
        pred = model(X)
        loss = loss_fn(torch.permute(pred, (0,-1,-2)), y)
        #print(f"step:{batch}, loss:{loss.item()}")

        ## Train and test evaluation
        if (batch) % 100 == 0:
            #print("Entered into evaluation block")
            train_loss = eval_loss(model, dataloader, loss_fn)
            #print("Train evaluation done! going to test evaluation")
            val_loss = test_loop(val_data, model, loss_fn)
            train_bar.write(f"Step:{batch} train loss:{loss.to('cpu').item()}, val loss:{val_loss.to('cpu').item()}")        
                
        #backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_bar.update(1)




## Test function
@torch.no_grad
def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    losses = torch.ones(num_batches)
    test_loss , correct = 0,0
    test_bar = tqdm(total=len(dataloader), desc='val loss step', position=1, leave=False) 
    for i, data in enumerate(dataloader):        
        X,y = data["inputs"].to(device), data["targets"].to(device)
        pred = model(X)
        losses[i] = loss_fn(torch.permute(pred,(0,-1,-2)), y)
        test_bar.update(1)
    model.train()
    return losses.mean()


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
    parser.add_argument('--model_save_path', dest="model_save_path", type=str,
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
            self.embedding_dim = 384
            self.n_heads = 6
            self.head_size = 512
            self.block_size = 256
            self.vocab_size = 65
            self.epochs = 5
            self.n_heads = 6
            self.n_blocks = 6

    config = Config()




    ### Prepare and load data

    _, train_data_loader = getTrainDataLoader(subset_size=2000)
    _, val_data_loader = getValDataLoader(subset_size=1000)    

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
    ## Train the data
    
    for t in range(config.epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_data_loader, model, loss_fn, optimizer, max_iters = max_iters, val_data = val_data_loader)
        #test_loop(val_data_loader, model, loss_fn)
    print("Done!")
    torch.save(model, model_save_path)