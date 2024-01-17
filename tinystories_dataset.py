from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader

from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def process_rows_func(examples, block_size, tokenizer):     
    idx = 0 
    inputs, targets = [],[]
    
    
    tokenized_examples = tokenizer(examples["text"])["input_ids"]
    merged_text = [token for example in tokenized_examples for token in example+[tokenizer.eos_token_id,]]
    
    while idx < len(merged_text)-block_size:
        chunk = merged_text[idx:idx+block_size+1]
        inputs.append(chunk[:-1])
        targets.append(chunk[1:])
        idx += block_size
    return {"inputs":inputs, "targets":targets}



def getTrainDataLoader(batch_size = 64, block_size = 256, from_disk=False, 
                       path=None, subset_size=-1, save_to_disk=False, save_path=None):
    if from_disk:
        if path is None:
            raise Exception("To load the dataset from disk, you need to give valid path")
        reloaded_dataset = load_from_disk(path).with_format("torch")
        train_loader = DataLoader(reloaded_dataset, batch_size=batch_size)
        return reloaded_dataset, train_loader
    else:
        dataset = load_dataset("roneneldan/TinyStories")
        dataset = dataset["train"]
        if subset_size != -1:
            if subset_size <=0: 
                raise Exception("sample size should be a positive number larger than zero")
            dataset = dataset.select(range(subset_size))
        
        dataset = dataset.map(
            process_rows_func,
            fn_kwargs={'block_size':block_size, 'tokenizer':tokenizer},
            batched=True,
            num_proc=4,
            remove_columns=dataset.column_names    
        )
        
        if save_to_disk:
            if save_path is None: raise Exception("save path can't be None")
            dataset.save_to_disk(save_path)
            
        dataset.set_format(type='torch', columns=dataset.column_names)
        train_loader = DataLoader(dataset, batch_size=batch_size)
        
    
        
        return dataset, train_loader  
    
    

def getValDataLoader(batch_size = 64, block_size = 256, from_disk=False, 
                       path=None, subset_size=-1, save_to_disk=False, save_path=None):
    if from_disk:
        if path is None:
            raise Exception("To load the dataset from disk, you need to give valid path")
        reloaded_dataset = load_from_disk(path).with_format("torch")
        val_loader = DataLoader(reloaded_dataset, batch_size=batch_size)
        return reloaded_dataset, val_loader
    else:
        dataset = load_dataset("roneneldan/TinyStories")
        dataset = dataset["validation"]
        if subset_size != -1:
            if subset_size <=0: 
                raise Exception("sample size should be a positive number larger than zero")
            dataset = dataset.select(range(subset_size))
        
        dataset = dataset.map(
            process_rows_func,
            fn_kwargs={'block_size':block_size, 'tokenizer':tokenizer},
            batched=True,
            num_proc=4,
            remove_columns=dataset.column_names    
        )
        
        if save_to_disk:
            if save_path is None: raise Exception("save path can't be None")
            dataset.save_to_disk(save_path)
            
        dataset.set_format(type='torch', columns=dataset.column_names)
        val_loader = DataLoader(dataset, batch_size=batch_size)
        
    
        
        return dataset, val_loader
def getVocabSize():
    return int(tokenizer.vocab_size)
    
