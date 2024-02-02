from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
import sentencepiece as spm
import os
from accelerate import Accelerator

class TinyStories:
    def __init__(self, vocab_size, context_length, tokenizers_path):
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.tokenizer_model_path = os.path.join(tokenizers_path,f"tok_{vocab_size}.model")
        self.tokenizer = spm.SentencePieceProcessor(model_file=self.tokenizer_model_path)

    
    def process_rows_func(self,examples):     
        idx = 0 
        inputs, targets = [],[]
        
        
        tokenized_examples = self.tokenizer.encode(examples["text"], add_bos=True)
        merged_text = [token for example in tokenized_examples for token in example]
        
        while idx < len(merged_text)-self.context_length:
            chunk = merged_text[idx:idx+self.context_length+1]
            inputs.append(chunk[:-1])
            targets.append(chunk[1:])
            idx += self.context_length
        return {"inputs":inputs, "targets":targets}



    def getTrainDataLoader(self, accelerator, batch_size = 64, from_disk=False, 
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
            
            with accelerator.main_process_first():
                dataset = dataset.map(
                    self.process_rows_func,                
                    batched=True,
                    num_proc=4,
                    remove_columns=dataset.column_names    
                )
            
            if save_to_disk:
                if save_path is None: raise Exception("save path can't be None")
                dataset.save_to_disk(save_path)
                
            dataset.set_format(type='torch', columns=dataset.column_names)
            train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
        
            
            return dataset, train_loader  
    
    

    def getValDataLoader(self, accelerator, batch_size = 64, from_disk=False, 
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
            
            with accelerator.main_process_first():
                dataset = dataset.map(
                    self.process_rows_func,                
                    batched=True,
                    num_proc=4,
                    remove_columns=dataset.column_names    
                )
            
            if save_to_disk:
                if save_path is None: raise Exception("save path can't be None")
                dataset.save_to_disk(save_path)
                
            dataset.set_format(type='torch', columns=dataset.column_names)
            val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True) 
        
            return dataset, val_loader
        
    def getVocabSize(self):
        return int(self.tokenizer.vocab_size())
    


