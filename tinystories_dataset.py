from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader, DistributedSampler
import sentencepiece as spm
import os
from accelerate import Accelerator

class TinyStories:
    def __init__(self, config):
        self.vocab_size = config.vocab_size
        self.context_length = config.seq_len
        self.batch_size = config.batch_size
        self.tokenizer_model_path = os.path.join(config.tokenizer_path,f"tok_{config.vocab_size}.model")
        self.tokenizer = spm.SentencePieceProcessor(model_file=self.tokenizer_model_path)

        #build dataset
        self.dataset = load_dataset("roneneldan/TinyStories")
        self.train_dataset = self.dataset["train"]
        self.val_dataset = self.dataset["validation"]
        if config.n_train_examples != -1:
            self.train_dataset = self.train_dataset.select(range(config.n_train_examples))
        if config.n_val_examples != -1:
            self.val_dataset = self.val_dataset.select(range(config.n_val_examples))

        self.train_dataset = self.train_dataset.map(
                    self.process_rows_func,                
                    batched=True,
                    num_proc=4,
                    remove_columns=self.train_dataset.column_names    
                )
        self.train_dataset.set_format(type='torch', columns=self.train_dataset.column_names)

        self.val_dataset = self.val_dataset.map(
                    self.process_rows_func,                
                    batched=True,
                    num_proc=4,
                    remove_columns=self.val_dataset.column_names    
                )
        self.val_dataset.set_format(type='torch', columns=self.val_dataset.column_names)             

    
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



    def getTrainDataLoader(self, ddp):
        sampler = DistributedSampler(self.train_dataset) if ddp else None
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, sampler=sampler)
        return train_loader  
    
    

    def getValDataLoader(self): 
        val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True)    
        return val_loader
        
    def getVocabSize(self):
        return int(self.tokenizer.vocab_size())
    


