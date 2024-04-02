from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader, DistributedSampler
import sentencepiece as spm
import os


class TinyStories:
    def __init__(self, config):
        self.vocab_size = config.vocab_size
        self.context_length = config.seq_len
        self.batch_size = config.batch_size
        self.tokenizer_model_path = os.path.join(config.tokenizer_path,f"tok_{config.vocab_size}.model")
        self.tokenizer = spm.SentencePieceProcessor(model_file=self.tokenizer_model_path)

        #build dataset
        self.dataset = load_dataset("roneneldan/TinyStories")        
        self.test_dataset = self.dataset["validation"]
        
        if config.n_test_examples != -1:
            self.test_dataset = self.test_dataset.select(range(config.n_val_examples))

        self.test_dataset = self.test_dataset.map(
                    self.process_rows_func,                
                    batched=True,
                    num_proc=4,
                    remove_columns=self.test_dataset.column_names    
                )
        self.test_dataset.set_format(type='torch', columns=self.test_dataset.column_names)             

    
    def process_rows_func(self,examples):     
        idx = 0 
        p_len = 30
        inputs, targets = [],[]
        
        
        tokenized_examples = self.tokenizer.encode(examples["text"], add_bos=True)
        for example in tokenized_examples:        
            inputs.append(example[:p_len])

        return {"prompt":inputs}        




    def getTestDataLoader(self, ddp):
        sampler = DistributedSampler(self.test_dataset) if ddp else None
        shuffle = True if not ddp else None
        train_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=shuffle, sampler=sampler)
        return train_loader  
    
        
    def getVocabSize(self):
        return int(self.tokenizer.vocab_size())
    


