import torch
import argparse
import tqdm
import os
import pickle
from model import Model
import sentencepiece as spm
from datetime import datetime


class Config:
    def __init__(self):
        self.vocab_size = 4096
        self.dim = 768
        self.n_heads = 8
        self.head_size = self.dim // self.n_heads
        self.n_layers = 8
        self.n_kv_heads = 8
        self.seq_len = 1024
        self.multiple_of = 256                
        self.batch_size = 16
        self.global_batch_size = 150_000 # number of tokens per update
        self.world_size = torch.cuda.device_count()
        self.grad_accumulation_steps = int(self.global_batch_size/(self.seq_len * self.batch_size * self.world_size))
        self.grad_clip = 1.0
        self.learning_rate = 5e-4
        self.min_lr = self.learning_rate / 10
        self.warmup_steps = 1000
        self.total_params = 0
        self.tokenizer_path = "saved_artifacts/tokenizers"        
        self.rank = 0
        self.enable_kv_cache = False


config = Config()
model = Model(config)    
checkpoint = torch.load("saved_artifacts/models/model240221/checkpoint.pt")
## load model state dict
model.load_state_dict(checkpoint['model'])
model = model.to("cuda")
model.eval()
## Tokenizer
tokenizer = spm.SentencePieceProcessor(model_file='saved_artifacts/tokenizers/tok_4096.model')        



def generate(prompt, max_new_tokens=300, temperature=0.0):
    model.reset_kv_cache()
    idx = torch.tensor(tokenizer.encode([prompt], add_bos=True), dtype=torch.long).to("cuda")
    t = 0
    start_time = datetime.now()    
    gen_loop = tqdm.tqdm(total=max_new_tokens, desc="gen_progress")
    
    while t <= max_new_tokens and int(idx[:,-1].item()) != int(tokenizer.bos_id()):
        #print(f"Timestep: {t}")
        if config.enable_kv_cache and t != 0:
            prompt = torch.unsqueeze(idx[:,-1], dim=-1)
        else:
            prompt = idx[:,-config.seq_len:]                
            
        #print(f"Shape of the prompt: {prompt.shape}")
        #print(f"prompt: {prompt}")
            
        pred = model(prompt)
        logits = pred[:,-1,:]

        if temperature == 0.0:
            # "sample" the single most likely index
            _, logits = torch.topk(logits, k=1, dim=-1)
        else:
            # pluck the logits at the final step and scale by desired temperature
            logits = logits / temperature

        #logits = top_k_top_p_filtering(logits, top_p=top_p, top_k=top_k)


        logits = torch.softmax(logits, axis=-1)
        next_idx = torch.multinomial(logits, num_samples=1)        
        idx = torch.cat([idx, next_idx], axis=1)
        t += 1
        gen_loop.update(1)
    
    tokens_rate = int(t / ((datetime.now() - start_time).total_seconds()))
    print(f"token generation rate per second: {tokens_rate}")
    generated_text = tokenizer.decode(idx.to("cpu").numpy().tolist())
    gen_loop.write("\n###############################")
    gen_loop.write(generated_text[0])
    gen_loop.write("\n###############################")


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model", required = False, default="saved_artifacts/models/model240221/checkpoint.pt", help="path of the model")
    parser.add_argument("--max_new_tokens", required=False, default=1000, help="Number of new tokens to generate by the model")
    parser.add_argument("--prompt", required=True, help="prompt to begain the generation with")      
    parser.add_argument("--temperature", required= False, default=0.1, type=float) 
    args = parser.parse_args()
    
    generate(args.prompt, args.max_new_tokens, args.temperature)
    
