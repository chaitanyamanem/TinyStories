"""

"""
import os
import torch
from torch import nn

class AutoModel(nn.Module):
    def __init__(self, model, model_config, device:int):
        super().__init__()
        self.model = model
        self.checkpoint = model_config.saved_checkpoint_path
        self.device = device
        self.model_config = model_config
        self.model = self.__load_model()        
        
    def __load_model(self):
        checkpoint = torch.load(os.path.join(self.checkpoint,"checkpoint.pt"))
        self.model.load_state_dict(checkpoint['model'])
        print(f"Loading model to device: {self.device}")
        self.model = self.model.to(self.device)
        return self.model
    
    def forward(self, x):
        return self.model(x)

    def generate(self, x, generation_config):
        padding_token = generation_config['padding_token']
        prompts = []
        generated_tokens_batch = []
        total_tokens_count = 0
        for i in range(x.shape[0]):
            prompt = x[i,:] # get individual prompt in the batch
            #get the prompt, removign the paddign by getting 1st padding token position
            # pad_start = torch.argmax((prompt == padding_token).to(torch.long) , axis=-1)
            # prompt = prompt[: pad_start] # trim the padding
            prompts.append(prompt.tolist())
            prompt_len = len(prompt.tolist())
            prompt = prompt.unsqueeze(axis=0)
            generated_tokens = prompt.detach().clone()
            print(f"prompt shape: {prompt.shape}")

            if self.model_config.enable_kv_cache:
                ## Reset kv_cache for every exampel if is enabled.
                self.model.reset_kv_cache()

            for step in range(generation_config["max_new_tokens"]):

                prompt = prompt[:,-self.model_config.seq_len:] #2 dimension
                logits = self.model(prompt)[:,-1,:] # 2 dimension
                if generation_config["temperature"] != 0.0:
                    logits = logits / generation_config["temperature"]
                logits = torch.softmax(logits, axis=-1)
                next_idx = torch.multinomial(logits, num_samples=1)
                generated_tokens = torch.cat([generated_tokens,next_idx], axis=1)
                prompt = next_idx if self.model_config.enable_kv_cache else torch.cat([prompt, next_idx], axis=1)
                     
                if next_idx == generation_config["bos_id"]:
                    break
            
            new_tokens = step+1
            generated_tokens_batch.append(generated_tokens[0].tolist())
            total_tokens_count += new_tokens
            

        return prompts, generated_tokens_batch, total_tokens_count
