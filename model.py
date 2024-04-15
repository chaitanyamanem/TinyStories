import torch
from torch import nn
import numpy as np
import inspect



class RoPE(nn.Module):
    """
    *custom implementaiton*
    This class provides Rotary Positional Encodign for the input data (word embeddings)
    Usally applied on Q and V
    reference: https://arxiv.org/abs/2104.09864
    """
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.cos_mthetas = None
        self.sin_mthetas = None
        self.__set_thetas()
        
    def __set_thetas(self):
        """This sets the parameters of the rope as per the formula
        Θ = {θi = 10000−2(i−1)/d, i ∈ [1, 2, ..., d/2]}

        it also precomputes real (cos mtheta) and imaginary (sin mtheta) parts of the RoPE equation to further use in the 
        `forward` method

        reference: https://arxiv.org/abs/2104.09864
        """
        assert self.config.head_size % 2 == 0, f"Head size:{self.config.head_size} shouls be even number"
        self.d = self.config.head_size
        self.m = self.config.seq_len
        
        i_s = torch.tensor(range(1,self.d//2+1))
        i_s = torch.cat([i_s,i_s], axis=-1)
        m_s = torch.tensor(range(self.m)).unsqueeze(axis=-1)
        
        thetas = 10000 ** (-2*(i_s-1)/self.d)
        self.cos_mthetas = torch.cos(m_s * thetas).to(self.config.rank)
        self.sin_mthetas = torch.sin(m_s * thetas).to(self.config.rank)
        
    def forward(self, x):
        """ 
        Encodes positional information into context embeddings Q and V
        It expects the shape of the input `x` to be 4 dimensional
        x should be in the form x[b,h,t,s]
        here,   b - batch dimension
                h - number of heads
                t - time steps (number of words)
                s - size of the head

        reference: https://arxiv.org/abs/2104.09864                
        """
        d, t   = x.shape[-1], x.shape[-2]
        
        assert d == self.d
        
        x = self.cos_mthetas[:t,:] * x + self.sin_mthetas[:t,:] * torch.cat([-1 * x[:,:,:,self.d//2:],x[:,:,:,:self.d//2]], axis=-1)
        return x        
    
    
    
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    *custom implementaiton*
    This funciton is used for grouped query attention GQA.
    This replicates the K and V to match the dimension along the heads.
    Expected shape of the input(x) is
        x[b,t,h,s]
        here,   b - batch dimension
                t - time steps (number of words)
                h - number of heads                
                s - size of the head

    reference: https://arxiv.org/pdf/2305.13245
    """
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

class MultiHeadAttention(nn.Module):
    """
    This module is a multi head attention part of the Transformer.
    This computes the scaled dot product between the Q, K and V
    """
    def __init__(self, config, layer_id):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.n_rep = self.config.n_heads // self.config.n_kv_heads
        self.query = nn.Linear(self.config.dim, self.config.n_heads * self.config.head_size, bias=False)
        self.key = nn.Linear(self.config.dim, self.config.n_kv_heads * self.config.head_size, bias=False)
        self.value = nn.Linear(self.config.dim, self.config.n_kv_heads * self.config.head_size, bias=False)
        self.rope = RoPE(self.config)
        self.proj = nn.Linear(self.config.n_heads * self.config.head_size, self.config.n_heads * self.config.head_size, bias=False)
        
    def forward(self, x, y=None, enable_kv_cache=False, kv_cache=None):
        b,t,d = x.shape
        q = self.query(x).view(b,t,self.config.n_heads,self.config.head_size)
        k = self.key(x).view(b,t,self.config.n_kv_heads,self.config.head_size)
        v = self.value(x).view(b,t,self.config.n_kv_heads,self.config.head_size)

        ## concat k,v from cache
        if kv_cache[self.layer_id][0] is not None:
            #print(f"Detected existing cache for the layer {self.layer_id}")
            k = torch.cat((kv_cache[self.layer_id][0],k), axis=1)
            v = torch.cat((kv_cache[self.layer_id][1],v), axis=1)
            ## Makesure K and v is not more than model allowed context length
            k = k[:,-self.config.seq_len:, :, :]
            v = v[:, -self.config.seq_len:, :, :]        
        
        ## Add rotary embeddings        
        q = self.rope(q.permute(0,2,1,3)).permute(0,2,1,3) # (bs, n_local_heads, seqlen, head_dim)
        k = self.rope(k.permute(0,2,1,3)).permute(0,2,1,3)
        
        ##GQA 
        #k = repeat_kv(k,self.n_rep)
        #v = repeat_kv(v,self.n_rep)
        
        # make heads into a batch dimension
        q = q.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        
        #print(f"q and k shapes: {q.shape, k.shape}")
        ## Flash attention
        if enable_kv_cache:
            mask = torch.ones(q.size()[-2], k.size()[-2], dtype=torch.bool).to(k.device)
            x = nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False,attn_mask=mask)
        else:
            x = nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)

        ## add the new token to the cache
        if enable_kv_cache:
            kv_cache[self.layer_id] = (k.transpose(1, 2),v.transpose(1, 2)) # (bs, seqlen, n_local_heads, head_dim)
        
         # restore time as batch dimension and concat heads
        x = x.transpose(1, 2).contiguous().view(b, t, -1)

        # final projection into the residual stream
        x = self.proj(x)
        
        
        return x, kv_cache        
    
    
class FeedForwordNetwork(nn.Module):
    """
    This is a stack of linear layers and one SiLU activation
    All togather forms SwiGLU
    reference: https://arxiv.org/pdf/2002.05202
    """
    def __init__(self, config):
        super().__init__()
        self.config = config        
        hidden_units = int(self.config.dim * 4 * 2/3) ##
        hidden_units = int(hidden_units - (hidden_units % self.config.multiple_of) + self.config.multiple_of)
        
        self.w = nn.Linear(self.config.dim, hidden_units)
        self.v = nn.Linear(self.config.dim, hidden_units)
        self.w2 = nn.Linear(hidden_units, self.config.dim)
        self.silu = nn.SiLU()
        
    def forward(self, x):
        out = self.w2(self.silu(self.w(x)) * self.v(x))
        return out
    
class RMSNorm(nn.Module):
    """
    *custom implementaiton*
    Similar to Layer norm but with only scaling and omiiting the centering part.
    reference: https://arxiv.org/abs/1910.07467
    """
    def __init__(self, config):
        super().__init__()
        self.eps = 1e-5
        self.dim = config.dim
        self.weight = nn.Parameter(torch.ones(self.dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
    
class AttentionLayer(nn.Module):
    """
    This is equal to one block or layer in the transformer, comprises of 
    RMSNorm
    Residual connection
    Multi Head Attention
    feedforword block
    """
    def __init__(self, config, i):
        super().__init__()
        self.config = config
        self.layer_id = i
        self.mha = MultiHeadAttention(self.config, self.layer_id)
        self.ffn = FeedForwordNetwork(self.config)
        self.anorm = RMSNorm(config)
        self.fnorm = RMSNorm(config)
        
        
    def forward(self, inputs:tuple):
        x, enable_kv_cache, kv_cache = inputs        
        x_new, kv_cache = self.mha(self.anorm(x), enable_kv_cache=enable_kv_cache, kv_cache=kv_cache)
        x = x + x_new
        x = x + self.ffn(self.fnorm(x))
        return (x, enable_kv_cache, kv_cache)

class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.enable_kv_cache = config.enable_kv_cache #boolean value
        self.kv_cache = [(None,None)   for _ in range(self.config.n_layers)]
        self.embedding_layer = nn.Embedding(self.config.vocab_size, self.config.dim)
        self.layers = nn.Sequential(*[AttentionLayer(self.config, i) for i in range(self.config.n_layers)])
        self.hnorm = RMSNorm(config)
        self.clf_head = nn.Linear(self.config.dim, self.config.vocab_size)
    
    def forward(self, x, y=None):
        x = self.embedding_layer(x)
        x, _, self.kv_cache = self.layers((x, self.enable_kv_cache, self.kv_cache))
        x = self.hnorm(x)
        x = self.clf_head(x)
        # if self.enable_kv_cache:
        #     return x, self.kv_cache
        # else:
        return x
        
    def reset_kv_cache(self):
        self.kv_cache = [(None,None)   for _ in range(self.config.n_layers)]
    
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """
        This create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        """
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

        


    

