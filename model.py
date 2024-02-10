import torch
from torch import nn
import numpy as np



class RoPE(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.cos_mthetas = None
        self.sin_mthetas = None
        self.__set_thetas()
        
    def __set_thetas(self):
        """This sets the parameters of the rope as per the formula
        Θ = {θi = 10000−2(i−1)/d, i ∈ [1, 2, ..., d/2]}
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
        """ assumes x in the format of """
        d  = x.shape[-1]
        
        assert d == self.d
        
        x = self.cos_mthetas * x + self.sin_mthetas * torch.cat([-1 * x[:,:,:,self.d//2:],x[:,:,:,:self.d//2]], axis=-1)
        return x        
    
    
    
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_rep = self.config.n_heads // self.config.n_kv_heads
        self.query = nn.Linear(self.config.dim, self.config.n_heads * self.config.head_size, bias=False)
        self.key = nn.Linear(self.config.dim, self.config.n_kv_heads * self.config.head_size, bias=False)
        self.value = nn.Linear(self.config.dim, self.config.n_kv_heads * self.config.head_size, bias=False)
        self.rope = RoPE(self.config)
        self.proj = nn.Linear(self.config.n_heads * self.config.head_size, self.config.n_heads * self.config.head_size, bias=False)
        
    def forward(self, x, y=None):
        b,t,d = x.shape
        q = self.query(x).view(b,t,self.config.n_heads,self.config.head_size)
        k = self.key(x).view(b,t,self.config.n_kv_heads,self.config.head_size)
        v = self.value(x).view(b,t,self.config.n_kv_heads,self.config.head_size)
        
        
        ## Add rotary embeddings        
        q = self.rope(q.permute(0,2,1,3)).permute(0,2,1,3)
        k = self.rope(k.permute(0,2,1,3)).permute(0,2,1,3)
        
        ##GQA 
        k = repeat_kv(k,self.n_rep)
        v = repeat_kv(v,self.n_rep)
        
        # make heads into a batch dimension
        q = q.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        ## Flash attention
        x = nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        
         # restore time as batch dimension and concat heads
        x = x.transpose(1, 2).contiguous().view(b, t, -1)

        # final projection into the residual stream
        x = self.proj(x)
        
        return x        
    
    
class FeedForwordNetwork(nn.Module):
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
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.mha = MultiHeadAttention(self.config)
        self.ffn = FeedForwordNetwork(self.config)
        self.anorm = RMSNorm(config)
        self.fnorm = RMSNorm(config)
        
    def forward(self,x):
        x = x + self.mha(self.anorm(x))
        x = x + self.ffn(self.fnorm(x))
        return x

class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding_layer = nn.Embedding(self.config.vocab_size, self.config.dim)
        self.layers = nn.Sequential(*[AttentionLayer(self.config) for _ in range(self.config.n_layers)])
        self.hnorm = RMSNorm(config)
        self.clf_head = nn.Linear(self.config.dim, self.config.vocab_size)
    
    def forward(self, x, y=None):
        x = self.embedding_layer(x)
        x = self.layers(x)
        x = self.hnorm(x)
        x = self.clf_head(x)
        return x

        


    

