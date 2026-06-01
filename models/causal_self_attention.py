import torch
import torch.nn as nn
import math

# since this is a deocder only model we need a causal mask

# Input shape: (batch_size, seq_length = 65, embedding dim or d_model = 256)

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model = 256, n_heads = 8, dropout = 0.1, seq_len = 65):
        super().__init__()

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        # d_model should be divisible as later q, k, v are split into multiple heads

        self.n_heads = n_heads
        self.d_model = d_model

        self.single_head_dim = d_model // n_heads # 256 // 8 = 32 for splitting into n_heads 

        # use 3*d_model to get qkv directly from one linear layer
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias = False)
        self.out_proj = nn.Linear(d_model, d_model, bias = False)
        self.dropout = nn.Dropout(dropout)

        # Causal mask to hide the future tokens so model makes prodiction only based on current and previous tokens
        # Causal mask will be an upper triangle filled with -inf
        # registered as buffer so it moves to correct device automatically

        mask = torch.tril(torch.ones(seq_len, seq_len))
        mask = mask.view(1, 1, seq_len, seq_len)
        self.register_buffer('mask', mask)
    
    def forward(self, x):
        #x shape: (batch_Size, seq_length, d_mdel/embedding_dim = 256)

        B, T, C = x.shape # B = batch_size, T = seq_length, C = d_model

        qkv = self.qkv_proj(x) # output shape: (batch_Size, seq_length, 3*d_model = 768)

        Q, K, V = qkv.split(self.d_model , dim = 2) # split the 3 *  d_model into equalal sized d_model

        # shape of q, k, v: (batch_Size, seq_length, d_model)
        # Further split each q, k, v for each of the 8 attention heads

        Q = Q.view(B, T, self.n_heads, self.single_head_dim).transpose(1, 2)
        K = K.view(B, T, self.n_heads, self.single_head_dim).transpose(1, 2)
        V = V.view(B, T, self.n_heads, self.single_head_dim).transpose(1, 2)

        # (batch_Size, seq_length, d_model) -> (batch_size, seq_length, n_heads = 8, single_head_dim = 32) -> (batch_size, n_heads = 8, seq_length = T, single_head_dim = 32)

        # Calculate attention scores = (q @ k.transpose) / sqrt(single_head dim / d_model in single head attention)
        scale = math.sqrt(self.single_head_dim)
        scores = Q @ K.transpose(-2, -1) # swap last two dims of K 
        # K's shape: (batch_size, n_heads = 8, seq_length, single_head_dim = 32) -> (batch_size, n_heads, single_head_dim, seq_length)
        # scores shape: (batch_size, n_heads = 8, seq_length = T, seq_length = T)

        # masking future tokens
    
        scores = scores.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf')) 

        # softmax + dropout
        attn_scores = torch.softmax(scores, dim = -1)
        attn_scores = self.dropout(attn_scores)

        # weighted sum with values
        out = attn_scores @ V    

        # attn_score shape (batch_size, n_heads, T, T) , Vshape: ((batch_size, n_heads = 8, seq_length = T, single_head_dim = 32)

        # out shape: (batch_size, n_heads = 8, seq_len = T, single_head_dim = 32)
        # (T, T) @ (T, 32) = (T, 32)


        # concat attention across all heads

        out = out.transpose(1, 2).contiguous() # shape: (batch_size, seq_len = T, n_heads = 8, single_head_dim = 32)
        out = out.view(B, T, self.d_model) # shape: (batch_size, seq_length = T, dmodel = 256)
        out = self.out_proj(out) # shape: (batch_size, seq_length = T, dmodel = 256)

        return out
    

if __name__ == "__main__":
    x = torch.randn(1, 65, 256)
    attention = CausalSelfAttention(d_model=256, n_heads=8, seq_len=65)
    out = attention(x)
    print(f'Input shape: {x.shape}')
    print(f'output shape: {out.shape}')






