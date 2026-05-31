import torch
import torch.nn as nn
from causal_self_attention import CausalSelfAttention

class TransformerDecoderBlock(nn.Module):
    def __init__(self, d_model = 256, n_heads = 8, d_feed_forward = 1024, dropout = 0.1, seq_length = 65):
        super().__init__()

        self.layer_norm1 = nn.LayerNorm(d_model)
        self.attention = CausalSelfAttention(d_model, n_heads, dropout, seq_length)

        self.layer_norm2 = nn.LayerNorm(d_model)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_feed_forward),
            nn.GELU(),
            nn.Linear(d_feed_forward, d_model),
            nn.Dropout(dropout),
        )

    
    def forward(self, x):

        x = x + self.attention(x) # shape (batch_size, seq_length = 65, d_model = 256)

        x = x + self.feed_forward(x)  # shape (batch_size, seq_length = 65, d_model = 256)

        return x



if __name__ == "__main__":
    block = TransformerDecoderBlock(d_model=256, n_heads=8, d_feed_forward=1024, seq_length=65)
    x     = torch.randn(4, 65, 256)
    out   = block(x)
    print(out.shape)


