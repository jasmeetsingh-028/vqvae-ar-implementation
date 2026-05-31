import torch
import torch.nn as nn
from transformer_docoder import TransformerDecoderBlock

class GPT(nn.Module): 
    def __init__(self, vocab_size = 513, seq_len = 65, d_model = 256,
                 n_heads = 8, n_layers = 6, d_feed_forward = 1024, 
                  dropout = 0.1): # 512 + BOS
        
        super().__init__()

        self.seq_len = seq_len

        #positional and token embeddings

        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.positional_embeddings = nn.Embedding(seq_len, d_model)
        self.dropout = nn.Dropout()

        self.transformer_decoder_blocks = nn.ModuleList([
                TransformerDecoderBlock(d_model,
                                        n_heads,
                                        d_feed_forward,
                                        dropout,
                                        seq_len
                                        )
             for _ in range(n_layers)]
        )

        # final layer norm and liner layer to go from d_model to vocab_size

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias = False)

        
        # weight tying — share weights between token embedding and output head
        # token_emb maps index → vector, 
        # head maps vector → index
        # tying them saves parameters and improves performance

        self.head.weight = self.token_embeddings.weight

        # initialize weights
        self.apply(self._init_weights)


    def _init_weights(self, module):
        # initializing weights according to what gpt used, with std = 0.02

        # Why std=0.02 specifically? 
        # Small enough that activations don't explode or vanish 
        # at the start of training. With 6 transformer blocks stacked, 
        # if initial weights are too large the signal either explodes or collapses 
        # by the time it reaches the last layer.

        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self, x):
        _, T = x.shape  # shape: (batch_size, seq_len = 65)

        #token + postional embeddings

        positions = torch.arange(T, device = x.device) #shape(seq_len = 65, )
        token_emb = self.token_embeddings(x)  # shape: (batch_size, seq_len) -> (batch_size, seq_len, d_model)
        positional_emb = self.positional_embeddings(positions) # shape: (seq_len = 65, ) -> (seq_len = 65, d_model)

        x = self.dropout(token_emb+ positional_emb) # shape: (batch_size, seq_len = 65 = T, d_model = 256)

        for block in self.transformer_decoder_blocks:
            x = block(x)

        #outputs

        x = self.norm(x)   # shape: (batch_size, seq_len = 65 = T, d_model = 256)
        logits = self.head(x)  # shape: (batch_size, seq_len = 65 = T, vocab_size = 513)

        return logits 


if __name__ == "__main__":
    x = torch.randint(0, 513, (4, 65)).long()  # batch of 4 sequences each of seq_len 65
    model = GPT(vocab_size=513,
                seq_len=65,
                d_model=256,
                n_heads=8,
                n_layers=6,
                d_feed_forward=1024,
                dropout=0.1)
    
    output_logits = model(x)
    print(output_logits.shape) 
