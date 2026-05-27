import torch
import torch.nn as nn

from encoder import Encoder
from codebook import Codebook

class Decoder(nn.Module):
    def __init__(self, latent_dim = 256, hidden_dim = 128, out_channels = 3):
        super().__init__()

        self.net = nn.Sequential(

            # input shape: (batch_size, latent_dim/hidden_dim = 256, 8, 8) -> (batch_size, 256*2, 8, 8) (identity layer to start with)
            nn.Conv2d(latent_dim, hidden_dim * 2, kernel_size=1),
            nn.ReLU(),

            #transpose convolution to upsample from (batch_size, 256*2, 8, 8) -> (batch_size, 256, 16, 16)
            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),

            #transpose convolution to upsample from (batch_size, 256, 16, 16) -> (batch_size, out_channels, 32, 32)
            nn.ConvTranspose2d(hidden_dim, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),  # output in range [-1, 1] matching normalized input
        )
    
    def forward(self, z_q):
        return self.net(z_q)


if __name__ == "__main__":
    x = torch.randn(1, 3, 32, 32)
    encoder = Encoder()
    codebook = Codebook()
    decoder = Decoder()

    z_e = encoder(x) # shape: (batch_size = 1, latent_dim = 256, 8, 8)
    print("z_e shape:", z_e.shape)

    z_q_st, indices = codebook(z_e) # z_q_st shape: (batch_size, latent_dim, 8, 8) = (1, 256, 8, 8), indices shape: (batch_size, 8, 8) = (1, 8, 8)
    print("z_q_st shape:", z_q_st.shape)
    print("indices shape:", indices.shape)

    x_recon = decoder(z_q_st) # shape: (batch_size, out_channels, 32, 32) = (1, 3, 32, 32)
    print("x_recon shape:", x_recon.shape)