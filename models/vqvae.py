import torch
import torch.nn as nn
from codebook import Codebook
from encoder import Encoder
from decoder import Decoder

class VQVAE(nn.Module):
    def __init__(self, in_channels = 3, hidden_dim = 128, latent_dim = 256, 
                 num_embeddings = 512, beta = 0.25):
        
        super().__init__()

        self.beta = beta # for commitment loss to encourage the encoder output to commit to the codebook embeddings
        
        self.encoder = Encoder(in_channels=in_channels,
                               hidden_dim=hidden_dim,
                               latent_dim=latent_dim)
        
        self.codebook = Codebook(num_embeddings=num_embeddings,
                                 latent_dim=latent_dim)
        
        self.decoder = Decoder(latent_dim=latent_dim,
                               hidden_dim=hidden_dim,
                                 out_channels=in_channels)
    
    def forward(self, x):
        z_e = self.encoder(x) # shape: (batch_size, 3, 32, 32) [Input image] -> (batch_size, 256, 8, 8) [encoder outputs]

        z_q, indices = self.codebook(z_e) # shape: (batch_Size, 256, 8, 8) [encoder outputs] -> (batch_size, 256, 8, 8) [quantized codebook vectors], indices shape: (batch_size, 8, 8) {index for each codebook vector}

        x_recon = self.decoder(z_q) # shape: (batch_size, 256, 8, 8) [quantized latent vectors] -> (batch_size, 3, 32, 32) [reconstructed image]

        # putting losses here to make it easier to compute the total loss in one step during training

        # reconstruction loss: MSE between input and reconstructed image
        recon_loss = nn.functional.mse_loss(x_recon, x)

        # codebook loss: push codebook entries towards encoder outputs
        # sg(z_e) means z_e is detached, only codebook updates 
        codebook_loss = nn.functional.mse_loss(z_e.detach(), z_q)

        # commitment loss: push encoder outputs towards codebook entries
        # sg(z_q) means z_q is detached, only encoder updates
        commitment_loss = nn.functional.mse_loss(z_e, z_q.detach())

        total_loss = recon_loss + codebook_loss + self.beta * commitment_loss

        return x_recon, total_loss, indices

if __name__ == "__main__":
    x = torch.randn(1, 3, 32, 32)
    model = VQVAE()
    x_recon, total_loss, indices = model(x)

    print("x_recon shape:", x_recon.shape)
    print("total_loss:", total_loss.item())
    print("indices shape:", indices.shape)

    print('-'*50)

    # exploring the model

    for name, module in model.named_modules():
        print(name, module)

    print('-'*50)

    # exploring model parameters
    for name, param in model.named_parameters():
        print(name, param.shape)