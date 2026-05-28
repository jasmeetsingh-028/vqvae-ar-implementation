import torch
import torch.nn as nn
from models.encoder import Encoder

class Codebook(nn.Module):
    def __init__(self, num_embeddings = 512, latent_dim = 256):  ## num_embeddings was 8192 for 256*256*3 images but we are using 32*32*3 images so we can reduce it to 512
        super().__init__()

        self.num_embeddings = num_embeddings
        self.latent_dim = latent_dim

        # the learnable lookup table (512, 256)
        self.codebook = nn.Embedding(num_embeddings, latent_dim)

        # initialize the codebook embeddings uniformly in the range -1/num_embeddings to 1/num_embeddings
        nn.init.uniform_(self.codebook.weight, -1/num_embeddings, 1/num_embeddings)


    def forward(self, z_e):
        # z_e is of shape: (batch_size, latent_dim = 256, 8, 8)
        # reshape z_e to (batch_size*8*8, latent_dim)

        # 1. permute the channel dim to last: (batch_size, 8, 8, latent_dim)
        z_e = z_e.permute(0, 2, 3, 1).contiguous() 
        # Shape:  (batch_size, 8, 8, latent_dim)

        #print("z_e shape after permute:", z_e.shape)
    
        # 2. reshape to (batch_size*8*8, latent_dim)
        flat_z_e = z_e.view(-1, self.latent_dim)
        # Shape: (batch_size*8*8, latent_dim)

        # Compute the L2 distance between z_e and the codebook embeddings
        # codebook embeddings shape: (num_embeddings, latent_dim)
        # we want to compute the distance between each z_e and each codebook embedding

        # ||z - e||² = ||z||² + ||e||² - 2·z·eᵀ
        # alternative torch.cdist can be used to compute pairwise distances between two sets of vectors, but this is more optimized

        distances = (
            flat_z_e.pow(2).sum(dim=1, keepdim=True) +
            self.codebook.weight.pow(2).sum(dim=1) -
            2 * flat_z_e @ self.codebook.weight.t() 
        )  #result will be distances of shape (batch_size*8*8, num_embeddings): (batch_size * 64, 512) {distance for each z_e to each codebook embedding}

        # find the nearest codebook embedding for each z_e
        indices = distances.argmin(dim=1) # shape: (batch_size*8*8) gives nearest index in codebook for each z_e

        # find the codebook vectors associated with these indices
        z_q = self.codebook(indices) # shape: (batch_size*8*8, latent_dim) gives the quantized vectors

        # reshape z_q back to (batch_size, 8, 8, latent_dim)
        # 1. reshape from (batch_size*8*8, latent_dim) -> (batch_size, 8, 8, latent_dim)
        z_q = z_q.view(z_e.shape)

        # straight through estimator: during backpropagation, we want the gradients to flow through z_q to z_e, so we use the straight-through estimator trick
        z_q_st = z_e + (z_q - z_e).detach() # during forward pass, z_q_st is equal to z_q, but during backward pass, the gradients will flow through z_e instead of z_q

        # forward pass utilizes z_q (quantized vectors)
        # backward pass utilizes z_e (encoder output) for gradient flow
        # detach() stops gradient going through the (z_q - z_e) term

        # reshape z_q and z_q_st for the decoder input: (batch_size, latent_dim, 8, 8)

        z_q    = z_q.permute(0, 3, 1, 2).contiguous()    # (batch_size, 256, 8, 8)
        z_q_st = z_q_st.permute(0, 3, 1, 2).contiguous() # (batch_size, 256, 8, 8)


        #shape of indices is (batch_size*8*8) but we want to return it as (batch_size, 8, 8) to match the spatial dimensions of the encoder output
        return z_q_st, indices.view(-1, 8, 8)  # return the quantized vectors and the indices of the codebook embeddings used for quantization

if __name__ == "__main__":
    encoder = Encoder()
    codebook = Codebook()

    x = torch.randn(1, 3, 32, 32)
    z_e = encoder(x)
    print(z_e.shape)

    z_q, indices = codebook(z_e)
    print(z_q.shape) # should be same as z_e shape: (batch_size, latent_dim, 8, 8)
    print(indices.shape) # should be (batch_size, 8, 8) gives the codebook indices for each spatial location in the encoder output
    print(indices.min(), indices.max()) # should be between 0 and num_embeddings-1 (0 to 511) since we have 512 codebook embeddings


