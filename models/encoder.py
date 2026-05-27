import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, in_channels = 3, hidden_dim = 128, latent_dim = 256):
        super().__init__()

        self.net = nn.Sequential(
            # (batch_Size, 3, 32, 32) -> (batch_size, 128, 16, 16))
            nn.Conv2d(in_channels, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),

            # (batch_Size, 128, 16, 16) -> (batch_size, 256, 8, 8))
            nn.Conv2d(hidden_dim, hidden_dim*2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),

            # (batch_Size, 256, 8, 8) -> (batch_size, 256, 8, 8))
            nn.Conv2d(hidden_dim*2, latent_dim, kernel_size = 1),
        )
    
    def forward(self, x):
        return self.net(x)



if __name__ == "__main__":
    encoder = Encoder()
    x = torch.randn(1, 3, 32, 32)
    z = encoder(x)
    print(z.shape)

    ## this is z_e from the encoder: (batch_size, latent_dim = 256, 8, 8)

    # trace parameter shapes through the model

    for name, param in encoder.named_parameters():
        print(name, param.shape)

    