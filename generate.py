import torch
import matplotlib.pyplot as plt
from models.vqvae import VQVAE
from models.GPT import GPT

def generate_images(
        vqvae_checkpoint = 'checkpoints/vqvae.pth',
        gpt_checkpoint = 'checkpoints/gpt_final.pth',
        num_images = 8,
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
):
    
    #load models
    vqvae = VQVAE().to(device)
    vqvae.load_state_dict(torch.load(vqvae_checkpoint, map_location=device))
    vqvae.eval()


    gpt = GPT().to(device)
    gpt.load_state_dict(torch.load(gpt_checkpoint, map_location=device))
    gpt.eval()

    images = []

    with torch.no_grad():
        for i in range(num_images):

            #1. sample 64 indices from transformer model
            tokens = gpt.generate(device = device)   # output shape: (1, 64)
            tokens = tokens.squeeze(0)  #shape: (64, )

            #2. codebook lookups
            # get qunatized vectors corresponsding to the generated tokens
            z_q = vqvae.codebook.codebook(tokens)  # gives 64 quantized vectors corresponding to each token/idx, each of shape 256, shape: (64, 256)

            #3. reshape the codebook outputs to feed them to the decoder
            z_q = z_q.view(8, 8, 256)  # shape: (8, 8, 256)
            z_q = z_q.permute(2, 0, 1) #shape: (256, 8, 8)
            z_q = z_q.unsqueeze(0) #shape: (1, 256, 8, 8) (bacth_size, latent_dim, h, w)

            #4. decode the quantized vectors to get the generated image
            x_recon = vqvae.decoder(z_q) # shape: (1, 3, 32, 32)
            images.append(x_recon.squeeze(0).cpu()) # shape: (3, 32, 32)
        
    
    # denormalize images from [-1, 1] to [0, 1]
    images = torch.stack(images)  #shape: (num_images, 3, 32, 32)
    images = (images * 0.5 + 0.5).clamp(0, 1)

    # plot the generated images in a grid and save the figure
    fig, axes = plt.subplots(1, num_images, figsize=(num_images * 2, 2))

    for i in range(num_images):
        axes[i].imshow(images[i].permute(1, 2, 0).numpy())
        axes[i].axis('off')

    plt.suptitle('Generated Images', fontsize=12)
    plt.tight_layout()
    plt.savefig('outputs/generated_images/generated.png', dpi=150)
    plt.close()
    print(f"saved generated.png")



if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    generate_images(device=device)


