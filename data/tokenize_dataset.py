import os
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from data.dataset import get_cifar_10_dataset
from models.vqvae import VQVAE


def tokenize_cifar10_dataset(model_path = 'checkpoints/vqvae.pth',
                             save_path = 'tokenized_cifar10/tokens.pt',
                             batch_size = 64,
                             device = 'cuda' if torch.cuda.is_available() else 'cpu'):
    
    if os.path.exists(save_path):
        print(f"tokens already exist at {save_path}, skipping tokenization.")
        return
    
    print('Loading VQ-VAE')

    model = VQVAE().to(device)
    # load model state dict
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    train_loader, _ = get_cifar_10_dataset(batch_size = batch_size)

    all_tokens = []

    with torch.no_grad():
        for images, _ in tqdm(train_loader, desc= 'Tokeinizing dataset: '):
            
            images = images.to(device)

            # get continous latents for the images

            z_e = model.encoder(images)

            # get corresponding indicies for each continous latent vector by comparing them with discrete codebook vectors
            _, _, indicies = model.codebook(z_e)   # shape: (batch_size, 8, 8)

            #flatten tokens/indicies to sequence for transformer

            tokens = indicies.view(images.size(0), -1)   # shape: (batch_size, 64)
            
            all_tokens.append(tokens.cpu()) #list of all tokens
        

        #stack all tokens together: (50000, 64)

    all_tokens = torch.cat(all_tokens, dim = 0)

    print(f'Shape of all tokens: {all_tokens.shape}')
    print(f'token range: {all_tokens.min(), all_tokens.max()}')

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(all_tokens, save_path)
    print(f"saved to {save_path}")

        


    
