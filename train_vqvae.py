import torch
from training.train_vqvae import train


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

train(num_epochs=50, batch_size=64, learning_rate=1e-4, device=device)