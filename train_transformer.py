import torch
from training.train_gpt import train

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"using device: {device}")

train(num_epochs=50, batch_size=64, learning_rate=2e-4, device=device)