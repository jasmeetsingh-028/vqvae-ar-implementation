import torch
from data.token_dataset import TokenDataset
from torch.utils.data import Dataset, DataLoader

def get_token_dataloader(token_path = 'tokenized_cifar10/tokens.pt', 
                         batch_size = 64, 
                         bos_token = 512):

    dataset = TokenDataset(token_path = token_path,
                           bos_token = bos_token)
    
    loader = DataLoader(dataset = dataset, 
                        batch_size = 64, 
                        shuffle = True, 
                        num_workers = 2, 
                        pin_memory = True)
    
    return loader
