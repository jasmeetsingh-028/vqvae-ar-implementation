import os
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class TokenDataset(Dataset): #inherits from dataset class

    def __init__(self, token_path = 'tokenized_cifar10/tokens.pt', bos_token=512):
        self.tokens = torch.load(token_path)
        self.bos_token = bos_token # beggining of sequence token

    
    def __len__(self):
        return len(self.tokens)
    
    def __getitem__(self, idx):
        tokens = self.tokens[idx]

        # prepend beggining of seq token

        bos = torch.tensor([self.bos_token], dtype = torch.long)
        input_sequence = torch.cat([bos, tokens[-1]]) #[bos, t0, ... tn-1]
        target_Sequence = tokens #[t0, .. tn]

        return input_sequence, target_Sequence