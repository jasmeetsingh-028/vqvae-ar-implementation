# from data.tokenize_dataset import tokenize_cifar10_dataset

# print('Processing cifar 10 dataset...')
# tokenize_cifar10_dataset()

from data.token_dataloader import get_token_dataloader


loader = get_token_dataloader()

input_seq, target_seq = next(iter(loader))

print(input_seq.shape)   # (64, 65)
print(target_seq.shape)  # (64, 64)
print(input_seq[0])      # Input sequence starts with 512 (BOS)
print(target_seq[0])     