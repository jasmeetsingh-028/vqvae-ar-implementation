import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_cifar_10_dataset(batch_size = 64, num_workers = 2, data_dir = "./data/files"):
    
    transform = transforms.Compose([
        transforms.ToTensor(), # convert PIL image to tensor and scale pixel values to [0, 1]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # normalize to [-1, 1]
    ])
    #use tanh activation in the decoder to output in range [-1, 1] to match the normalized input, so we normalize the input images to [-1, 1] as well

    train_dataset = datasets.CIFAR10(root=data_dir, 
                                    train=True,
                                    download=True, 
                                    transform=transform)
    

    test_dataset = datasets.CIFAR10(root=data_dir, 
                                    train=False, 
                                    download=True, 
                                    transform=transform)


    train_loader = DataLoader(train_dataset, 
                              batch_size=batch_size, 
                              shuffle=True, 
                              num_workers=num_workers, 
                              pin_memory=True)
    
    # pin_memory = True allows faster data transfer to GPU, set it to False if you are using CPU or if you encounter memory issues
    # pin_memory = False, data is nomal CPU memory
    # pin_memory = True, data is stored in pinned memory for faster transfer to GPU
    

    test_loader = DataLoader(test_dataset, 
                             batch_size=batch_size, 
                             shuffle=False, 
                             num_workers=num_workers,
                             pin_memory=True)

    return train_loader, test_loader


if __name__ == "__main__":
    train_loader, test_loader = get_cifar_10_dataset()
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of test batches: {len(test_loader)}")

    # iterate through one batch of the training data
    for images, labels in train_loader:
        print(f"Batch of images shape: {images.shape}") # should be (batch_size, 3, 32, 32)
        print(f"Batch of labels shape: {labels.shape}") # should be (batch_size,)
        break