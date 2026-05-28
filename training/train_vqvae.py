import torch
import torch.nn as nn
import mlflow
import mlflow.pytorch
from tqdm import tqdm 
from data.dataset import get_cifar_10_dataset
from models.vqvae import VQVAE

def train(num_epochs = 10, batch_size = 64, learning_rate = 2e-4, device = "cuda" if torch.cuda.is_available() else "cpu"):
    train_loader, test_loader = get_cifar_10_dataset(batch_size=batch_size)

    model = VQVAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # start mlflow run
    mlflow.set_experiment("vqvae_cifar10")

    with mlflow.start_run():

        # log hyperparameters once
        mlflow.log_params({
            "num_epochs":    num_epochs,
            "batch_size":    batch_size,
            "lr":            learning_rate,
            "num_embeddings": 512,
            "latent_dim":    256,
            "hidden_dim":    128,
            "beta":          0.25,
            "device":        device
        })


    for epoch in range(num_epochs):

        model.train()
        train_recon = train_codebook = train_commit = train_total = 0.0


        for images, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images = images.to(device)

            optimizer.zero_grad()
            x_recon, total_loss, recon_loss, codebook_loss, commitment_loss, _ = model(batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        
        avg_loss = train_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    
        # validate model after every epoch
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, _  in test_loader:
                images = images.to(device)
                X_recon, loss, _ = model(images)
                val_loss += loss.item()
        
        print(f"Epoch {epoch+1} | "
              f"train loss: {train_loss/len(train_loader):.4f} | "
              f"val loss: {val_loss/len(test_loader):.4f}")
    
    # save weights after training
    torch.save(model.state_dict(), 'vqvae.pth')
    print("saved vqvae.pth")