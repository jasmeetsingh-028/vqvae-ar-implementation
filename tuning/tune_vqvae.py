import optuna
import torch
from data.dataset import get_cifar_10_dataset
from models.vqvae import VQVAE


# tree structures parzen estimator
# uses results from previous trials
# focuses on region with high potential

# dynamic search space

def objective(trial, device):

    # define the hyperparameters to tune
    # hidden and latant dim for encoder and decoder
    # num_embeddings for codebook
    # learning rate and beta for loss weighting
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
    latent_dim = trial.suggest_categorical("latent_dim", [128, 256, 512])
    #beta = trial.suggest_float("beta", 0.0, 1.0, step=0.25)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)
    num_embeddings = trial.suggest_categorical("num_embeddings", [64, 128, 256, 512])
    init_range     = trial.suggest_categorical("init_range", [0.1, 0.5, 1.0, 2.0])

    # create the model with the suggested hyperparameters
    vqvae = VQVAE(hidden_dim=hidden_dim, latent_dim=latent_dim, num_embeddings=num_embeddings, beta=0.25, init_range=init_range).to(device)

    # train the model for a few epochs and return the final loss

    batch_size = 64
    train_loader, test_loader = get_cifar_10_dataset(batch_size=batch_size)

    # train for total_loss for 10 epochs and return the final average loss
    optimizer = torch.optim.Adam(vqvae.parameters(), lr=learning_rate) #fixed parameter


    best_val_loss = float("inf")

    for epoch in range(25):

        vqvae.train()
        #total_loss = 0.0

        for images, _ in train_loader:
            images = images.to(device)

            optimizer.zero_grad()
            x_recon, total_loss, recon_loss, codebook_loss, commitment_loss,  _ = vqvae(images)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(vqvae.parameters(), max_norm=1.0)  
            optimizer.step()


        # validate
        vqvae.eval()
        val_total_loss = 0.0

        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(device)
                x_recon, total_loss, recon_loss, codebook_loss, commitment_loss,  _ = vqvae(images)
                val_total_loss += total_loss.item()
        
        avg_val_loss = val_total_loss / len(test_loader)

        # # track the average validation loss for this trial
        # trial.report(avg_val_loss, epoch)

        # track the best validation loss for this trial

        best_val_loss = min(best_val_loss, avg_val_loss)

        # report to optuna

        trial.report(avg_val_loss, epoch)


    return best_val_loss







        


        



