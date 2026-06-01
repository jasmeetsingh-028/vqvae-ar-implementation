import os
import torch
import mlflow 
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm
from models.GPT import GPT
from data.token_dataloader import get_token_dataloader


def train(num_epochs = 10, batch_size = 64, learning_rate = 2e-4, device = 'cuda' if torch.cuda.is_available() else 'cpu'):

    loader = get_token_dataloader(batch_size = batch_size)

    model = GPT().to(device)
    optimizer = Adam(model.parameters(), lr = learning_rate)

    mlflow.set_experiment("GPT for image generation in AR setting")

    params = {
    "num_epochs": num_epochs,
    "batch_size": batch_size,
    "lr": learning_rate,
    "n_layers": 6,
    "n_self_attention_heads": 8,
    "d_model": 256,
    "d_ff": 1024,
    "seq_len": 65,
    "vocab_size": 513,
    }

    with mlflow.start_run():

        mlflow.log_params(params)

        for epoch in range(num_epochs):

            model.train()
            train_loss = 0.0

            for input_seq, target_seq in tqdm(loader, desc = f"Epoch : {epoch}"):
                input_seq = input_seq.to(device) #(batch_Size = 64, seq_len = 65)
                target_seq = target_seq.to(device) #(batch_Size = 64, seq_len = 64)

                optimizer.zero_grad()

                logits = model(input_seq)   # output logits across the entire vocabulary,  shape: (batch_Size = 64, seq_length = 65, vocab = 513)

                # targets: shifted intput sequence (one correct token per index) 
                # logits : prob scores across all the 513 token in vocabs 

                ## IMP: “COMPARE: Out of all 513 scores, how good is the score for the correct token?”
                
                logits = logits[:, :-1, :]   # (B, 64, 513) — drop last position


                ## why dropping last postion? why comparing logits and targets

                loss = F.cross_entropy(
                        logits.reshape(-1, 513),          # (B*64, 513)   
                        target_seq.reshape(-1)            # (B*64,)  
                    )
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item()
            
            avg_loss = train_loss / len(loader)

            mlflow.log_metrics({"train/loss": avg_loss}, step=epoch)
            print(f"Epoch [{epoch+1}/{num_epochs}] | loss: {avg_loss:.4f}")

            # save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                os.makedirs("checkpoints", exist_ok=True)
                torch.save(model.state_dict(), f"checkpoints/gpt_epoch_{epoch+1}.pth")
                mlflow.log_artifact(f"checkpoints/gpt_epoch_{epoch+1}.pth")

            # save final model
            torch.save(model.state_dict(), "checkpoints/gpt_final.pth")
            mlflow.log_artifact("checkpoints/gpt_final.pth")
            print("saved gpt_final.pth")