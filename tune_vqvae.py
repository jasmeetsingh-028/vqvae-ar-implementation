import optuna
import torch
import datetime
from tuning.tune_vqvae import objective


n_trials = 40
device = 'cuda' if torch.cuda.is_available() else 'cpu'


storage = f"sqlite:///optuna_db/study_{n_trials}.db"

study = optuna.create_study(
    direction = 'minimize',
    storage= storage,
    study_name=f'vqvae_tuning_{n_trials}_trials',
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=3) # to kill bad trials early
845)

study.optimize(
    lambda trial: objective(trial, device),
    n_trials = n_trials
)

# print best results
print("Best trial:")
print(study.best_trial.value)
print(study.best_trial.params)