import optuna
import torch
from tuning.tune_vqvae import objective


n_trials = 10
device = 'cuda' if torch.cuda.is_available() else 'cpu'

study = optuna.create_study(
    direction = 'minimize',
    storage='sqlite:///optuna_db/study.db',
    study_name='vqvae_tuning',
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=3) # to kill bad trials early
)

study.optimize(
    lambda trial: objective(trial, device),
    n_trials = n_trials
)

# print best results
print("Best trial:")
print(study.best_trial.value)
print(study.best_trial.params)