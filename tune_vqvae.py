import optuna
import torch
import datetime
from tuning.tune_vqvae import objective
import optuna.visualization as vis


n_trials = 50
device = 'cuda' if torch.cuda.is_available() else 'cpu'


storage = f"sqlite:///optuna_db/study_{n_trials}_with_init.db"

study = optuna.create_study(
    direction = 'minimize',
    storage= storage,
    study_name=f'vqvae_tuning_{n_trials}_trials',
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=3) # to kill bad trials early if avg_val_loss is worse than the median
)

study.optimize(
    lambda trial: objective(trial, device),
    n_trials = n_trials
)

# print best results
print("Best trial:")
print(study.best_trial.value)
print(study.best_trial.params)


p1 = vis.plot_parallel_coordinate(study)
p2 = vis.plot_param_importances(study)
p3 = vis.plot_optimization_history(study)


p1.write_image("results_and_plots/tuning_plots/optuna_parallel.png")
p2.write_image("results_and_plots/tuning_plots/optuna_importances.png")
p3.write_image("results_and_plots/tuning_plots/optuna_history.png")