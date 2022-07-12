import optuna


def basic_objective(trial):
    x = trial.suggest_float('x', -10, 10)
    return (x - 2) ** 2


study = optuna.create_study()
study.optimize(basic_objective, n_trials=100)

print(study.best_params)