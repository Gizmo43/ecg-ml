## Hyperparameter Search

You can define hyperparameter search by adding new config file to [config/hparams_search](../config/hparams_search). Start from the Optuna example:

```bash
cp config/hparams_search/optuna.yaml.example config/hparams_search/optuna.yaml
```

<details>
<summary><b>Show example hyperparameter search config</b></summary>

```yaml
# @package _global_

defaults:
  - override /hydra/sweeper: optuna

# choose metric which will be optimized by Optuna
# make sure this is the correct name of some metric logged in lightning module!
# `_best` metrics are the best value over the epochs: val/loss_best, val/f1/macro_best
optimized_metric: "val/f1/macro_best"

# here we define Optuna hyperparameter search
# it optimizes for value returned from function with @hydra.main decorator
hydra:
  mode: "MULTIRUN" # set hydra to multirun by default if this config is attached

  sweeper:
    _target_: hydra_plugins.hydra_optuna_sweeper.optuna_sweeper.OptunaSweeper

    # 'minimize' or 'maximize' the objective
    direction: maximize

    # total number of runs that will be executed
    n_trials: 20

    # choose Optuna hyperparameter sampler
    # docs: https://optuna.readthedocs.io/en/stable/reference/samplers.html
    sampler:
      _target_: optuna.samplers.TPESampler
      seed: 1234
      n_startup_trials: 10 # number of random sampling runs before optimization starts

    # define hyperparameter search space
    params:
      model.optimizer.lr: interval(0.0001, 0.1)
      data.batch_size: choice(32, 64, 128, 256)
      model.network.growth_rate: choice(12, 24, 32)
      model.network.drop_rate: interval(0.0, 0.5)
```

</details>

Next, execute it with: `python -m ecg_ml.train -m hparams_search=optuna experiment=<experiment>`

Using this approach doesn't require adding any boilerplate to code, everything is defined in a single config file. The only necessary thing is to return the optimized metric value from the launch file, which `ecg_ml.train` does.

Every trial also tests its best checkpoint; add `test=False` to skip that and save time.

You can use different optimization frameworks integrated with Hydra, like [Optuna, Ax or Nevergrad](https://hydra.cc/docs/plugins/optuna_sweeper/).

The `optimization_results.yaml` will be available under `logs/train/multiruns/<timestamp>` folder, with every trial in a numbered subfolder.

This approach doesn't support advanced techniques like pruning - for more sophisticated search and workflows, you should probably write a dedicated optimization task (without multirun feature).
