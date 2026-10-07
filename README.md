# ecg-ml: ECG classification

Multi-label classification of 12-lead ECG recordings with a 1D DenseNet, built on
[PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) and configured with
[Hydra](https://hydra.cc/).

- **Data**: datasets exported from [`ecg-warehouse`](#datasets) (NPZ signals and CSV metadata),
  split into train / val / test by patient, so no patient appears in more than one part
- **Preprocessing**: configurable pipeline of zero-phase Butterworth and notch filters and
  per-lead z-normalization
- **Model**: `DenseNet1d`, a DenseNet-BC with 1D convolutions, trained with `BCEWithLogitsLoss`
- **Metrics**: per-class and macro precision, recall and F1, a per-class confusion matrix and a
  scikit-learn classification report on the test set
- **Tracking**: CSV or MLflow loggers, Optuna hyperparameter search through the Hydra sweeper

## Requirements

- Python 3.12 or 3.13
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- An NVIDIA GPU (CUDA), Apple Silicon (MPS) or CPU

## Installation

```bash
uv sync
```

or, with pip:

```bash
pip install -e .
```

Both install the `ecg_ml` package and two console commands, `train_command` and `eval_command`,
equivalent to `python -m ecg_ml.train` and `python -m ecg_ml.eval`.

To log to MLflow, copy `.env.example` to `.env` and fill in the credentials. The project root
`.env` is loaded automatically on start.

```bash
cp .env.example .env
```

## Quick start

1. Export a dataset from `ecg-warehouse` into `data/` (see [Datasets](#datasets)).
2. Create the dataset, model and experiment configs (see [Configuration](#configuration)).
3. Train, then evaluate the best checkpoint:

```bash
uv run python -m ecg_ml.train experiment=<experiment> logger=csv
```

```bash
uv run python -m ecg_ml.eval experiment=<experiment> logger=csv ckpt_path=logs/train/runs/<run>/checkpoints/epoch_XXX.ckpt
```

`experiment` is required. Training also tests the best checkpoint at the end (`test: True`), so
`ecg_ml.eval` is only needed to re-test a saved checkpoint.

## Datasets

### ecg-warehouse export

`WarehouseDatasetModule` reads a dataset exported by `ecg-warehouse`:

```bash
uv run ecg-warehouse dataset export <name>@<version> --format npz --output ../classification/data/<name>/<version>
```

The export directory must contain:

| File | Contents |
| --- | --- |
| `signals.npz` | `X` (recordings × leads × samples), `y` (recordings × classes, boolean), `classes`, `recording_id`, `split` (`train` / `test`) |
| `meta.csv` | one row per recording with `recording_id`, `source`, `patient_id`; `sampling_rate_hz` or `duration_sec` is needed when filtering is on |

The test set is the warehouse `test` split. The validation set is carved out of `train` by
patient (`train_val_ratio`, 10% by default), grouping recordings by `(source, patient_id)`. Loading
fails if a patient has recordings in both `train` and `test`, if arrays disagree in length, or if
the signals contain NaN or inf values.

To use an export, copy [config/dataset/warehouse.yaml](config/dataset/warehouse.yaml) to
`config/dataset/<name>.yaml` and set `path`:

```yaml
_target_: ecg_ml.data.warehouse.WarehouseDatasetModule
path: ${paths.data_dir}/<name>/<version>
random_state: 42
train_val_ratio: 0.1
```

### Three bases (legacy)

`ThreeBasesDatasetModule` loads the older `three_bases_*.npy` files (PTB-XL records only, NORM vs.
AFIB). It is kept for reference; new experiments should use warehouse exports.

## Configuration

All configs live in [config/](config). [config/train.yaml](config/train.yaml) composes them, and
[config/eval.yaml](config/eval.yaml) and [config/compile.yaml](config/compile.yaml) extend it for
evaluation and TorchScript export:

| Group | Purpose | Files |
| --- | --- | --- |
| `experiment` | ties a dataset and a model together and overrides anything else | `test.yaml`, `experiment.yaml.example` |
| `dataset` | which data to load | `warehouse.yaml`, `afib_dataset.yaml` |
| `model` | Lightning module, network, loss, optimizer, LR scheduler | `densenet.yaml`, `model.yaml.example` |
| `trainer` | `lightning.Trainer` arguments | `default.yaml` |
| `callbacks` | checkpointing, early stopping, progress bar, LR monitor | `default.yaml` |
| `logger` | `csv` or `mlflow` (none by default) | `csv.yaml`, `mlflow.yaml` |
| `hparams_search` | Optuna sweeps | `optuna.yaml.example` |
| `local` | machine-specific overrides, optional | `default.yaml` (create it yourself) |

### Experiments

An experiment is the usual entry point. Start from
[config/experiment/experiment.yaml.example](config/experiment/experiment.yaml.example):

```yaml
# @package _global_

defaults:
  - override /dataset: <name>
  - override /model: <name>

seed: 69
tags: ["ptb-xl"]

trainer:
  max_epochs: 30
  gradient_clip_val: 0.5
  accelerator: gpu # gpu, mps or cpu
  precision: 16-mixed # 32 on CPU and MPS

data:
  batch_size: 64
```

### Model

Copy [config/model/model.yaml.example](config/model/model.yaml.example) (DenseNet-121) or start
from [config/model/densenet.yaml](config/model/densenet.yaml) (a smaller network). Things to keep in
mind:

- `labels: null` takes the class names and their order from the dataset, and
  `network.output_size: null` takes the number of classes. Values set explicitly must match the
  dataset.
- The stem and transitions shorten the signal `2 ** (len(block_config) + 1)` times, so recordings
  need at least that many samples.
- `kernel_size` must be odd.

### Preprocessing

The `preprocessing` pipeline in `train.yaml` runs on every recording before the split. Steps run in
order; disable one with `preprocessing.steps.<name>=null`.

| Step | Parameters |
| --- | --- |
| `filter` | `highpass_hz`, `lowpass_hz`, `order`, `notch_hz`, `notch_q`; `null` cutoffs and an empty `notch_hz` are off |
| `znorm` | per-lead z-score; constant leads become zeros |

Every frequency must stay below Nyquist (sampling rate / 2), e.g.:

- 500 Hz: `highpass_hz=0.05`, `lowpass_hz=35`, `notch_hz=[50,60]`
- 100 Hz: `highpass_hz=0.5`, `lowpass_hz=35`, no 50/60 Hz notches

### Command-line overrides

Any value can be overridden from the command line:

```bash
uv run python -m ecg_ml.train experiment=<experiment> \
  trainer.max_epochs=50 data.batch_size=32 model.optimizer.lr=0.001 \
  preprocessing.steps.filter.lowpass_hz=35 callbacks.early_stopping.enabled=true
```

Quote values that contain lists: `"'model.network.block_config=[6,12,24]'"`. Disable all callbacks
with `~callbacks`, and resume training with `ckpt_path=<path>`.

## Outputs

Each run writes to `logs/<task_name>/runs/<YYYY-MM-DD_HH-MM-SS>/` (`task_name` is `train` or
`eval`):

```text
logs/train/runs/2026-10-04_23-24-25/
├── .hydra/              # resolved config, Hydra config and overrides
├── checkpoints/         # best checkpoint by val/f1/macro and last.ckpt
├── csv/version_0/       # metrics.csv and hparams.yaml (logger=csv)
├── config_tree.log
└── train.log
```

Logged metrics:

- `train/loss`, `val/loss`
- `val/{precision,recall,f1}/macro` and per class as `val/f1/<class>`
- `val/loss_best`, `val/f1/macro_best`: the best values so far, used as Optuna objectives
- `test/...`: the same metrics on the test set

With `logger=mlflow`, runs go to the `dev/ecg/classification` experiment on the tracking server in
[config/logger/mlflow.yaml](config/logger/mlflow.yaml), together with the model and the test
confusion matrix (`test/confusion_matrix.png`). `MLFLOW_TRACKING_USERNAME` and
`MLFLOW_TRACKING_PASSWORD` must be set.

## Hyperparameter search

```bash
cp config/hparams_search/optuna.yaml.example config/hparams_search/optuna.yaml
```

```bash
uv run python -m ecg_ml.train -m hparams_search=optuna experiment=<experiment>
```

The example minimizes `val/loss_best` over 20 TPE trials, searching the learning rate and batch
size. Set `optimized_metric` and `direction` together (e.g. `val/f1/macro_best` with `maximize`),
and edit `hydra.sweeper.params` to change the search space. Results go to
`logs/train/multiruns/<timestamp>/`. See [docs/hparams_search.md](docs/hparams_search.md) for
more.

## TorchScript export

[src/ecg_ml/compile.py](src/ecg_ml/compile.py) loads a checkpoint and saves it as TorchScript to
`logs/compile/runs/<timestamp>/model.pt`:

```bash
uv run python -m ecg_ml.compile experiment=<experiment> ckpt_path=<path to .ckpt>
```

## Docker

[docker/Dockerfile](docker/Dockerfile) builds a CUDA image for training and evaluation on NVIDIA
GPUs (driver 580 or newer and the NVIDIA Container Toolkit on the host).

```bash
docker build -f docker/Dockerfile -t classification:gpu .
```

```bash
docker run --rm --gpus all --ipc=host -v "$(pwd)/data:/app/data" -v "$(pwd)/logs:/app/logs" --env-file .env classification:gpu python -m ecg_ml.train experiment=<experiment> trainer.accelerator=gpu
```

`data/` and `logs/` are mounted at runtime.

## Project structure

```text
.
├── config/                     # Hydra configs (see Configuration)
├── data/                       # datasets (not version-controlled)
├── docker/                     # GPU Dockerfile
├── docs/                       # extra documentation
├── logs/                       # run outputs (not version-controlled)
├── src/ecg_ml/
│   ├── train.py                # training entry point
│   ├── eval.py                 # checkpoint evaluation entry point
│   ├── compile.py              # TorchScript export
│   ├── data/
│   │   ├── warehouse.py        # ecg-warehouse NPZ export
│   │   ├── three_bases.py      # legacy dataset
│   │   ├── preprocessors/      # filtering, z-normalization, pipeline
│   │   └── utils/              # DatasetModule base, patient-wise split, hashing
│   ├── models/densenet.py      # DenseNet1d
│   ├── modules/classifier.py   # Lightning module: training loop and metrics
│   └── utils/                  # logging, instantiation helpers
└── pyproject.toml
```
