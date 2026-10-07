import os

import hydra
import lightning
import rootutils
from lightning import LightningDataModule, Trainer
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, open_dict

from ecg_ml.data.utils.dataset import DatasetModule
from ecg_ml.utils.instantiators import instantiate_callbacks, instantiate_loggers
from ecg_ml.utils.logging_utils import log_hyperparameters
from ecg_ml.utils.rank_logger import RankedLogger
from ecg_ml.utils.utils import extras, get_metric_value

# The project root holds config/ and .env, which this also loads
ROOT = rootutils.setup_root(__file__)

log = RankedLogger(__name__, rank_zero_only=True)


def run(cfg: DictConfig) -> float | None:
    """Trains and/or tests as `cfg.train` and `cfg.test` say; returns `cfg.optimized_metric`."""
    extras(cfg)

    logger: dict[str, Logger] = instantiate_loggers(cfg.get("logger"))

    if (
        "mlflow" in logger
        and not os.getenv("MLFLOW_TRACKING_USERNAME")
        and not os.getenv("MLFLOW_TRACKING_PASSWORD")
    ):
        raise RuntimeError(
            "Please set MLflow authentication environment variables: MLFLOW_TRACKING_USERNAME, MLFLOW_TRACKING_PASSWORD"
        )

    if cfg.get("seed"):
        lightning.seed_everything(cfg.seed, workers=True)

    log.info("Loading dataset...")
    preprocessing = hydra.utils.instantiate(cfg.preprocessing)
    dataset: DatasetModule = hydra.utils.instantiate(
        cfg.dataset, preprocessing=preprocessing
    )
    datamodule: LightningDataModule = dataset.to_datamodule(**cfg.data)
    log.info("Loaded dataset")

    # Label i of the model is column i of the dataset Y, so the names come from the dataset.
    labels = cfg.model.get("labels")
    if labels is not None and [str(c) for c in labels] != [
        str(c) for c in dataset.classes
    ]:
        raise ValueError(
            f"model.labels {list(labels)} differ from the dataset classes {dataset.classes}: "
            "set model.labels=null to take the classes from the dataset"
        )
    # The network has an output per class, so their number comes from the dataset as well.
    output_size = cfg.model.network.get("output_size")
    if output_size is not None and output_size != len(dataset.classes):
        raise ValueError(
            f"model.network.output_size={output_size} differs from the {len(dataset.classes)} "
            "dataset classes: set model.network.output_size=null to take it from the dataset"
        )
    with open_dict(cfg):
        cfg.model.labels = list(dataset.classes)
        if "output_size" in cfg.model.network:
            cfg.model.network.output_size = len(dataset.classes)

    model = hydra.utils.instantiate(cfg.model)

    callbacks: list[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger.values()
    )

    object_dict = {
        "cfg": cfg,
        "dataset": dataset,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    # Checkpoints pickle the hyperparameters (partial optimizer, DictConfig, nn.Modules), which
    # torch.load's default weights_only=True rejects; they are our own runs, so load them fully.
    metrics = {}

    if cfg.get("train"):
        trainer.fit(
            model=model,
            datamodule=datamodule,
            ckpt_path=cfg.get("ckpt_path"),
            weights_only=False,
        )
        metrics.update(trainer.callback_metrics)

    if cfg.get("test"):
        if cfg.get("train"):
            # cfg.ckpt_path is where training resumed from, so test what training selected
            ckpt_path = None
            checkpoint_callback: ModelCheckpoint | None = trainer.checkpoint_callback  # type: ignore
            if checkpoint_callback is not None and checkpoint_callback.best_model_path:
                ckpt_path = checkpoint_callback.best_model_path
                log.info(f"Testing best ckpt: {ckpt_path}")
            else:
                log.warning("Best ckpt not found! Testing current weights...")
        else:
            ckpt_path = cfg.get("ckpt_path")
            if ckpt_path is not None:
                log.info(f"Testing ckpt: {ckpt_path}")
            else:
                log.warning("No ckpt_path given! Testing initial weights...")
        trainer.test(
            model=model, datamodule=datamodule, ckpt_path=ckpt_path, weights_only=False
        )
        metrics.update(trainer.callback_metrics)

    metric_value = get_metric_value(
        metrics_dict=metrics, metric_name=cfg.get("optimized_metric")
    )

    return metric_value


# Hydra resolves a relative config_path against the package when run as an entry point
# (`train_command`) instead of this file, so the path is absolute.
@hydra.main(
    version_base="1.3", config_path=str(ROOT / "config"), config_name="train.yaml"
)
def main(cfg: DictConfig) -> float | None:
    return run(cfg)


if __name__ == "__main__":
    main()
