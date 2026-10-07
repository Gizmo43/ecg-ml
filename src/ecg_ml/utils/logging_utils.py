from typing import Any

from lightning.pytorch.loggers import MLFlowLogger
from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import OmegaConf

from . import rank_logger

log = rank_logger.RankedLogger(__name__, rank_zero_only=True)


@rank_zero_only
def log_hyperparameters(object_dict: dict[str, Any]) -> None:
    """Controls which config parts are saved by Lightning loggers.

    Additionally saves:
        - Number of model parameters
        - Dataset fingerprint: classes, content hashes, part sizes
        - The whole resolved config as `config.yaml` (MLflow only)

    :param object_dict: A dictionary containing the following objects:
        - `"cfg"`: A DictConfig object containing the main config.
        - `"dataset"`: The DatasetModule.
        - `"model"`: The Lightning model.
        - `"trainer"`: The Lightning trainer.
    """
    hparams = {}

    cfg: dict[str, Any] = OmegaConf.to_container(object_dict["cfg"], resolve=True)  # type: ignore
    dataset = object_dict["dataset"]
    model = object_dict["model"]
    trainer = object_dict["trainer"]

    if not trainer.logger:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    hparams["model"] = cfg["model"]

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    hparams["dataset"] = cfg["dataset"]
    hparams["dataset/fingerprint"] = dataset.fingerprint()
    hparams["preprocessing"] = cfg.get("preprocessing")
    hparams["data"] = cfg["data"]
    hparams["trainer"] = cfg["trainer"]

    hparams["callbacks"] = cfg.get("callbacks")
    hparams["extras"] = cfg.get("extras")

    hparams["task_name"] = cfg.get("task_name")
    hparams["ckpt_path"] = cfg.get("ckpt_path")
    hparams["seed"] = cfg.get("seed")

    # send hparams to all loggers
    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)
        # MLflow cuts param values at 250 characters, so it gets the whole config as well
        if isinstance(logger, MLFlowLogger):
            logger.experiment.log_text(
                run_id=logger.run_id,
                text=OmegaConf.to_yaml(object_dict["cfg"], resolve=True),
                artifact_file="config.yaml",
            )
