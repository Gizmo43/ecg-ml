from pathlib import Path

import hydra
import rootutils
from omegaconf import DictConfig

from ecg_ml.utils.rank_logger import RankedLogger

ROOT = rootutils.setup_root(__file__)

log = RankedLogger(__name__, rank_zero_only=True)


@hydra.main(
    version_base="1.3", config_path=str(ROOT / "config"), config_name="compile.yaml"
)
def main(cfg: DictConfig) -> None:
    if cfg.ckpt_path is None:
        raise ValueError("Set ckpt_path to the checkpoint to compile")

    # labels and the rest of the hyperparameters come from the checkpoint, which pickles them,
    # so it is loaded fully as in train.py
    model_class = hydra.utils.get_class(cfg.model._target_)
    model = model_class.load_from_checkpoint(cfg.ckpt_path, weights_only=False)

    file_path = Path(cfg.paths.output_dir) / "model.pt"
    model.to_torchscript(file_path=file_path)
    log.info(f"Saved TorchScript model to {file_path}")


if __name__ == "__main__":
    main()
