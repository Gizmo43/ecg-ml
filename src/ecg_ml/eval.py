import hydra
from omegaconf import DictConfig

from ecg_ml.train import ROOT, run


# An absolute config_path, as in train.py, works for `eval_command` too
@hydra.main(
    version_base="1.3", config_path=str(ROOT / "config"), config_name="eval.yaml"
)
def main(cfg: DictConfig) -> None:
    # without a checkpoint the test would measure the initial weights
    if cfg.ckpt_path is None:
        raise ValueError("Set ckpt_path to the checkpoint to evaluate")
    run(cfg)


if __name__ == "__main__":
    main()
