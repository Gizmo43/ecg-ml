from collections.abc import Mapping

from ecg_ml.data.utils.typing import XType
from ecg_ml.utils.rank_logger import RankedLogger

from .base import Preprocessor

log = RankedLogger(__name__, rank_zero_only=True)


class Pipeline(Preprocessor):
    """Runs named steps in order; a `None` step is skipped, so a config can switch it off."""

    def __init__(self, steps: Mapping[str, Preprocessor | None] | None = None) -> None:
        self.steps = {
            name: step for name, step in (steps or {}).items() if step is not None
        }

    @property
    def requires_sampling_rate(self) -> bool:
        return any(step.requires_sampling_rate for step in self.steps.values())

    def __call__(self, X: XType, sampling_rate_hz: float | None = None) -> XType:
        log.info(f"Preprocessing steps: {list(self.steps) or 'none'}")
        for name, step in self.steps.items():
            log.info(f"Preprocessing step {name}: {step}")
            X = step(X, sampling_rate_hz)
        return X
