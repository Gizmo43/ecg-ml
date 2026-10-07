import numpy as np

from ecg_ml.data.utils.typing import XType

from .base import Preprocessor


class ZNormalization(Preprocessor):
    """Z-normalize every lead of every recording in place; a constant lead becomes zeros."""

    def __init__(self, batch_size: int = 1024) -> None:
        self.batch_size = batch_size

    def __call__(self, X: XType, sampling_rate_hz: float | None = None) -> XType:
        for begin in range(0, len(X), self.batch_size):
            chunk = X[begin : begin + self.batch_size]
            mean = chunk.mean(axis=-1, keepdims=True)
            # A NaN or inf anywhere in a lead makes its mean non-finite.
            if not np.isfinite(mean).all():
                raise ValueError("Cannot z-normalize signals with NaN or inf values")
            constant = chunk.max(axis=-1) == chunk.min(axis=-1)
            std = chunk.std(axis=-1, keepdims=True)
            std[constant] = 1
            chunk -= mean
            chunk /= std
            chunk[constant] = 0
        return X
