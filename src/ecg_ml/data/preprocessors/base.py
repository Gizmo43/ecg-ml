from ecg_ml.data.utils.typing import XType


class Preprocessor:
    """A preprocessing step over signals `X` of shape (recordings, leads, samples).

    A step may change `X` in place and returns the result.
    """

    # Datasets only work out the sampling rate when some step needs it.
    requires_sampling_rate: bool = False

    def __call__(self, X: XType, sampling_rate_hz: float | None = None) -> XType:
        raise NotImplementedError()

    def __repr__(self) -> str:
        params = ", ".join(f"{name}={value!r}" for name, value in vars(self).items())
        return f"{type(self).__name__}({params})"
