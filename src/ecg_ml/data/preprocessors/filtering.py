from collections.abc import Sequence

from ecg_ml.data.utils.typing import XType
from ecg_ml.utils.rank_logger import RankedLogger

from .base import Preprocessor
from .filters.buider import FilterBuilder

log = RankedLogger(__name__, rank_zero_only=True)


class Filtering(Preprocessor):
    """Zero-phase filtering of every lead: Butterworth highpass, then lowpass, then notches.

    A `None` cutoff and an empty `notch_hz` turn that filter off. All frequencies must stay
    below Nyquist (sampling_rate_hz / 2), e.g. 50 Hz notches do not fit 100 Hz signals.
    """

    def __init__(
        self,
        highpass_hz: float | None = None,
        lowpass_hz: float | None = None,
        order: int = 5,
        notch_hz: Sequence[float] = (),
        notch_q: float = 10,
    ) -> None:
        if (
            highpass_hz is not None
            and lowpass_hz is not None
            and highpass_hz >= lowpass_hz
        ):
            raise ValueError(
                f"highpass_hz={highpass_hz} must be below lowpass_hz={lowpass_hz}"
            )
        self.highpass_hz = highpass_hz
        self.lowpass_hz = lowpass_hz
        self.order = order
        self.notch_hz = tuple(notch_hz)
        self.notch_q = notch_q

    @property
    def requires_sampling_rate(self) -> bool:
        return (
            self.highpass_hz is not None
            or self.lowpass_hz is not None
            or bool(self.notch_hz)
        )

    def __call__(self, X: XType, sampling_rate_hz: float | None = None) -> XType:
        if not self.requires_sampling_rate:
            return X
        if sampling_rate_hz is None:
            raise ValueError("Filtering needs the sampling rate of the signals")

        nyquist = sampling_rate_hz / 2
        frequencies = {
            "highpass_hz": [self.highpass_hz],
            "lowpass_hz": [self.lowpass_hz],
            "notch_hz": self.notch_hz,
        }
        for name, values in frequencies.items():
            for value in values:
                if value is not None and not 0 < value < nyquist:
                    raise ValueError(
                        f"{name}={value} must be in (0, {nyquist:g}) Hz: "
                        f"the signals are sampled at {sampling_rate_hz:g} Hz"
                    )

        builder = FilterBuilder(sampling_rate_hz)
        if self.highpass_hz is not None:
            builder.highpass(self.highpass_hz, self.order)
        if self.lowpass_hz is not None:
            builder.lowpass(self.lowpass_hz, self.order)
        for notch_hz in self.notch_hz:
            builder.notch(notch_hz, self.notch_q)
        log.info(f"Filtering at {sampling_rate_hz:g} Hz")
        return builder.build()(X, inplace=True)
