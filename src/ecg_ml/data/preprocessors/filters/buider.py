from typing import Self

from .filters import (
    ButterworthFilter,
    ComposeFilter,
    Filter,
    MovingAverage,
    NotchFilter,
)


class FilterBuilder:
    def __init__(self, fs: float) -> None:
        self._fs = fs
        self._filters: list[Filter] = []

    def lowpass(self, fs: float, order: int = 5) -> Self:
        return self._add("lowpass", fs, order)

    def highpass(self, fs: float, order: int = 5) -> Self:
        return self._add("highpass", fs, order)

    def bandpass(self, low: float, high: float, order: int = 5) -> Self:
        return self._add("bandpass", (low, high), order)

    def bandstop(self, low: float, high: float, order: int = 5) -> Self:
        return self._add("bandstop", (low, high), order)

    def notch(self, fs: float, Q: float = 10) -> Self:
        self._filters.append(NotchFilter(self._fs, fs, Q))
        return self

    def moving_average(self, size: int) -> Self:
        self._filters.append(MovingAverage(size))
        return self

    def build(self) -> Filter:
        return ComposeFilter(self._filters)

    def _add(self, btype: str, Wn: float | tuple[float, float], order: int) -> Self:
        self._filters.append(ButterworthFilter(self._fs, btype, Wn, order))
        return self


def make_filter(fs: float) -> Filter:
    return FilterBuilder(fs).notch(50).notch(60).lowpass(35).highpass(0.05).build()
