import numpy as np
import numpy.typing as npt
from scipy import signal
from scipy.ndimage import uniform_filter1d


class Filter:
    def _apply(self, X: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        raise NotImplementedError()

    def apply(
        self, X: npt.NDArray[np.float32], inplace: bool = False, batch_size: int = 512
    ) -> npt.NDArray[np.float32]:
        X = X if inplace else X.copy()
        original_shape = X.shape
        X = X.reshape(-1, original_shape[-1])

        for begin in range(0, len(X), batch_size):
            X[begin : begin + batch_size] = self._apply(X[begin : begin + batch_size])

        return X.reshape(original_shape)

    def __call__(
        self, X: npt.NDArray[np.float32], inplace: bool = False, batch_size: int = 512
    ) -> npt.NDArray[np.float32]:
        return self.apply(X, inplace=inplace, batch_size=batch_size)


class ButterworthFilter(Filter):
    def __init__(
        self, fs: float, btype: str, Wn: float | tuple[float, float], order: int = 5
    ) -> None:
        self._sos = signal.butter(order, Wn, btype=btype, output="sos", fs=fs)

    def _apply(self, X: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        return signal.sosfiltfilt(self._sos, X)


class NotchFilter(Filter):
    def __init__(self, fs: float, fs_notch: float, Q: float = 10) -> None:
        self._b, self._a = signal.iirnotch(fs_notch, Q, fs)
        self._b = self._b
        self._a = self._a

    def _apply(self, X: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        return signal.filtfilt(self._b, self._a, X)


class MovingAverage(Filter):
    def __init__(self, size: int) -> None:
        self._size = size

    def _apply(self, X: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        return uniform_filter1d(X, self._size)


class ComposeFilter(Filter):
    def __init__(self, filters: list[Filter]) -> None:
        self._filters = filters

    def _apply(self, X: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        for f in self._filters:
            X = f(X)
        return X
