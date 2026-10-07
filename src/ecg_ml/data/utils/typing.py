from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pandas as pd

type XType = npt.NDArray[np.float32]
type YType = npt.NDArray[np.bool_]

type YPredType = npt.NDArray[np.bool_]
type Function = Callable[[XType], YPredType]

type TrainValData = tuple[XType, XType, YType, YType, pd.DataFrame, pd.DataFrame]
