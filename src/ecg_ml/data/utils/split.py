from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import sklearn.utils
from sklearn.model_selection import GroupShuffleSplit

from .typing import TrainValData, XType, YType


def split_by_indexes(
    X: XType,
    Y: YType,
    meta: pd.DataFrame,
    train_idx: npt.NDArray[np.int64],
    val_idx: npt.NDArray[np.int64],
) -> TrainValData:
    return (
        X[train_idx],
        X[val_idx],
        Y[train_idx],
        Y[val_idx],
        meta.iloc[train_idx],
        meta.iloc[val_idx],
    )


def maybe_shuffle(
    result: TrainValData, shuffle: bool, random_state: int | None = None
) -> TrainValData:
    if not shuffle:
        return result

    train_part = (result[0], result[2], result[4])
    train_part = sklearn.utils.shuffle(*train_part, random_state=random_state)

    val_part = (result[1], result[3], result[5])
    val_part = sklearn.utils.shuffle(*val_part, random_state=random_state)

    return (
        train_part[0],
        val_part[0],
        train_part[1],
        val_part[1],
        train_part[2],
        val_part[2],
    )  # type: ignore


def train_val_split(
    X: XType,
    Y: YType,
    meta: pd.DataFrame,
    groups: Any,
    test_size: float | None = None,
    shuffle: bool = False,
    random_state: int | None = None,
) -> TrainValData:
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)

    for train_idx, val_idx in gss.split(X, Y, groups):
        result = split_by_indexes(X, Y, meta, train_idx, val_idx)
        return maybe_shuffle(result, shuffle, random_state)

    raise RuntimeError("Unreachable")
