from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ecg_ml.data.utils.dataset import DatasetModule
from ecg_ml.data.utils.split import train_val_split
from ecg_ml.data.utils.typing import XType, YType
from ecg_ml.data.utils.yadisk import YaDiskDataMixin
from ecg_ml.utils.rank_logger import RankedLogger
from scipy.stats import zscore

# from ml_validation.filters import make_filter, FilterBuilder

from tqdm import tqdm

log = RankedLogger(__name__, rank_zero_only=True)


class ThreeBasesDatasetModule(DatasetModule, YaDiskDataMixin):
    name = "Three bases"

    resources = [
        ("three_bases_X.npy", "ed4dab3d0fa323cbfcccc1d1a938e526"),
        ("three_bases_Y.npy", "e61507cea4fe84d93857d957fb9266d1"),
        ("three_bases_noise_indexes.npy", "351c6b75e4c628e640d3ab7bea0239f0"),
        ("three_bases_zero_indexes.npy", "3685d64a09b346db29e3835fbeba04c7"),
        ("three_bases_meta.csv", "e4d19098c187283625b4a9aff6f39ff3"),
    ]

    train_test_ratio = 0.1
    train_test_seed = 42

    def __init__(
        self,
        path: Path,
        classes: List[int] = [0, 6],
        random_state: Optional[int] = None,
        train_val_ratio: float = 0.1,
        preprocessing = None
    ):
        super().__init__(path)

        self.classes = ["Norm(1)", "Afib(7)"]
        X, Y, meta = self._load_data(self.path, classes=classes)
        X = self._preprocess(X)

        groups = meta["Patient ID in source database"]
        X_train, self.X_test, Y_train, self.Y_test, meta_train, _ = train_val_split(
            X,
            Y,
            meta,
            groups,
            self.train_test_ratio,
            shuffle=False,
            random_state=self.train_test_seed,
        )

        del X, Y, meta

        train_groups = meta_train["Patient ID in source database"]
        self.X_train, self.X_val, self.Y_train, self.Y_val, _, _ = train_val_split(
            X_train,
            Y_train,
            meta_train,
            train_groups,
            train_val_ratio,
            shuffle=True,
            random_state=random_state,
        )
        # self.X_train = np.concat([self.X_train, X_avb1])
        # self.Y_train = np.concat([self.Y_train, Y_avb1])
        print(len(self.X_train), len(self.Y_train))

        with np.printoptions(threshold=np.inf):
            num_positives = self.Y_train.sum(axis=0)
            num_negatives = self.Y_train.shape[0] - self.Y_train.sum(axis=0)
            pos_weights = num_negatives / (num_positives + 1e-8)
            pos_weights[pos_weights > 1e5] = 1.0
            print("pos_weights=", pos_weights)
        self.pos_weight = pos_weights

    def train_dataset(self) -> Dataset:
        return torch.utils.data.TensorDataset(
            torch.from_numpy(self.X_train), torch.from_numpy(self.Y_train).float()
        )

    def val_dataset(self) -> Dataset:
        return torch.utils.data.TensorDataset(
            torch.from_numpy(self.X_val), torch.from_numpy(self.Y_val).float()
        )

    def test_dataset(self) -> Dataset:
        return torch.utils.data.TensorDataset(
            torch.from_numpy(self.X_test), torch.from_numpy(self.Y_test).float()
        )

    def download(self) -> None:
        self._yadisk_download(public_key="3GvQyz3wWMbK1w")
        self._yadisk_download(public_key="R3xTmgtjDRTqSQ")

    @staticmethod
    def _preprocess(X):
        # filter = make_filter(500)
        # X = filter(X)

        for j, s in tqdm(enumerate(X)):
            s_norm = zscore(s, axis=1, nan_policy="raise")
            same_values = np.all(s == s[:, 0][:, np.newaxis], axis=1)
            s_norm[same_values] = 0
            X[j] = s_norm
        return X

    @staticmethod
    def _load_data(path: Path, classes: List[int]) -> tuple[XType, YType, pd.DataFrame]:
        X = np.load(path / "three_bases_X.npy").astype(np.float32) / 1000
        Y = np.load(path / "three_bases_Y.npy")[:, classes]

        meta = pd.read_csv(path / "three_bases_meta.csv", index_col=0)

        assert len(X) == len(Y)
        assert len(X) == len(meta)

        zero = np.load(path / "three_bases_zero_indexes.npy")
        noise = np.load(path / "three_bases_noise_indexes.npy")

        good = np.ones(len(X), dtype=np.bool_)
        good[zero] = False
        good[noise] = False
        good[meta[meta["Source database name"] != "ptb_xl"].index.to_numpy()] = False

        return X[good], Y[good], meta.iloc[good]
