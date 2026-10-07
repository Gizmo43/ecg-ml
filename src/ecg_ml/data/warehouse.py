from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ecg_ml.data.preprocessors import Preprocessor
from ecg_ml.data.utils.dataset import DatasetModule
from ecg_ml.data.utils.hash import calculate_sha256
from ecg_ml.data.utils.split import train_val_split
from ecg_ml.data.utils.typing import XType, YType
from ecg_ml.utils.rank_logger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

NPZ_FILE = "signals.npz"
META_CSV = "meta.csv"

TRAIN_SPLIT = "train"
TEST_SPLIT = "test"


class WarehouseDatasetModule(DatasetModule):
    """A dataset exported by `ecg-warehouse dataset export <name>@<version> --format npz --output <path>`.

    `test` is the warehouse split (`split == "test"`); `val` is carved out of `train` by patient,
    grouping recordings by `(source, patient_id)`. A patient in both `train` and `test` is an error.
    """

    def __init__(
        self,
        path: Path | str,
        preprocessing: Preprocessor | None = None,
        random_state: int | None = None,
        train_val_ratio: float = 0.1,
    ):
        self.path = Path(path)

        X, Y, meta, self.classes = self._load_data(self.path)
        fs = (
            self._sampling_rate(X, meta)
            if preprocessing and preprocessing.requires_sampling_rate
            else None
        )
        if preprocessing:
            X = preprocessing(X, fs)

        groups = self._patient_groups(meta)
        split = meta["split"].to_numpy()
        unknown = sorted(set(split) - {TRAIN_SPLIT, TEST_SPLIT})
        if unknown:
            raise ValueError(
                f"Unexpected split values {unknown} in {self.path / NPZ_FILE}"
            )

        train = split == TRAIN_SPLIT
        # The export splits by itself, so a patient on both sides would leak test into training.
        shared = np.intersect1d(groups[train], groups[~train])
        if len(shared):
            example = shared[0].replace("\x1f", "/")
            raise ValueError(
                f"{len(shared)} patient(s) have recordings in both {TRAIN_SPLIT} and {TEST_SPLIT} "
                f"of {self.path}, e.g. source/patient_id {example!r}"
            )

        self.X_test, self.Y_test = X[~train], Y[~train]
        meta_test = meta[~train]
        X_train, Y_train, meta_train = X[train], Y[train], meta[train]
        del X, Y, meta

        # The dataloader shuffles train itself, and shuffling here would copy the signals once more.
        self.X_train, self.X_val, self.Y_train, self.Y_val, meta_train, meta_val = (
            train_val_split(
                X_train,
                Y_train,
                meta_train,
                groups[train],
                train_val_ratio,
                shuffle=False,
                random_state=random_state,
            )
        )
        del X_train, Y_train

        for part, Y_part, meta_part in (
            ("train", self.Y_train, meta_train),
            ("val", self.Y_val, meta_val),
            ("test", self.Y_test, meta_test),
        ):
            self._log_part(part, Y_part, groups[meta_part.index])

        self.pos_weight = self._pos_weight(self.Y_train)
        log.info(
            f"pos_weight = train negatives / positives for {self.classes}: "
            f"{[round(float(w), 4) for w in self.pos_weight]}"
        )

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

    def fingerprint(self) -> dict[str, Any]:
        """Content hashes of the export and the part sizes; the split parameters are in the config."""
        return {
            **super().fingerprint(),
            "sha256": {
                name: calculate_sha256(self.path / name)
                for name in (NPZ_FILE, META_CSV)
            },
            "recordings": {
                "train": len(self.Y_train),
                "val": len(self.Y_val),
                "test": len(self.Y_test),
            },
        }

    @staticmethod
    def _load_data(path: Path) -> tuple[XType, YType, pd.DataFrame, list[str]]:
        """Read signals.npz and meta.csv, reordering meta rows to the npz rows by `recording_id`.

        meta keeps a positional index, so `groups[meta.index]` follows it through the splits.
        """
        with np.load(path / NPZ_FILE) as data:
            missing = [
                key for key in ("X", "y", "classes", "recording_id") if key not in data
            ]
            if missing:
                raise ValueError(
                    f"{path / NPZ_FILE} has no {', '.join(missing)} array(s)"
                )
            if "split" not in data:
                raise ValueError(
                    f"{path / NPZ_FILE} has no split array: build the dataset with a `split` in its spec"
                )
            X = data["X"].astype(np.float32)
            Y = data["y"].astype(np.bool_)
            classes = data["classes"].tolist()
            recording_id = data["recording_id"].astype(str)
            split = data["split"].astype(str)

        if not (len(X) == len(Y) == len(recording_id) == len(split)):
            raise ValueError(
                f"{path / NPZ_FILE} arrays disagree in length: X {len(X)}, y {len(Y)}, "
                f"recording_id {len(recording_id)}, split {len(split)}"
            )
        if Y.shape[1] != len(classes):
            raise ValueError(
                f"y has {Y.shape[1]} columns, but there are {len(classes)} classes"
            )
        if not np.isfinite(X).all():
            raise ValueError(f"{path / NPZ_FILE} has NaN or inf values in X")

        meta = pd.read_csv(
            path / META_CSV,
            dtype={"recording_id": str, "source": str, "patient_id": str, "split": str},
        )
        missing = [
            c for c in ("recording_id", "source", "patient_id") if c not in meta.columns
        ]
        if missing:
            raise ValueError(
                f"{path / META_CSV} has no {', '.join(missing)} column(s): add them to the spec's recording_columns"
            )
        if meta["recording_id"].duplicated().any():
            raise ValueError(f"{path / META_CSV} has duplicate recording_id values")
        if len(set(recording_id)) != len(recording_id):
            raise ValueError(f"{path / NPZ_FILE} has duplicate recording_id values")

        meta = meta.set_index("recording_id")
        absent = pd.Index(recording_id).difference(meta.index)
        if len(absent):
            raise ValueError(
                f"{len(absent)} recording(s) of {NPZ_FILE} are not in {META_CSV}, e.g. {absent[0]!r}"
            )
        meta = meta.loc[recording_id].reset_index()
        if "split" in meta.columns and not (meta["split"].to_numpy() == split).all():
            raise ValueError(f"split in {META_CSV} disagrees with split in {NPZ_FILE}")
        meta["split"] = split

        log.info(
            f"Loaded {len(X)} recordings {X.shape[1:]} with classes {classes} from {path}"
        )
        return X, Y, meta, classes

    @staticmethod
    def _patient_groups(meta: pd.DataFrame) -> np.ndarray:
        """`(source, patient_id)` per recording; a recording without patient_id is its own group."""
        patient = meta["source"].fillna("") + "\x1f" + meta["patient_id"]
        alone = "\x1frecording\x1f" + meta["recording_id"]
        return patient.where(meta["patient_id"].notna(), alone).to_numpy()

    @staticmethod
    def _sampling_rate(X: XType, meta: pd.DataFrame) -> float:
        if "sampling_rate_hz" in meta.columns:
            rates = meta["sampling_rate_hz"].unique()
        elif "duration_sec" in meta.columns:
            rates = np.unique(X.shape[-1] / meta["duration_sec"].to_numpy())
        else:
            raise ValueError(
                f"Filtering needs a sampling_rate_hz or duration_sec column in {META_CSV}"
            )
        if len(rates) != 1:
            raise ValueError(f"Recordings have different sampling rates: {rates}")
        return float(rates[0])

    def _pos_weight(self, Y: YType) -> np.ndarray:
        positives = Y.sum(axis=0).astype(np.float64)
        negatives = len(Y) - positives
        pos_weight = np.divide(
            negatives,
            positives,
            out=np.full_like(positives, np.nan),
            where=positives > 0,
        )
        if (positives == 0).any():
            absent = [c for c, k in zip(self.classes, positives) if k == 0]
            log.warning(f"Classes without train positives get pos_weight NaN: {absent}")
        return pos_weight

    def _log_part(self, part: str, Y: YType, groups: np.ndarray) -> None:
        n = len(Y)
        counts = Y.sum(axis=0)
        frequencies = ", ".join(
            f"{c}={k} ({k / max(n, 1):.1%})" for c, k in zip(self.classes, counts)
        )
        unlabeled = int((~Y.any(axis=1)).sum())
        log.info(
            f"{part}: {n} recordings, {len(np.unique(groups))} patients, "
            f"{unlabeled} without labels; classes: {frequencies}"
        )
