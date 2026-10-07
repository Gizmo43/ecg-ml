from typing import Any

from lightning import LightningDataModule
from torch.utils.data import Dataset


class DatasetModule:
    # Names of the Y columns, in order
    classes: list[str]

    def train_dataset(self) -> Dataset: ...
    def val_dataset(self) -> Dataset: ...
    def test_dataset(self) -> Dataset: ...

    def fingerprint(self) -> dict[str, Any]:
        """What tells this data apart from other data; logged with the run hyperparameters."""
        return {"classes": list(self.classes)}

    def to_datamodule(
        self, batch_size: int = 1, num_workers: int = 0, **datamodule_kwargs
    ):
        return LightningDataModule.from_datasets(
            train_dataset=self.train_dataset(),
            val_dataset=self.val_dataset(),
            test_dataset=self.test_dataset(),
            batch_size=batch_size,
            num_workers=num_workers,
            **datamodule_kwargs,
        )
