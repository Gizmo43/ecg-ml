from typing import Any

import lightning
import lightning.pytorch.loggers
import matplotlib.pyplot as plt
import numpy.typing as npt
import seaborn as sn
import torch
from lightning.pytorch.utilities.model_summary import summarize
from matplotlib.figure import Figure
from torchmetrics import MaxMetric, MeanMetric, MetricCollection, MinMetric
from torchmetrics.classification import (
    MultilabelConfusionMatrix,
    MultilabelF1Score,
    MultilabelPrecision,
    MultilabelRecall,
)


class ClassifierLModule(lightning.LightningModule):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: dict,
        criterion: torch.nn.Module,
        network: torch.nn.Module,
        labels: list[str] | None = None,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=True)

        self.network = network

        self.criterion = criterion

        self.labels = labels if labels is not None else []

        num_labels = len(self.labels)

        # Probability above which a label is predicted; applied only in `_binarize`.
        self.threshold = 0.5

        # The metrics get labels from `_binarize`, not logits: TorchMetrics would apply sigmoid
        # only to batches with values outside [0, 1], so F1 would depend on the batching.
        self.val_metrics = MetricCollection(
            {
                "precision": MultilabelPrecision(average=None, num_labels=num_labels),
                "recall": MultilabelRecall(average=None, num_labels=num_labels),
                "f1": MultilabelF1Score(average=None, num_labels=num_labels),
                "precision/macro": MultilabelPrecision(
                    average="macro", num_labels=num_labels
                ),
                "recall/macro": MultilabelRecall(
                    average="macro", num_labels=num_labels
                ),
                "f1/macro": MultilabelF1Score(average="macro", num_labels=num_labels),
            },
            prefix="val/",
        )

        self.val_loss = MeanMetric()
        # Best values over the epochs: an Optuna objective must not depend on the last epoch only
        self.val_loss_best = MinMetric()
        self.val_f1_macro_best = MaxMetric()

        self.test_metrics = self.val_metrics.clone(prefix="test/")
        self.test_confusion_matrix = MultilabelConfusionMatrix(num_labels=num_labels)

        self.test_preds = []
        self.test_trues = []

    def forward(self, x: torch.Tensor) -> Any:
        return self.network(x)

    def _binarize(self, logits: torch.Tensor) -> torch.Tensor:
        """0/1 labels from logits, the same for the metrics, the confusion matrix and the report.

        Sigmoid runs in float32, so half precision does not round probabilities onto the threshold.
        """
        return (torch.sigmoid(logits.float()) > self.threshold).int()

    def configure_optimizers(self) -> dict[str, Any]:
        optimizers_conf = {}
        if self.hparams.get("optimizer") is not None:
            optimizers_conf["optimizer"] = self.hparams["optimizer"](
                self.network.parameters()
            )

        if self.hparams.get("lr_scheduler") is not None:
            scheduler = self.hparams["lr_scheduler"]["scheduler"](
                optimizers_conf["optimizer"]
            )

            optimizers_conf["lr_scheduler"] = dict(self.hparams["lr_scheduler"])
            optimizers_conf["lr_scheduler"]["scheduler"] = scheduler

        return optimizers_conf

    def on_train_start(self) -> None:
        # by default lightning executes validation and test steps sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_metrics.reset()
        self.val_loss.reset()
        self.val_loss_best.reset()
        self.val_f1_macro_best.reset()
        self.test_metrics.reset()
        self.test_confusion_matrix.reset()

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        # trainer.fit/test(ckpt_path=...) restore only the weights; the labels come from the dataset
        labels = checkpoint.get(self.CHECKPOINT_HYPER_PARAMS_KEY, {}).get("labels")
        if labels is not None and [str(c) for c in labels] != [
            str(c) for c in self.labels
        ]:
            raise ValueError(
                f"The checkpoint is trained for labels {list(labels)}, "
                f"but the dataset classes are {self.labels}"
            )

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> torch.Tensor:
        x, y_true = batch
        y_pred = self.forward(x)

        loss = self.criterion(y_pred, y_true)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> torch.Tensor:
        x, y_true = batch
        y_pred = self.forward(x)

        loss = self.criterion(y_pred, y_true)
        self.val_loss.update(loss, weight=len(y_true))

        self.val_metrics.update(self._binarize(y_pred), y_true.int())

        return loss

    def on_validation_epoch_end(self) -> None:
        loss = self.val_loss.compute()
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.val_loss.reset()

        metrics_dict = self._metrics_to_plain_dict(
            self.val_metrics.compute(), labels=self.labels
        )
        self.val_metrics.reset()

        self.val_loss_best.update(loss)
        self.val_f1_macro_best.update(metrics_dict["val/f1/macro"])
        metrics_dict["val/loss_best"] = self.val_loss_best.compute()
        metrics_dict["val/f1/macro_best"] = self.val_f1_macro_best.compute()

        self.log_dict(metrics_dict, on_step=False, on_epoch=True, prog_bar=False)

    def on_test_epoch_start(self) -> None:
        # a previous trainer.test() that failed midway could leave its predictions here
        self.test_preds.clear()
        self.test_trues.clear()

    def test_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        x, y_true = batch
        y_pred = self._binarize(self.forward(x))
        y_true = y_true.int()

        self.test_metrics.update(y_pred, y_true)
        self.test_confusion_matrix.update(y_pred, y_true)

        self.test_preds.append(y_pred.cpu())
        self.test_trues.append(y_true.cpu())

    def on_test_epoch_end(self) -> None:
        metrics_dict = self._metrics_to_plain_dict(
            self.test_metrics.compute(), labels=self.labels
        )
        self.log_dict(metrics_dict, on_step=False, on_epoch=True, prog_bar=False)
        self.test_metrics.reset()

        confusion_matrix = self.test_confusion_matrix.compute()
        self.log_confusion_matrix(
            confusion_matrix, artifact_file="test/confusion_matrix.png"
        )
        self.test_confusion_matrix.reset()

        Y_preds = torch.cat(self.test_preds).numpy()
        Y_trues = torch.cat(self.test_trues).numpy()
        self.test_preds.clear()
        self.test_trues.clear()

        from sklearn.metrics import classification_report

        # zero_division=0 is what TorchMetrics returns for a label without predictions
        report = classification_report(
            y_true=Y_trues,
            y_pred=Y_preds,
            target_names=[str(label) for label in self.labels],
            zero_division=0,
        )
        print(report)

    def log_summary(self) -> None:
        model_summary = str(summarize(lightning_module=self, max_depth=-1))  # type: ignore
        if isinstance(self.logger, lightning.pytorch.loggers.MLFlowLogger):
            self.logger.experiment.log_text(
                run_id=self.logger.run_id,
                text=model_summary,
                artifact_file="summary.txt",
            )

    def log_confusion_matrix(
        self, confusion_matrix: torch.Tensor, artifact_file: str
    ) -> None:
        if isinstance(self.logger, lightning.pytorch.loggers.MLFlowLogger):
            fig = self._confusion_matrix_figure(
                confusion_matrix.numpy(force=True), labels=self.labels
            )
            self.logger.experiment.log_figure(
                run_id=self.logger.run_id, figure=fig, artifact_file=artifact_file
            )
            plt.close(fig)

    @staticmethod
    def _metrics_to_plain_dict(
        metrics: dict[str, Any], labels: list[str]
    ) -> dict[str, torch.Tensor]:
        metrics_dict = {}
        for metric_name, metric_values in metrics.items():
            if metric_values.shape:
                for metric_label, metric_value in zip(labels, metric_values):
                    metrics_dict[f"{metric_name}/{metric_label}"] = metric_value
            else:
                metrics_dict[metric_name] = metric_values

        return metrics_dict

    @staticmethod
    def _confusion_matrix_figure(
        confusion_matrix: npt.NDArray, labels: list[str]
    ) -> Figure:
        fig, axs = plt.subplots(ncols=int(confusion_matrix.shape[0]), figsize=(10, 5))
        for idx, label_confusion_matrix in enumerate(confusion_matrix):
            sn.heatmap(
                label_confusion_matrix,
                ax=axs[idx],
                annot=True,
                cmap=plt.get_cmap("YlGn"),
                cbar=False,
                fmt=".10g",
                square=True,
            )
            axs[idx].set_title(f"{labels[idx]}")
            axs[idx].set_xlabel("Predicted")
            axs[idx].set_ylabel("Actual")
        fig.tight_layout()

        return fig
