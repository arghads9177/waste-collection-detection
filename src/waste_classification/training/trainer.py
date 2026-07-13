"""Trainer: warmup (frozen backbone) + fine-tune (full network) training loop.

Driven entirely by function calls, no CLI — this is imported directly by the
Colab training notebook (Phase 3), not run as a script.
"""
import logging
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from waste_classification.models.factory import save_checkpoint, set_backbone_frozen
from waste_classification.training.metrics import compute_classification_metrics

logger = logging.getLogger(__name__)


class Trainer:
    """Encapsulates warmup + fine-tune training phases for a classification model."""

    def __init__(
        self,
        model: nn.Module,
        backbone: str,
        class_names: list[str],
        train_loader: DataLoader,
        val_loader: DataLoader,
        checkpoint_dir: Path | str,
        lr: float = 1e-3,
        early_stopping_patience: int = 10,
        class_weights: Optional[torch.Tensor] = None,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.backbone = backbone
        self.class_names = class_names
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.lr = lr
        self.early_stopping_patience = early_stopping_patience
        self.device = device

        weight = class_weights.to(device) if class_weights is not None else None
        self.criterion = nn.CrossEntropyLoss(weight=weight)

        self.best_val_accuracy = -1.0
        self.epochs_without_improvement = 0
        self.history: list[dict[str, Any]] = []

    def fit(self, warmup_epochs: int = 5, fine_tune_epochs: int = 0) -> dict[str, Any]:
        """Run warmup (frozen backbone) then fine-tune (full network) phases.

        Args:
            warmup_epochs: Epochs with the backbone frozen, only the head trained.
            fine_tune_epochs: Epochs with the full network trainable, at lr/10.

        Returns:
            Dict with 'history' (per-epoch metrics) and 'best_val_accuracy'.
        """
        total_epochs = warmup_epochs + fine_tune_epochs
        epoch_num = 0

        if warmup_epochs > 0:
            set_backbone_frozen(self.model, frozen=True)
            optimizer = AdamW(
                (p for p in self.model.parameters() if p.requires_grad), lr=self.lr
            )
            scheduler = CosineAnnealingLR(optimizer, T_max=warmup_epochs)
            for _ in range(warmup_epochs):
                epoch_num += 1
                self._train_one_epoch(optimizer, scheduler, epoch_num, total_epochs, phase="warmup")
                if self._should_stop():
                    return self._result()

        if fine_tune_epochs > 0:
            set_backbone_frozen(self.model, frozen=False)
            optimizer = AdamW(self.model.parameters(), lr=self.lr / 10)
            scheduler = CosineAnnealingLR(optimizer, T_max=fine_tune_epochs)
            for _ in range(fine_tune_epochs):
                epoch_num += 1
                self._train_one_epoch(optimizer, scheduler, epoch_num, total_epochs, phase="fine_tune")
                if self._should_stop():
                    return self._result()

        return self._result()

    def _run_epoch(
        self, loader: DataLoader, optimizer: Optional[torch.optim.Optimizer]
    ) -> dict[str, Any]:
        is_train = optimizer is not None
        self.model.train(is_train)

        total_loss = 0.0
        all_preds: list[int] = []
        all_labels: list[int] = []

        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)

            with torch.set_grad_enabled(is_train):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                if is_train:
                    assert optimizer is not None
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            total_loss += loss.item() * images.size(0)
            all_preds.extend(outputs.argmax(dim=1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

        metrics = compute_classification_metrics(all_labels, all_preds, self.class_names)
        metrics["loss"] = total_loss / len(loader.dataset)
        return metrics

    def _train_one_epoch(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        epoch_num: int,
        total_epochs: int,
        phase: str,
    ) -> None:
        train_metrics = self._run_epoch(self.train_loader, optimizer)
        val_metrics = self._run_epoch(self.val_loader, optimizer=None)
        scheduler.step()

        logger.info(
            f"[{phase}] epoch {epoch_num}/{total_epochs} | "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f}"
        )

        self.history.append(
            {
                "epoch": epoch_num,
                "phase": phase,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
            }
        )

        hyperparameters = {"lr": self.lr, "phase": phase, "epoch": epoch_num}
        save_checkpoint(
            self.checkpoint_dir / f"epoch_{epoch_num:03d}.pt",
            self.model,
            self.backbone,
            self.class_names,
            hyperparameters=hyperparameters,
            val_metrics=val_metrics,
        )

        if val_metrics["accuracy"] > self.best_val_accuracy:
            self.best_val_accuracy = val_metrics["accuracy"]
            self.epochs_without_improvement = 0
            save_checkpoint(
                self.checkpoint_dir / "best.pt",
                self.model,
                self.backbone,
                self.class_names,
                hyperparameters=hyperparameters,
                val_metrics=val_metrics,
            )
        else:
            self.epochs_without_improvement += 1

    def _should_stop(self) -> bool:
        if self.epochs_without_improvement >= self.early_stopping_patience:
            logger.info(
                f"Early stopping: no val improvement for {self.epochs_without_improvement} epochs"
            )
            return True
        return False

    def _result(self) -> dict[str, Any]:
        return {"history": self.history, "best_val_accuracy": self.best_val_accuracy}
