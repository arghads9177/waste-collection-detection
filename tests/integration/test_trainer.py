"""Integration test: Trainer.fit() for one epoch on a tiny fixture set (CPU smoke).

Proves the training loop (forward/backward, checkpointing, freeze/unfreeze phases)
works end-to-end before it's imported into the Colab notebook (Phase 3) and used
for real, GPU-hours-costing training runs.
"""
from pathlib import Path

import pandas as pd
import pytest
from torch.utils.data import DataLoader

from waste_classification.data.dataset import WasteDataset, build_eval_transforms
from waste_classification.models.factory import build_model, load_checkpoint
from waste_classification.training.trainer import Trainer


@pytest.fixture
def fixture_root() -> Path:
    return Path(__file__).parent.parent / "fixtures" / "garbage_classification"


@pytest.fixture
def tiny_two_class_csv(fixture_root: Path, tmp_path: Path) -> Path:
    """4-image, 2-class CSV built directly from committed fixtures."""
    rows = [
        {"path": "paper/paper_0.jpg", "label": "paper"},
        {"path": "paper/paper_1.jpg", "label": "paper"},
        {"path": "metal/metal_0.jpg", "label": "metal"},
        {"path": "metal/metal_1.jpg", "label": "metal"},
    ]
    csv_path = tmp_path / "tiny.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


def test_trainer_fit_one_epoch_cpu_smoke(
    fixture_root: Path, tiny_two_class_csv: Path, tmp_path: Path
) -> None:
    class_names = ["metal", "paper"]
    transform = build_eval_transforms(64)  # small size to keep the CPU smoke test fast

    dataset = WasteDataset(
        csv_path=tiny_two_class_csv,
        data_root=fixture_root,
        transform=transform,
        class_names=class_names,
    )
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    model = build_model("efficientnet_b0", num_classes=2, pretrained=False)
    checkpoint_dir = tmp_path / "checkpoints"

    trainer = Trainer(
        model=model,
        backbone="efficientnet_b0",
        class_names=class_names,
        train_loader=loader,
        val_loader=loader,
        checkpoint_dir=checkpoint_dir,
        lr=1e-3,
        early_stopping_patience=5,
        device="cpu",
    )

    result = trainer.fit(warmup_epochs=1, fine_tune_epochs=0)

    assert len(result["history"]) == 1
    assert result["history"][0]["phase"] == "warmup"
    assert 0.0 <= result["best_val_accuracy"] <= 1.0

    assert (checkpoint_dir / "epoch_001.pt").exists()
    assert (checkpoint_dir / "best.pt").exists()

    # Best checkpoint must be self-describing enough to rebuild without .env.
    _, checkpoint = load_checkpoint(checkpoint_dir / "best.pt")
    assert checkpoint["backbone"] == "efficientnet_b0"
    assert checkpoint["class_names"] == class_names
    assert checkpoint["num_classes"] == 2
    assert "val_metrics" in checkpoint
    assert "hyperparameters" in checkpoint
