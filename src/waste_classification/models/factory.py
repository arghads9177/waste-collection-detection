"""Model factory: build classification backbones, save/load self-describing checkpoints."""
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    EfficientNet_B0_Weights,
    convnext_tiny,
    efficientnet_b0,
)

logger = logging.getLogger(__name__)

SUPPORTED_BACKBONES = ("efficientnet_b0", "convnext_tiny")


def build_model(backbone: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """Build a classification model with its head replaced for num_classes outputs.

    Args:
        backbone: One of SUPPORTED_BACKBONES.
        num_classes: Number of output classes.
        pretrained: Load ImageNet-pretrained weights.

    Returns:
        The model, with `.classifier[-1]` replaced by a fresh Linear(*, num_classes).
    """
    if backbone == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = efficientnet_b0(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    elif backbone == "convnext_tiny":
        weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = convnext_tiny(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(
            f"Unknown backbone: {backbone!r}. Supported: {SUPPORTED_BACKBONES}"
        )

    logger.info(f"Built {backbone} with {num_classes} output classes (pretrained={pretrained})")
    return model


def set_backbone_frozen(model: nn.Module, frozen: bool) -> None:
    """Freeze/unfreeze all params except the classification head (`model.classifier`).

    Both supported backbones expose their head as `model.classifier`, so this
    works uniformly regardless of which one was built.
    """
    head_param_ids = {id(p) for p in model.classifier.parameters()}

    for p in model.parameters():
        p.requires_grad = True if id(p) in head_param_ids else not frozen

    logger.info(f"Backbone {'frozen' if frozen else 'unfrozen'} (classification head always trainable)")


def save_checkpoint(
    path: Path | str,
    model: nn.Module,
    backbone: str,
    class_names: list[str],
    hyperparameters: dict[str, Any],
    val_metrics: dict[str, Any],
) -> None:
    """Save a self-describing checkpoint dict that can rebuild its own architecture."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "backbone": backbone,
            "num_classes": len(class_names),
            "class_names": class_names,
            "state_dict": model.state_dict(),
            "hyperparameters": hyperparameters,
            "val_metrics": val_metrics,
        },
        path,
    )
    logger.info(f"Saved checkpoint to {path}")


def load_checkpoint(path: Path | str, device: str = "cpu") -> tuple[nn.Module, dict[str, Any]]:
    """Load a self-describing checkpoint, rebuilding the architecture from the file itself.

    Args:
        path: Path to a checkpoint saved by `save_checkpoint()`.
        device: Device to load the model onto.

    Returns:
        (model, checkpoint_dict) — model has weights loaded and is on `device`.
    """
    checkpoint = torch.load(Path(path), map_location=device)
    model = build_model(checkpoint["backbone"], checkpoint["num_classes"], pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    logger.info(f"Loaded checkpoint from {path} (backbone={checkpoint['backbone']})")
    return model, checkpoint
