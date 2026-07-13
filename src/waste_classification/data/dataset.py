"""PyTorch Dataset and augmentation transforms for waste classification."""
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
import torchvision.transforms.v2 as transforms
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class WasteDataset(Dataset):
    """Dataset for waste classification from split CSV files."""

    def __init__(
        self,
        csv_path: Path | str,
        data_root: Path | str,
        transform: Optional[transforms.Compose] = None,
        class_names: Optional[list[str]] = None,
    ):
        """Initialize WasteDataset.

        Args:
            csv_path: Path to split CSV (must have 'path' and 'label' columns).
            data_root: Root directory where image paths are relative to.
            transform: torchvision transforms to apply to images.
            class_names: Sorted list of class names. If None, loads from classes.txt
                        sibling to csv_path, or falls back to sorted unique labels in CSV.
        """
        self.csv_path = Path(csv_path)
        self.data_root = Path(data_root)
        self.transform = transform

        # Load split CSV
        self.df = pd.read_csv(self.csv_path)

        # Load or derive class list
        if class_names is not None:
            self.class_names = class_names
        else:
            classes_txt = self.csv_path.parent / "classes.txt"
            if classes_txt.exists():
                with open(classes_txt) as f:
                    self.class_names = [line.strip() for line in f if line.strip()]
                logger.info(f"Loaded class names from {classes_txt}")
            else:
                # Fallback: sorted unique labels in CSV (less reliable)
                self.class_names = sorted(self.df["label"].unique().tolist())
                logger.warning(
                    f"No classes.txt found; derived class list from CSV labels: {self.class_names}"
                )

        # Build class -> index mapping
        self.class_to_idx = {cls: i for i, cls in enumerate(self.class_names)}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        img_rel_path = row["path"]
        label_name = row["label"]

        # Open image
        img_path = self.data_root / img_rel_path
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.error(f"Failed to load {img_path}: {e}")
            raise

        # Apply transform
        if self.transform is not None:
            img = self.transform(img)

        # Convert label name to index
        label_idx = self.class_to_idx[label_name]

        return img, label_idx


def build_train_transforms(image_size: int) -> transforms.Compose:
    """Build augmentation transforms for training.

    Args:
        image_size: Size to resize/crop images to (e.g. 224).

    Returns:
        Composed transforms for training data.
    """
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                size=(image_size, image_size),
                scale=(0.8, 1.0),
                ratio=(0.9, 1.1),
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_eval_transforms(image_size: int) -> transforms.Compose:
    """Build transforms for evaluation (no augmentation).

    Args:
        image_size: Size to resize/crop images to (e.g. 224).

    Returns:
        Composed transforms for eval data.
    """
    return transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.15)),
            transforms.CenterCrop((image_size, image_size)),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
