"""Integration tests for WasteDataset."""
import tempfile
from pathlib import Path

import pytest
import torch

from waste_classification.config import settings
from waste_classification.data.dataset import (
    WasteDataset,
    build_eval_transforms,
    build_train_transforms,
)
from waste_classification.data.preprocessing import (
    create_splits,
    find_corrupted_images,
    find_duplicate_images,
)


@pytest.fixture
def fixture_root() -> Path:
    """Get the path to test fixtures."""
    return Path(__file__).parent.parent / "fixtures" / "garbage_classification"


@pytest.fixture
def prepared_splits(fixture_root: Path) -> tuple[Path, Path]:
    """Create splits from fixtures and return (fixture_root, splits_dir)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Preprocess fixtures
        corrupted = find_corrupted_images(fixture_root)
        duplicates = find_duplicate_images(fixture_root, exclude=set(corrupted))
        exclude = set(corrupted) | set(duplicates)

        # Create splits
        create_splits(fixture_root, output_dir, seed=42, exclude=exclude)

        yield fixture_root, output_dir


def test_waste_dataset_load_and_iterate(
    prepared_splits: tuple[Path, Path],
) -> None:
    """Test loading WasteDataset and iterating samples."""
    fixture_root, splits_dir = prepared_splits

    # Create dataset with eval transforms
    dataset = WasteDataset(
        csv_path=splits_dir / "train.csv",
        data_root=fixture_root,
        transform=build_eval_transforms(settings.model_image_size),
    )

    assert len(dataset) > 0, "Dataset should contain samples"

    # Iterate a few samples
    for i in range(min(3, len(dataset))):
        img_tensor, label_idx = dataset[i]

        # Verify tensor shape and dtype
        assert isinstance(img_tensor, torch.Tensor), "Image should be a tensor"
        assert img_tensor.dtype == torch.float32, "Image should be float32"
        assert img_tensor.shape == (
            3,
            settings.model_image_size,
            settings.model_image_size,
        ), f"Image shape should be (3, {settings.model_image_size}, {settings.model_image_size}), got {img_tensor.shape}"

        # Verify label
        assert isinstance(label_idx, int), "Label should be an integer"
        assert 0 <= label_idx < len(dataset.class_names), (
            f"Label should be in [0, {len(dataset.class_names)}), got {label_idx}"
        )


def test_waste_dataset_class_names(prepared_splits: tuple[Path, Path]) -> None:
    """Test that class names are loaded correctly."""
    fixture_root, splits_dir = prepared_splits

    # Create dataset
    dataset = WasteDataset(
        csv_path=splits_dir / "train.csv",
        data_root=fixture_root,
    )

    # Verify class names are loaded
    assert len(dataset.class_names) > 0, "Should have at least one class"
    assert isinstance(dataset.class_names, list), "class_names should be a list"

    # Verify class_to_idx mapping
    assert len(dataset.class_to_idx) == len(dataset.class_names), (
        "class_to_idx should have one entry per class"
    )
    for class_name, idx in dataset.class_to_idx.items():
        assert class_name in dataset.class_names, f"class_to_idx references unknown class: {class_name}"
        assert idx == dataset.class_names.index(class_name), (
            f"class_to_idx mismatch for {class_name}"
        )


def test_waste_dataset_with_explicit_class_names(
    prepared_splits: tuple[Path, Path],
) -> None:
    """Test WasteDataset with explicit class_names parameter."""
    fixture_root, splits_dir = prepared_splits

    explicit_classes = ["plastic", "paper", "metal"]

    # Create dataset with explicit class names
    dataset = WasteDataset(
        csv_path=splits_dir / "train.csv",
        data_root=fixture_root,
        class_names=explicit_classes,
    )

    assert dataset.class_names == explicit_classes, "Should use provided class_names"
    assert len(dataset.class_to_idx) == len(explicit_classes), (
        "class_to_idx should match explicit class names"
    )


def test_build_train_transforms_augmentation() -> None:
    """Test that training transforms apply augmentation."""
    from PIL import Image
    import numpy as np

    transform = build_train_transforms(224)
    img = Image.fromarray(np.full((256, 256, 3), 128, dtype=np.uint8))

    # Apply transform
    tensor = transform(img)

    # Verify output
    assert isinstance(tensor, torch.Tensor), "Output should be a tensor"
    assert tensor.shape == (3, 224, 224), "Output shape should be (3, 224, 224)"
    assert tensor.dtype == torch.float32, "Output should be float32"
    assert tensor.min() >= -2.5, "Normalized values should be in reasonable range"
    assert tensor.max() <= 2.5, "Normalized values should be in reasonable range"


def test_build_eval_transforms_no_augmentation() -> None:
    """Test that eval transforms don't apply augmentation."""
    from PIL import Image
    import numpy as np

    transform = build_eval_transforms(224)
    img = Image.fromarray(np.full((256, 256, 3), 128, dtype=np.uint8))

    # Apply transform
    tensor = transform(img)

    # Verify output
    assert isinstance(tensor, torch.Tensor), "Output should be a tensor"
    assert tensor.shape == (3, 224, 224), "Output shape should be (3, 224, 224)"
    assert tensor.dtype == torch.float32, "Output should be float32"
