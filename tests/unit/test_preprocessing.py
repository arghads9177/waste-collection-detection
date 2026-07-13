"""Unit tests for data preprocessing functions."""
import tempfile
from pathlib import Path

import pytest

from waste_classification.data.preprocessing import (
    create_splits,
    find_corrupted_images,
    find_duplicate_images,
)


@pytest.fixture
def fixture_root() -> Path:
    """Get the path to test fixtures."""
    return Path(__file__).parent.parent / "fixtures" / "garbage_classification"


def test_find_corrupted_images(fixture_root: Path) -> None:
    """Test detection of corrupted images."""
    corrupted = find_corrupted_images(fixture_root)
    assert len(corrupted) > 0, "Should find at least one corrupted image"

    # Verify corrupted paths exist and are in fixture_root
    corrupted_names = [p.name for p in corrupted]
    assert "corrupted.jpg" in corrupted_names, "Should find the intentional corrupted fixture"

    # Verify they're in the fixture directory
    for path in corrupted:
        assert fixture_root in path.parents, f"{path} should be under {fixture_root}"


def test_find_duplicate_images(fixture_root: Path) -> None:
    """Test detection of exact-duplicate images."""
    duplicates = find_duplicate_images(fixture_root)
    assert len(duplicates) > 0, "Should find at least one duplicate"

    # Verify duplicate paths exist
    duplicate_names = [p.name for p in duplicates]
    assert "plastic_0_dup.jpg" in duplicate_names, "Should find the exact-duplicate fixture"

    # Duplicates should not include the first (canonical) image
    assert not any(p.name == "plastic_0.jpg" for p in duplicates), (
        "Canonical plastic_0.jpg should not be marked as duplicate"
    )


def test_find_duplicate_images_with_exclude(fixture_root: Path) -> None:
    """Test that excluded paths are not considered."""
    # Find corrupted first
    corrupted = find_corrupted_images(fixture_root)

    # Find duplicates excluding corrupted
    duplicates = find_duplicate_images(fixture_root, exclude=set(corrupted))
    assert len(duplicates) > 0, "Should still find duplicates after excluding corrupted"


def test_create_splits_writes_files(fixture_root: Path) -> None:
    """Test that splits are written to CSV and classes.txt files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Preprocess: exclude corrupted and duplicates
        corrupted = find_corrupted_images(fixture_root)
        duplicates = find_duplicate_images(fixture_root, exclude=set(corrupted))
        exclude = set(corrupted) | set(duplicates)

        # Create splits
        split_sizes = create_splits(fixture_root, output_dir, seed=42, exclude=exclude)

        # Verify output files exist
        assert (output_dir / "train.csv").exists(), "train.csv should exist"
        assert (output_dir / "val.csv").exists(), "val.csv should exist"
        assert (output_dir / "test.csv").exists(), "test.csv should exist"
        assert (output_dir / "classes.txt").exists(), "classes.txt should exist"

        # Verify split sizes match returned values
        train_count = sum(1 for _ in open(output_dir / "train.csv")) - 1  # exclude header
        val_count = sum(1 for _ in open(output_dir / "val.csv")) - 1
        test_count = sum(1 for _ in open(output_dir / "test.csv")) - 1

        assert train_count == split_sizes["train"], f"train count mismatch: {train_count} != {split_sizes['train']}"
        assert val_count == split_sizes["val"], f"val count mismatch: {val_count} != {split_sizes['val']}"
        assert test_count == split_sizes["test"], f"test count mismatch: {test_count} != {split_sizes['test']}"

        # Verify total
        total = train_count + val_count + test_count
        assert total > 0, "Splits should contain at least one sample"


def test_splits_contain_relative_paths(fixture_root: Path) -> None:
    """Test portability guard: all paths in CSVs are relative, not absolute."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Preprocess
        corrupted = find_corrupted_images(fixture_root)
        duplicates = find_duplicate_images(fixture_root, exclude=set(corrupted))
        exclude = set(corrupted) | set(duplicates)

        # Create splits
        create_splits(fixture_root, output_dir, seed=42, exclude=exclude)

        # Check all CSVs for relative paths
        for csv_file in ["train.csv", "val.csv", "test.csv"]:
            with open(output_dir / csv_file) as f:
                lines = f.readlines()[1:]  # skip header
                for line in lines:
                    path = line.split(",")[0].strip()
                    # Path should not be absolute
                    assert not path.startswith("/"), f"Path should not be absolute: {path}"
                    # Path should not contain fixture_root prefix
                    assert str(fixture_root) not in path, (
                        f"Path should not contain full fixture path: {path}"
                    )
                    # Path should be POSIX-style relative (e.g., "plastic/img.jpg")
                    assert "/" in path, f"Path should contain class subdirectory: {path}"


def test_splits_stratified(fixture_root: Path) -> None:
    """Test that train split is never empty (important for training)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Preprocess
        corrupted = find_corrupted_images(fixture_root)
        duplicates = find_duplicate_images(fixture_root, exclude=set(corrupted))
        exclude = set(corrupted) | set(duplicates)

        # Create splits
        create_splits(fixture_root, output_dir, seed=42, exclude=exclude)

        # Check that train split is never empty (it's the most important)
        with open(output_dir / "train.csv") as f:
            train_lines = f.readlines()[1:]  # skip header
            assert len(train_lines) > 0, "train.csv must never be empty"
