"""Preprocess data: detect and remove corrupted/duplicate images, create train/val/test splits.

Usage:
    uv run python scripts/preprocess.py [--data-root ...] [--output-dir ...]
"""
import argparse
import logging
from pathlib import Path

from waste_classification.config import settings
from waste_classification.data.preprocessing import (
    create_splits,
    find_corrupted_images,
    find_duplicate_images,
)
from waste_classification.logging_config import setup_logging

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Preprocess waste classification dataset")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(settings.data_raw_dir),
        help=f"Path to raw dataset root (default: {settings.data_raw_dir})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(settings.data_processed_dir) / "splits",
        help=f"Path to write splits (default: {Path(settings.data_processed_dir) / 'splits'})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=settings.data_split_seed,
        help=f"Random seed for splits (default: {settings.data_split_seed})",
    )
    args = parser.parse_args()

    # Setup logging
    setup_logging(settings.log_level, settings.log_file)

    logger.info("=== Data Preprocessing ===")
    logger.info(f"Data root: {args.data_root}")
    logger.info(f"Output dir: {args.output_dir}")

    # Validate data root exists
    if not args.data_root.exists():
        logger.error(f"Data root does not exist: {args.data_root}")
        return 1

    # Step 1: Find corrupted images
    logger.info("\n[1/3] Finding corrupted images...")
    corrupted = find_corrupted_images(args.data_root)

    # Step 2: Find duplicate images
    logger.info("\n[2/3] Finding duplicate images...")
    duplicates = find_duplicate_images(args.data_root, exclude=set(corrupted))

    # Step 3: Create splits (excluding corrupted + duplicates)
    logger.info("\n[3/3] Creating stratified splits...")
    exclude = set(corrupted) | set(duplicates)
    split_sizes = create_splits(args.data_root, args.output_dir, args.seed, exclude=exclude)

    # Summary
    logger.info("\n=== Summary ===")
    logger.info(f"Corrupted images removed: {len(corrupted)}")
    logger.info(f"Duplicate images removed: {len(duplicates)}")
    logger.info(f"Train split: {split_sizes['train']} samples")
    logger.info(f"Val split: {split_sizes['val']} samples")
    logger.info(f"Test split: {split_sizes['test']} samples")
    logger.info(f"Total: {sum(split_sizes.values())} samples")

    logger.info("\n✓ Preprocessing complete!")
    return 0


if __name__ == "__main__":
    exit(main())
