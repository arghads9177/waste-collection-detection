"""Data preprocessing: corrupted/duplicate detection, stratified train/val/test splits."""
import hashlib
import logging
from pathlib import Path
from typing import Optional

import imagehash
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def find_corrupted_images(root: Path) -> list[Path]:
    """Find images that PIL cannot open/verify.

    Args:
        root: Path to dataset root (e.g. data/raw/garbage_classification).
              Assumes <root>/<class>/*.jpg structure.

    Returns:
        List of corrupt image paths.
    """
    corrupted = []
    root = Path(root)

    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir():
            continue

        class_corrupted = 0
        for img_path in sorted(class_dir.glob("*.jpg")):
            try:
                with Image.open(img_path) as img:
                    img.verify()
            except Exception as e:
                logger.warning(f"Corrupted image {img_path}: {e}")
                corrupted.append(img_path)
                class_corrupted += 1

        if class_corrupted > 0:
            logger.info(f"  {class_dir.name}: removed {class_corrupted} corrupted")

    logger.info(f"Total corrupted images removed: {len(corrupted)}")
    return corrupted


def find_duplicate_images(
    root: Path,
    exclude: Optional[set[Path]] = None,
    perceptual_threshold: Optional[int] = None,
) -> list[Path]:
    """Find and remove exact duplicates (MD5) and optionally near-duplicates (perceptual hash).

    Args:
        root: Path to dataset root.
        exclude: Set of paths to skip (e.g., already-identified corrupted images).
        perceptual_threshold: If set, also flag images with perceptual hash Hamming
                            distance < this threshold as duplicates. None = skip perceptual pass.

    Returns:
        List of duplicate image paths (the non-canonical copies to remove).
    """
    exclude = exclude or set()
    root = Path(root)
    duplicates = []

    # MD5 exact-match pass
    md5_to_images = {}
    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir():
            continue

        for img_path in sorted(class_dir.glob("*.jpg")):
            if img_path in exclude:
                continue

            try:
                with open(img_path, "rb") as f:
                    md5 = hashlib.md5(f.read()).hexdigest()
                if md5 not in md5_to_images:
                    md5_to_images[md5] = []
                md5_to_images[md5].append(img_path)
            except Exception as e:
                logger.warning(f"Could not hash {img_path}: {e}")

    # Keep first (sorted) image per MD5, mark rest as duplicates
    class_duplicates = {}
    for md5, paths in md5_to_images.items():
        if len(paths) > 1:
            for dup in paths[1:]:
                duplicates.append(dup)
                class_name = dup.parent.name
                class_duplicates[class_name] = class_duplicates.get(class_name, 0) + 1

    for class_name, count in sorted(class_duplicates.items()):
        logger.info(f"  {class_name}: removed {count} exact duplicates")

    logger.info(f"Total exact duplicates removed: {len(duplicates)}")

    # Perceptual hash pass (optional, off by default)
    if perceptual_threshold is not None:
        phash_duplicates = []
        valid_images = [
            p
            for class_dir in root.iterdir()
            if class_dir.is_dir()
            for p in sorted(class_dir.glob("*.jpg"))
            if p not in exclude and p not in duplicates
        ]

        phash_to_images = {}
        for img_path in valid_images:
            try:
                phash = imagehash.phash(Image.open(img_path))
                if phash not in phash_to_images:
                    phash_to_images[phash] = []
                phash_to_images[phash].append(img_path)
            except Exception as e:
                logger.warning(f"Could not compute perceptual hash for {img_path}: {e}")

        # Cluster near-duplicates by Hamming distance
        processed = set()
        for phash1, paths1 in phash_to_images.items():
            if phash1 in processed:
                continue

            cluster = set(paths1)
            for phash2, paths2 in phash_to_images.items():
                if phash2 in processed or phash2 == phash1:
                    continue
                if phash1 - phash2 < perceptual_threshold:
                    cluster.update(paths2)
                    processed.add(phash2)

            # Keep first, mark rest as near-duplicates
            if len(cluster) > 1:
                for dup in sorted(cluster)[1:]:
                    phash_duplicates.append(dup)

            processed.add(phash1)

        class_phash_duplicates = {}
        for dup in phash_duplicates:
            class_name = dup.parent.name
            class_phash_duplicates[class_name] = class_phash_duplicates.get(class_name, 0) + 1

        for class_name, count in sorted(class_phash_duplicates.items()):
            logger.info(f"  {class_name}: removed {count} near-duplicates (phash)")

        logger.info(f"Total near-duplicates removed: {len(phash_duplicates)}")
        duplicates.extend(phash_duplicates)

    return duplicates


def create_splits(
    root: Path,
    output_dir: Path,
    seed: int,
    exclude: Optional[set[Path]] = None,
) -> dict[str, int]:
    """Create stratified 70/15/15 train/val/test splits, write CSVs with relative paths.

    Args:
        root: Path to dataset root.
        output_dir: Directory to write {train,val,test}.csv and classes.txt.
        seed: Random seed for reproducibility.
        exclude: Set of image paths to skip (e.g., corrupted + duplicates).

    Returns:
        Dict with 'train', 'val', 'test' split sizes.
    """
    exclude = exclude or set()
    root = Path(root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect valid images per class
    class_images = {}
    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir():
            continue

        images = [
            p
            for p in sorted(class_dir.glob("*.jpg"))
            if p not in exclude and p.is_file()
        ]
        if images:
            class_images[class_dir.name] = images

    logger.info(f"Found {sum(len(v) for v in class_images.values())} valid images across {len(class_images)} classes")
    for class_name, images in sorted(class_images.items()):
        logger.info(f"  {class_name}: {len(images)} images")

    # Build dataframe with paths and labels
    data = []
    for class_name, images in sorted(class_images.items()):
        for img_path in images:
            # Store as relative path (POSIX style, e.g. "plastic/img.jpg")
            rel_path = img_path.relative_to(root).as_posix()
            data.append({"path": rel_path, "label": class_name})

    df = pd.DataFrame(data)
    total_samples = len(df)
    logger.info(f"Total samples: {total_samples}")

    # For very small datasets (< 3 samples), put all in train
    if total_samples < 3:
        logger.warning(
            f"Dataset too small ({total_samples} samples); putting all in train split"
        )
        train_df = df
        val_df = pd.DataFrame(columns=["path", "label"])
        test_df = pd.DataFrame(columns=["path", "label"])
    else:
        # Stratified 70/15/15 split (two-stage: first 70/30, then split 30 into 50/50)
        # For datasets with 3-5 samples, use simpler splits
        min_class_count = min(
            (df["label"] == cls).sum() for cls in df["label"].unique()
        )

        if min_class_count < 2 or total_samples < 6:
            logger.warning(
                f"Dataset too small for stratified split (min class count: {min_class_count}, "
                f"total: {total_samples}); falling back to simple random split"
            )
            # Random 70/30 first
            train_df, temp_df = train_test_split(
                df, test_size=0.30, random_state=seed
            )
            # If temp_df has at least 2 samples, split into val/test
            if len(temp_df) >= 2:
                val_df, test_df = train_test_split(
                    temp_df, test_size=0.50, random_state=seed
                )
            else:
                # Only 1 sample in temp: put it in val
                val_df = temp_df
                test_df = pd.DataFrame(columns=["path", "label"])
        else:
            # Stratified split
            train_df, temp_df = train_test_split(
                df,
                test_size=0.30,
                random_state=seed,
                stratify=df["label"],
            )

            val_df, test_df = train_test_split(
                temp_df,
                test_size=0.50,
                random_state=seed,
                stratify=temp_df["label"],
            )

    # Write CSVs
    train_csv = output_dir / "train.csv"
    val_csv = output_dir / "val.csv"
    test_csv = output_dir / "test.csv"

    train_df[["path", "label"]].to_csv(train_csv, index=False)
    val_df[["path", "label"]].to_csv(val_csv, index=False)
    test_df[["path", "label"]].to_csv(test_csv, index=False)

    logger.info(f"Wrote {train_csv}: {len(train_df)} samples")
    logger.info(f"Wrote {val_csv}: {len(val_df)} samples")
    logger.info(f"Wrote {test_csv}: {len(test_df)} samples")

    # Write class list (sorted)
    classes_txt = output_dir / "classes.txt"
    with open(classes_txt, "w") as f:
        for class_name in sorted(class_images.keys()):
            f.write(f"{class_name}\n")
    logger.info(f"Wrote {classes_txt}: {len(class_images)} classes")

    return {
        "train": len(train_df),
        "val": len(val_df),
        "test": len(test_df),
    }
