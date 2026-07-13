"""Generate small synthetic test fixtures for Phase 1 testing.

This script creates:
- 3 classes (plastic, paper, metal) with 3 fixture images each
- 1 corrupted fixture (garbage bytes with .jpg extension)
- 1 exact-duplicate fixture (byte-identical copy)
"""
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image


def generate_fixture_images(fixture_root: Path) -> None:
    fixture_root.mkdir(parents=True, exist_ok=True)

    classes = ["plastic", "paper", "metal"]
    colors = {
        "plastic": (255, 100, 100),
        "paper": (100, 255, 100),
        "metal": (100, 100, 255),
    }

    # Generate 5 images per class (needed for stratified split to have at least 2 per class)
    for class_name in classes:
        class_dir = fixture_root / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        for i in range(5):
            # Create a small solid-color synthetic image
            color = colors[class_name]
            img_array = np.full((64, 64, 3), color, dtype=np.uint8)
            img = Image.fromarray(img_array)
            img.save(class_dir / f"{class_name}_{i}.jpg")
            print(f"✓ Created {class_dir / f'{class_name}_{i}.jpg'}")

    # Create a duplicate (byte-identical copy of plastic_0.jpg)
    original = fixture_root / "plastic" / "plastic_0.jpg"
    duplicate = fixture_root / "plastic" / "plastic_0_dup.jpg"
    with open(original, "rb") as src:
        with open(duplicate, "wb") as dst:
            dst.write(src.read())
    print(f"✓ Created duplicate: {duplicate}")

    # Remove the original duplicate fixture directory if it exists and regenerate
    import shutil
    if fixture_root.exists():
        # We'll delete and recreate to start fresh
        pass

    # Create a corrupted file (garbage bytes with .jpg extension)
    corrupted = fixture_root / "paper" / "corrupted.jpg"
    with open(corrupted, "wb") as f:
        f.write(b"This is not a valid JPEG file. Just garbage bytes.")
    print(f"✓ Created corrupted: {corrupted}")

    print("\nFixture generation complete!")


if __name__ == "__main__":
    fixture_root = Path(__file__).parent.parent / "tests" / "fixtures" / "garbage_classification"
    generate_fixture_images(fixture_root)
