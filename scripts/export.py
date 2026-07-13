"""Export a trained checkpoint to ONNX and verify it against the PyTorch model.

Reads the backbone/num_classes from the checkpoint dict itself (self-describing),
so it works regardless of .env MODEL_BACKBONE.

Usage:
    uv run python scripts/export.py [--checkpoint ...] [--onnx-output ...]
"""
import argparse
import logging
from pathlib import Path

import torch

from waste_classification.config import settings
from waste_classification.inference.export import export_to_onnx, verify_onnx_output
from waste_classification.logging_config import setup_logging
from waste_classification.models.factory import load_checkpoint

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Export a checkpoint to ONNX")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(settings.model_weights),
        help=f"Path to the .pt checkpoint (default: {settings.model_weights})",
    )
    parser.add_argument(
        "--onnx-output",
        type=Path,
        default=None,
        help="Destination .onnx path (default: models/onnx/<checkpoint stem>.onnx)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=settings.model_image_size,
        help=f"Square input resolution (default: {settings.model_image_size})",
    )
    args = parser.parse_args()

    setup_logging(settings.log_level, settings.log_file)

    if not args.checkpoint.exists():
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        return 1

    onnx_output = args.onnx_output or Path("models/onnx") / f"{args.checkpoint.stem}.onnx"

    logger.info(f"Loading checkpoint from {args.checkpoint}")
    model, checkpoint = load_checkpoint(args.checkpoint)

    logger.info(f"Exporting to ONNX at {onnx_output}")
    export_to_onnx(model, onnx_output, image_size=args.image_size)

    logger.info("Verifying PyTorch vs ONNX output parity")
    sample_input = torch.randn(1, 3, args.image_size, args.image_size)
    max_abs_diff = verify_onnx_output(model, onnx_output, sample_input)

    logger.info(f"✓ Export verified (backbone={checkpoint['backbone']}, max-abs-diff={max_abs_diff:.6g})")
    return 0


if __name__ == "__main__":
    exit(main())
