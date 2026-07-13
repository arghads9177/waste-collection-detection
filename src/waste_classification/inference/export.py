"""Export checkpoints to ONNX and verify parity against the PyTorch model."""
import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

ONNX_OPSET = 17


def export_to_onnx(
    model: nn.Module,
    output_path: Path | str,
    image_size: int,
    opset: int = ONNX_OPSET,
) -> Path:
    """Export a PyTorch classification model to ONNX with a dynamic batch axis.

    Args:
        model: Model to export (will be run in eval mode).
        output_path: Destination .onnx path.
        image_size: Square input resolution (e.g. 224).
        opset: ONNX opset version.

    Returns:
        The output path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    dummy_input = torch.randn(1, 3, image_size, image_size)

    torch.onnx.export(
        model,
        (dummy_input,),
        str(output_path),
        opset_version=opset,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        dynamo=False,
    )
    logger.info(f"Exported ONNX model to {output_path} (opset={opset})")
    return output_path


def verify_onnx_output(
    model: nn.Module,
    onnx_path: Path | str,
    sample_input: torch.Tensor,
    atol: float = 1e-3,
) -> float:
    """Compare PyTorch vs ONNX Runtime outputs on the same input batch.

    Args:
        model: The source PyTorch model (eval mode).
        onnx_path: Path to the exported ONNX model.
        sample_input: A batch tensor of shape (N, 3, H, W).
        atol: Maximum acceptable absolute difference.

    Returns:
        The observed max-abs-diff between PyTorch and ONNX outputs.

    Raises:
        AssertionError: If the max-abs-diff exceeds `atol`.
    """
    model.eval()
    with torch.no_grad():
        torch_output = model(sample_input).numpy()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_output = session.run(None, {"input": sample_input.numpy()})[0]

    max_abs_diff = float(np.max(np.abs(torch_output - onnx_output)))
    logger.info(f"PyTorch vs ONNX max-abs-diff: {max_abs_diff:.6g}")

    assert max_abs_diff < atol, (
        f"ONNX output diverges from PyTorch: max-abs-diff={max_abs_diff:.6g} >= atol={atol}"
    )
    return max_abs_diff
