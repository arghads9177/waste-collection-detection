"""Integration test: ONNX export + verify_onnx_output() round-trip on a fixture image."""
from pathlib import Path

import torch
from PIL import Image

from waste_classification.data.dataset import build_eval_transforms
from waste_classification.inference.export import export_to_onnx, verify_onnx_output
from waste_classification.models.factory import build_model, load_checkpoint, save_checkpoint


def test_export_and_verify_onnx_round_trip(tmp_path: Path) -> None:
    class_names = ["metal", "paper"]
    image_size = 64

    model = build_model("efficientnet_b0", num_classes=len(class_names), pretrained=False)
    checkpoint_path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        backbone="efficientnet_b0",
        class_names=class_names,
        hyperparameters={"lr": 1e-3},
        val_metrics={"accuracy": 0.5},
    )

    loaded_model, checkpoint = load_checkpoint(checkpoint_path)
    assert checkpoint["backbone"] == "efficientnet_b0"

    onnx_path = tmp_path / "model.onnx"
    export_to_onnx(loaded_model, onnx_path, image_size=image_size)
    assert onnx_path.exists()

    fixture_image = (
        Path(__file__).parent.parent
        / "fixtures"
        / "garbage_classification"
        / "paper"
        / "paper_0.jpg"
    )
    img = Image.open(fixture_image).convert("RGB")
    sample_input = build_eval_transforms(image_size)(img).unsqueeze(0)

    max_abs_diff = verify_onnx_output(loaded_model, onnx_path, sample_input)
    assert max_abs_diff < 1e-3

    # Dynamic batch axis: a batch of 2 must also run through the exported graph.
    batched_input = torch.cat([sample_input, sample_input], dim=0)
    verify_onnx_output(loaded_model, onnx_path, batched_input)
