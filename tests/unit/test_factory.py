"""Unit tests for model factory: build_model output shape and backbone freezing."""
import pytest
import torch

from waste_classification.models.factory import build_model, set_backbone_frozen


@pytest.mark.parametrize("backbone", ["efficientnet_b0", "convnext_tiny"])
def test_build_model_output_shape(backbone):
    model = build_model(backbone, num_classes=5, pretrained=False)
    model.eval()

    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (2, 5)


@pytest.mark.parametrize("backbone", ["efficientnet_b0", "convnext_tiny"])
def test_set_backbone_frozen_freezes_all_but_head(backbone):
    model = build_model(backbone, num_classes=3, pretrained=False)

    set_backbone_frozen(model, frozen=True)
    head_param_ids = {id(p) for p in model.classifier.parameters()}
    for p in model.parameters():
        expected = True if id(p) in head_param_ids else False
        assert p.requires_grad == expected

    set_backbone_frozen(model, frozen=False)
    assert all(p.requires_grad for p in model.parameters())


def test_build_model_unknown_backbone_raises():
    with pytest.raises(ValueError, match="Unknown backbone"):
        build_model("resnet50", num_classes=3, pretrained=False)
