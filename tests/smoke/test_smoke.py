"""Smoke test: the pipeline stands up.

Replace the stub with a real check once a model exists: load weights from
settings.model_weights, run one fixture image through the full inference path,
and assert output shape/type/range sanity (not exact values).
"""
from waste_classification.config import settings


def test_config_loads():
    assert settings.train_epochs > 0
    assert settings.train_batch_size > 0
    assert 0.0 < settings.train_lr <= 1.0


def test_inference_pipeline_stub():
    # TODO: replace with real single-image inference once a model is exported.
    assert True
