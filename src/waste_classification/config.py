"""Typed configuration loaded from .env via pydantic-settings.

Usage:
    from waste_classification.config import settings
    print(settings.train_epochs)
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Data
    data_raw_dir: str = "data/raw/garbage_classification"
    data_processed_dir: str = "data/processed"
    data_split_seed: int = 42

    # Model
    model_backbone: str = "efficientnet_b0"
    model_weights: str = "models/exported/waste-classifier-v1.pt"
    model_device: str = "auto"

    # Training
    train_epochs: int = 50
    train_batch_size: int = 32
    train_lr: float = 1e-3
    train_warmup_epochs: int = 5
    train_early_stopping_patience: int = 10
    train_seed: int = 42

    # Deployment / API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    streamlit_api_url: str = "http://localhost:8000"
    prediction_log_db: str = "outputs/predictions.db"
    output_dir: str = "outputs"

    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/app.log"


settings = Settings()
