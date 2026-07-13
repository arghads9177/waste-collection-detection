"""Standard logging setup. Call setup_logging() once at every entry point."""
import logging
import logging.handlers
from pathlib import Path

FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def setup_logging(level: str = "INFO", log_file: str | None = "logs/app.log") -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(
            logging.handlers.RotatingFileHandler(
                log_file, maxBytes=10_000_000, backupCount=5
            )
        )
    logging.basicConfig(level=level.upper(), format=FORMAT, handlers=handlers)
    # Quiet down chatty third-party loggers
    for noisy in ("urllib3", "PIL", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
