# ============================================
# File: src/churn_model/utils.py
# ============================================
import json
import sys
from pathlib import Path
from typing import Any, Dict

import joblib
from loguru import logger

from .config import PROJECT_ROOT

LOGS_DIR = PROJECT_ROOT / "logs"
CONFIG_DIR = PROJECT_ROOT / "config"

DEFAULT_LOGGING_CONFIG = {
    "handlers": [
        {
            "sink": sys.stderr,
            "level": "INFO",
            "format": "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        },
        {
            "sink": LOGS_DIR / "churn_model_{time}.log",
            "rotation": "10 MB",
            "retention": "10 days",
            "level": "DEBUG",
            "enqueue": True,
            "backtrace": True,
            "diagnose": True,
            "format": "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        },
    ]
}

_logging_configured = False


def setup_logging(config_path: Path = CONFIG_DIR / "logging_config.json"):
    """Configures Loguru logger. Ensures it only runs once per process."""
    global _logging_configured
    if _logging_configured:
        logger.trace("Logging already configured.")
        return

    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(
            f"ERROR: Could not create logs directory {LOGS_DIR}: {e}", file=sys.stderr
        )
        try:
            logger.remove()
            logger.add(sys.stderr, level="INFO")
            logger.error("Failed to create log directory, using stderr only.")
            _logging_configured = True
        except Exception:
            print("CRITICAL: Failed even to configure stderr logging.", file=sys.stderr)
        return

    logger.remove()
    config_data = DEFAULT_LOGGING_CONFIG
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                config_data = json.load(f)
            print(f"INFO: Loaded logging configuration from {config_path}")
        except Exception as e:
            print(
                f"WARNING: Failed to load logging config from {config_path}, using defaults. Error: {e}",
                file=sys.stderr,
            )
            config_data = DEFAULT_LOGGING_CONFIG
    else:
        print(
            f"INFO: Logging config file not found at {config_path}, using default configuration."
        )

    try:
        logger.configure(**config_data)
        logger.info("Logging setup complete.")
        _logging_configured = True
    except Exception as e:
        logger.add(sys.stderr, level="INFO")
        logger.error(
            f"Failed to configure logging with provided settings: {e}. Using basic stderr logging."
        )
        _logging_configured = True


# --- Generic Save/Load Helpers ---


def save_json(data: Dict, file_path: Path):
    """Saves dictionary data to JSON file."""
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)
        logger.info(f"Successfully saved JSON to {file_path}")
    except TypeError as e:
        logger.error(
            f"Data type error saving JSON to {file_path}: {e}. Data: {str(data)[:200]}..."
        )
        raise
    except Exception as e:
        logger.error(f"Error saving JSON to {file_path}: {e}", exc_info=True)
        raise


def load_json(file_path: Path) -> Dict:
    """Loads dictionary data from JSON file."""
    if not file_path.exists():
        logger.error(f"JSON file not found at {file_path}")
        raise FileNotFoundError(f"JSON file not found at {file_path}")
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        logger.info(f"Successfully loaded JSON from {file_path}")
        return data
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {file_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading JSON from {file_path}: {e}", exc_info=True)
        raise


def save_pipeline_joblib(pipeline: Any, file_path: Path):
    """Saves any object (like a pipeline) using joblib."""
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, file_path)
        logger.info(f"Object saved successfully using joblib to {file_path}")
    except Exception as e:
        logger.error(
            f"Error saving object with joblib to {file_path}: {e}", exc_info=True
        )
        raise


def load_pipeline_joblib(file_path: Path) -> Any:
    """Loads an object (like a pipeline) using joblib."""
    if not file_path.exists():
        logger.error(f"Joblib file not found at {file_path}")
        raise FileNotFoundError(f"Joblib file not found at {file_path}")
    try:
        pipeline = joblib.load(file_path)
        logger.info(f"Object loaded successfully using joblib from {file_path}")
        return pipeline
    except Exception as e:
        logger.error(
            f"Error loading object with joblib from {file_path}: {e}", exc_info=True
        )
        raise
