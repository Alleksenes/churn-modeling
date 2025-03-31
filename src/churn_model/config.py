# ============================================
# File: src/churn_model/config.py
# ============================================
import sys
import traceback
from pathlib import Path
from typing import List  # , Any, Dict, Optional

import yaml
from loguru import logger
from pydantic import (  # DirectoryPath,; FilePath,
    BaseModel,
    Field,
    ValidationError,
    validator,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_FILE_PATH = PROJECT_ROOT / "config.yaml"

# --- Pydantic Models for Validation ---


class DataConfig(BaseModel):
    """Configuration for data paths and features."""

    raw_path: str
    processed_train_path: str
    processed_test_path: str
    target_column: str
    test_size: float = Field(..., gt=0, lt=1)
    random_state: int
    initial_features: List[str]
    numerical_vars: List[str]
    categorical_vars: List[str]

    @validator("raw_path", "processed_train_path", "processed_test_path", pre=True)
    def make_path_absolute(cls, v):
        """Make data paths absolute."""
        path = Path(v)
        return str(path) if path.is_absolute() else str(PROJECT_ROOT / v)

    # Make cross-field validators safer
    @validator("categorical_vars")
    def check_features_overlap(cls, cat_vars, values):
        """Check for overlap between numerical and categorical features."""
        if "numerical_vars" in values and values.get("numerical_vars"):
            num_set = set(values["numerical_vars"])
            cat_set = set(cat_vars)
            overlap = num_set.intersection(cat_set)
            if overlap:
                raise ValueError(
                    f"Features cannot be both numerical and categorical: {overlap}"
                )
        return cat_vars

    @validator("numerical_vars", "categorical_vars", each_item=False)
    def check_features_in_initial(cls, vars_list, values):
        """Check if features are in the initial feature list."""
        if "initial_features" in values and values.get("initial_features"):
            initial_set = set(values["initial_features"])
            if not set(vars_list).issubset(initial_set):
                missing = set(vars_list) - initial_set
                raise ValueError(
                    f"Features {missing} not in initial_features list in config"
                )
        return vars_list


# --- Other Config Models (TuningConfig, TrainingConfig, etc.) ---
class TuningConfig(BaseModel):
    """Configuration for hyperparameter tuning."""

    models_to_tune: List[str]
    cv_folds: int = Field(..., gt=1)
    optuna_trials_per_model: int = Field(..., gt=0)
    optimization_metric: str
    mlflow_experiment_name: str
    all_best_params_output_path: str

    @validator("all_best_params_output_path", pre=True)
    def make_path_absolute(cls, v):
        """Make the output path absolute."""
        path = Path(v)
        return str(path) if path.is_absolute() else str(PROJECT_ROOT / v)


class TrainingConfig(BaseModel):
    """Configuration for model training."""

    all_best_params_input_path: str
    final_model_output_base_path: str
    mlflow_experiment_name: str

    @validator("all_best_params_input_path", "final_model_output_base_path", pre=True)
    def make_path_absolute(cls, v):
        """Make the input/output paths absolute."""
        path = Path(v)
        return str(path) if path.is_absolute() else str(PROJECT_ROOT / v)


class EvaluationConfig(BaseModel):
    """Configuration for model evaluation."""

    plots_output_dir: str
    shap_plots_output_dir: str
    mlflow_experiment_name: str
    shap_kernel_background_samples: int = Field(..., gt=0)

    @validator("plots_output_dir", "shap_plots_output_dir", pre=True)
    def make_path_absolute(cls, v):
        """Make the plot output paths absolute."""
        path = Path(v)
        return str(path) if path.is_absolute() else str(PROJECT_ROOT / v)


class APIConfig(BaseModel):
    """Configuration for the API."""

    title: str
    version: str


class AppConfig(BaseModel):
    """Overall application configuration."""

    project_name: str
    version: str
    data: DataConfig
    tuning: TuningConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    api: APIConfig


# --- Loading Function (Add more detailed error logging) ---


def load_config(config_path: Path = CONFIG_FILE_PATH) -> AppConfig:
    """Loads and validates config from YAML."""
    if not config_path.exists():
        print(f"CRITICAL: Config file not found at {config_path}", file=sys.stderr)
        raise FileNotFoundError(f"Config file not found at {config_path}")
    try:
        print(f"INFO: Attempting to load config from {config_path}")
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        if config_dict is None:
            raise ValueError("Config file is empty.")
        print("INFO: YAML loaded successfully. Validating with Pydantic...")
        validated_config = AppConfig(**config_dict)
        try:
            logger.info(f"Configuration loaded and validated from {config_path}")
        except NameError:
            print(f"INFO: Configuration loaded and validated from {config_path}")
        return validated_config
    except yaml.YAMLError as e:
        print(
            f"CRITICAL: Error parsing YAML config file {config_path}: {e}",
            file=sys.stderr,
        )
        raise
    except ValidationError as e:
        print(
            f"CRITICAL: Pydantic validation failed for config file {config_path}:",
            file=sys.stderr,
        )
        for error in e.errors():
            print(
                f"  Field: {'.'.join(map(str, error['loc']))}, Error: {error['msg']}",
                file=sys.stderr,
            )
        raise
    except Exception as e:
        print(
            f"CRITICAL: Error loading or validating config file {config_path}: {type(e).__name__} - {e}",
            file=sys.stderr,
        )
        traceback.print_exc()
        raise
