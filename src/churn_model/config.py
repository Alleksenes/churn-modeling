# ============================================
# File: src/churn_model/config.py
# ============================================
import yaml
from pydantic import (
    BaseModel,
    Field,
    root_validator,
    ValidationError,
    field_validator,
)
from typing import List, Optional, Dict, Any
from pathlib import Path
from loguru import logger
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_FILE_PATH = PROJECT_ROOT / "config.yaml"


class DataConfig(BaseModel):
    raw_path: str
    processed_train_path: str
    processed_test_path: str
    target_column: str
    test_size: float = Field(..., gt=0, lt=1)
    random_state: int
    initial_features: List[str]
    numerical_vars: List[str]
    categorical_vars: List[str]

    @field_validator("raw_path", "processed_train_path", "processed_test_path", mode="before")
    @classmethod  # Decorate as class method
    def make_path_absolute(cls, v: Any) -> str:
        """Ensures paths are absolute, resolving relative to project root."""
        if not isinstance(v, (str, Path)):
            return v
        path = Path(v)
        if path.is_absolute():
            return str(path)
        absolute_path = (PROJECT_ROOT / path).resolve()
        logger.trace(f"Resolved relative path '{v}' to '{absolute_path}'")
        return str(absolute_path)

    @root_validator(skip_on_failure=True)
    @classmethod
    def check_feature_consistency(cls, values):
        """Checks overlap and ensures features are in initial list."""
        num_vars = values.get("numerical_vars")
        cat_vars = values.get("categorical_vars")
        initial_features = values.get("initial_features")

        if num_vars and cat_vars:
            num_set = set(num_vars)
            cat_set = set(cat_vars)
            overlap = num_set.intersection(cat_set)
            if overlap:
                raise ValueError(f"Features cannot be both numerical and categorical: {overlap}")

        if initial_features:
            initial_set = set(initial_features)
            if num_vars and not set(num_vars).issubset(initial_set):
                missing = set(num_vars) - initial_set
                raise ValueError(f"Numerical features {missing} not in initial_features list")
            if cat_vars and not set(cat_vars).issubset(initial_set):
                missing = set(cat_vars) - initial_set
                raise ValueError(f"Categorical features {missing} not in initial_features list")
        return values


class TuningConfig(BaseModel):
    models_to_tune: List[str]
    cv_folds: int = Field(..., gt=1)
    optuna_trials_per_model: int = Field(..., gt=0)
    optimization_metric: str
    mlflow_experiment_name: str
    all_best_params_output_path: str

    @field_validator("all_best_params_output_path", mode="before")
    @classmethod
    def make_path_absolute(cls, v: Any) -> str:
        """Ensures paths are absolute, resolving relative to project root."""
        if not isinstance(v, (str, Path)):
            return v
        path = Path(v)
        return str(path) if path.is_absolute() else str((PROJECT_ROOT / path).resolve())

class TrainingConfig(BaseModel):
    all_best_params_input_path: str
    final_model_output_base_path: str
    mlflow_experiment_name: str

    @field_validator("all_best_params_input_path", "final_model_output_base_path", mode="before")
    @classmethod
    def make_path_absolute(cls, v: Any) -> str:
        """Ensures paths are absolute, resolving relative to project root."""
        if not isinstance(v, (str, Path)):
            return v
        path = Path(v)
        return str(path) if path.is_absolute() else str((PROJECT_ROOT / path).resolve())


class EvaluationConfig(BaseModel):
    plots_output_dir: str
    shap_plots_output_dir: str
    mlflow_experiment_name: str
    shap_kernel_background_samples: int = Field(..., gt=0)

    @field_validator("plots_output_dir", "shap_plots_output_dir", mode="before")
    @classmethod
    def make_path_absolute(cls, v: Any) -> str:
        """Ensures paths are absolute, resolving relative to project root."""
        if not isinstance(v, (str, Path)):
            return v
        path = Path(v)
        return str(path) if path.is_absolute() else str((PROJECT_ROOT / path).resolve())


class APIConfig(BaseModel):
    title: str
    version: str


class AppConfig(BaseModel):
    project_name: str
    version: str
    data: DataConfig
    tuning: TuningConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    api: APIConfig


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
        print(f"INFO: YAML loaded successfully. Validating with Pydantic...")
        validated_config = AppConfig(**config_dict)
        try:
            logger.info(f"Configuration loaded and validated from {config_path}")
        except NameError:
            print(f"INFO: Configuration loaded and validated from {config_path}")
        return validated_config
    except yaml.YAMLError as e:
        print(f"CRITICAL: Error parsing YAML config file {config_path}: {e}", file=sys.stderr)
        raise
    except ValidationError as e:
        print(f"CRITICAL: Pydantic validation failed for config file {config_path}:", file=sys.stderr)
        for error in e.errors():
            print(f"  Field: {'.'.join(map(str, error['loc']))}, Error: {error['msg']}", file=sys.stderr)
        raise
    except Exception as e:
        print(
            f"CRITICAL: Error loading or validating config file {config_path}: {type(e).__name__} - {e}",
            file=sys.stderr,
        )
        raise