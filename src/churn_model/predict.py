# ============================================
# File: src/churn_model/predict.py
# ============================================
# import json, import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from loguru import logger

# from .config import PROJECT_ROOT, AppConfig, load_config
from .utils import load_json, load_pipeline_joblib, setup_logging

# --- Global Cache ---
_config_cache: Optional[AppConfig] = None
_model_path_cache: Optional[Path] = None
_model_load_error: Optional[str] = None


def _initialize_prediction_service():
    """Loads config and determines default model path."""
    global _config_cache, _model_path_cache, _model_load_error
    if _config_cache is not None:
        return

    setup_logging()
    logger.info("Initializing prediction service...")
    try:
        _config_cache = load_config()
        training_cfg = _config_cache.training

        all_params_path = Path(training_cfg.all_best_params_input_path)
        if not all_params_path.exists():
            raise FileNotFoundError(
                f"Tuning results file not found at {all_params_path}"
            )

        all_params_data = load_json(all_params_path)
        overall_best_model_name = all_params_data.get("overall_best_model_name")

        if not overall_best_model_name:
            raise ValueError(
                "Overall best model name not found in tuning results file."
            )

        # Construct the path to the default best model
        default_model_path = Path(
            f"{training_cfg.final_model_output_base_path}_{overall_best_model_name}.joblib"
        )

        if not default_model_path.exists():
            _model_load_error = (
                f"Default best model file not found at path: {default_model_path}"
            )
            logger.critical(f"CRITICAL: {_model_load_error}")
            _model_path_cache = None  # Ensure path is None if file missing
        else:
            _model_path_cache = default_model_path  # Store the path
            logger.info(
                f"Prediction service initialized. Default model path: {_model_path_cache}"
            )
            _model_load_error = None

    except FileNotFoundError as e:
        _model_load_error = f"Config or parameters file not found: {e}"
        logger.critical(f"CRITICAL: {_model_load_error}")
        _config_cache = None
        _model_path_cache = None
    except ValueError as e:
        _model_load_error = f"Error determining default model: {e}"
        logger.critical(f"CRITICAL: {_model_load_error}")
        _config_cache = None
        _model_path_cache = None
    except Exception as e:
        _model_load_error = f"Failed to initialize prediction service: {e}"
        logger.critical(f"CRITICAL: {_model_load_error}", exc_info=True)
        _config_cache = None
        _model_path_cache = None


_initialize_prediction_service()


# --- Prediction Function ---
def make_prediction(
    *, input_data: Union[pd.DataFrame, List[Dict[str, Any]]]
) -> Dict[str, Optional[List[Any]]]:
    """Make predictions using the saved final pipeline."""
    global _config_cache, _model_path_cache, _model_load_error

    if _model_load_error:
        logger.error(
            f"Prediction unavailable due to initialization error: {_model_load_error}"
        )
        return {
            "predictions": None,
            "probabilities": None,
            "error": f"Service initialization failed: {_model_load_error}",
        }
    if _config_cache is None or _model_path_cache is None:
        logger.error(
            "Prediction service not initialized correctly (config or model path missing)."
        )
        return {
            "predictions": None,
            "probabilities": None,
            "error": "Prediction service not ready",
        }

    # --- Load Pipeline (Per Request) ---
    try:
        pipeline = load_pipeline_joblib(_model_path_cache)
    except FileNotFoundError:
        logger.error(f"Model file not found at {_model_path_cache} during request.")
        return {
            "predictions": None,
            "probabilities": None,
            "error": "Model file not found",
        }
    except Exception as e:
        logger.error(
            f"Failed to load prediction pipeline during request: {e}", exc_info=True
        )
        return {
            "predictions": None,
            "probabilities": None,
            "error": "Failed to load model",
        }

    # --- Process Input Data ---
    try:
        if isinstance(input_data, list):
            input_df = pd.DataFrame(input_data)
        elif isinstance(input_data, pd.DataFrame):
            input_df = input_data
        else:
            raise TypeError(f"Unsupported input data type: {type(input_data)}")
        expected_cols = set(_config_cache.data.initial_features)
        missing_cols = expected_cols - set(input_df.columns)
        if missing_cols:
            raise ValueError(f"Missing expected input columns: {missing_cols}")
    except (TypeError, ValueError) as e:
        logger.warning(f"Invalid input data format or content: {e}")
        return {
            "predictions": None,
            "probabilities": None,
            "error": f"Invalid input data: {e}",
        }
    except Exception as e:
        logger.error(f"Error processing input data: {e}", exc_info=True)
        return {
            "predictions": None,
            "probabilities": None,
            "error": "Error processing input data",
        }

    # --- Make Prediction ---
    logger.info(f"Making prediction on {len(input_df)} sample(s)...")
    try:
        model_name = "Unknown"
        if "classifier" in pipeline.named_steps:
            model_name = pipeline.named_steps["classifier"].__class__.__name__

        if model_name == "XGBoost":
            logger.debug("Sanitizing feature names for XGBoost prediction.")
            processed_data = pipeline.named_steps["data_processing"].transform(input_df)
            sanitized_data = sanitize_feature_names(processed_data.copy())
            predictions = pipeline.named_steps["classifier"].predict(sanitized_data)
            if hasattr(pipeline.named_steps["classifier"], "predict_proba"):
                probabilities = pipeline.named_steps["classifier"].predict_proba(
                    sanitized_data
                )[:, 1]
            else:
                probabilities = None
        else:
            # Original prediction for other models
            predictions = pipeline.predict(input_df)
            if hasattr(pipeline, "predict_proba"):
                probabilities = pipeline.predict_proba(input_df)[:, 1]
            else:
                probabilities = None

        results: Dict[str, Optional[List[Any]]] = {"predictions": predictions.tolist()}
        results["probabilities"] = (
            probabilities.tolist() if probabilities is not None else None
        )
        logger.info(f"Prediction completed successfully for {len(input_df)} sample(s).")
        results["error"] = None
        return results

    except Exception as e:
        logger.error(f"Error during pipeline prediction: {e}", exc_info=True)
        return {
            "predictions": None,
            "probabilities": None,
            "error": f"Prediction failed: {e}",
        }


if __name__ == "__main__":
    if _config_cache and not _model_load_error:
        logger.info("Testing make_prediction function...")
        test_input_single = [
            {
                "CreditScore": 650,
                "Geography": "France",
                "Gender": "Male",
                "Age": 35,
                "Tenure": 5,
                "Balance": 10000.0,
                "NumOfProducts": 1,
                "HasCrCard": 1,
                "IsActiveMember": 0,
                "EstimatedSalary": 50000.0,
            }
        ]
        test_input_bulk = [
            {
                "CreditScore": 700,
                "Geography": "Spain",
                "Gender": "Female",
                "Age": 42,
                "Tenure": 2,
                "Balance": 0.0,
                "NumOfProducts": 2,
                "HasCrCard": 1,
                "IsActiveMember": 1,
                "EstimatedSalary": 100000.0,
            },
            {
                "CreditScore": 500,
                "Geography": "Germany",
                "Gender": "Male",
                "Age": 55,
                "Tenure": 8,
                "Balance": 120000.0,
                "NumOfProducts": 1,
                "HasCrCard": 0,
                "IsActiveMember": 0,
                "EstimatedSalary": 80000.0,
            },
        ]
        test_input_invalid = [
            {"CreditScore": 650, "Geography": "Moon", "Gender": "Other"}
        ]
        logger.info("\n--- Testing Single Prediction ---")
        result_single = make_prediction(input_data=test_input_single)
        logger.info(f"Single prediction result: {result_single}")
        logger.info("\n--- Testing Bulk Prediction ---")
        result_bulk = make_prediction(input_data=test_input_bulk)
        logger.info(f"Bulk prediction result: {result_bulk}")
        logger.info("\n--- Testing Invalid Input (Missing Features) ---")
        result_invalid = make_prediction(input_data=test_input_invalid)
        logger.info(f"Invalid input result: {result_invalid}")
        logger.info("\n--- Testing Invalid Input Type ---")
        result_invalid_type = make_prediction(input_data="not a list or dataframe")
        logger.info(f"Invalid type result: {result_invalid_type}")
    else:
        logger.error(
            "Cannot run prediction test because the service failed to initialize."
        )
