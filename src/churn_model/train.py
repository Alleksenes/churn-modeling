# ============================================
# File: src/churn_model/train.py
# ============================================
import argparse  # json, re
import sys
import time

# import pandas as pd
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow
import mlflow.sklearn
import numpy as np
from lightgbm import LGBMClassifier
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from xgboost import XGBClassifier

from .config import PROJECT_ROOT, AppConfig, load_config
from .pipeline import SanitizeNamesTransformer, create_data_processing_pipeline
from .processing import load_processed_data
from .utils import load_json, save_pipeline_joblib, setup_logging

# --- Setup ---
setup_logging()


# --- Model Definitions ---
def get_base_models(config: AppConfig) -> Dict[str, Any]:
    """Returns dictionary of instantiated base models."""
    random_state = config.data.random_state
    return {
        "LogisticRegression": LogisticRegression(
            random_state=random_state,
            max_iter=2000,
            class_weight="balanced",
            solver="saga",
        ),
        "RandomForest": RandomForestClassifier(
            random_state=random_state, class_weight="balanced", n_jobs=-1
        ),
        "LightGBM": LGBMClassifier(
            random_state=random_state, class_weight="balanced", n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            random_state=random_state,
            use_label_encoder=False,
            eval_metric="logloss",
            n_jobs=-1,
        ),
        "SVC": SVC(
            random_state=random_state,
            probability=True,
            class_weight="balanced",
            cache_size=500,
        ),
    }


# --- Main Training Function ---
def run_training(config: AppConfig, model_to_train: Optional[str] = None):
    """
    Trains a specified model (or the overall best) using hyperparameters
    found during tuning.
    """
    logger.info("--- Starting Final Model Training Workflow ---")
    start_time = time.time()
    training_cfg = config.training
    data_cfg = config.data

    # --- Load All Best Hyperparameters ---
    all_params_path = Path(training_cfg.all_best_params_input_path)
    if not all_params_path.exists():
        logger.error(
            f"All best parameters file not found at {all_params_path}. Run tuning first."
        )
        raise FileNotFoundError(
            f"All best parameters file not found at {all_params_path}"
        )
    try:
        all_params_data = load_json(all_params_path)
        models_results = all_params_data.get("models", {})
        overall_best_model_name = all_params_data.get("overall_best_model_name")
    except Exception as e:
        logger.error(
            f"Failed to load or parse all best parameters file: {e}", exc_info=True
        )
        raise

    # --- Determine Which Model to Train ---
    target_model_name: Optional[str] = None
    target_hyperparameters: Optional[Dict[str, Any]] = None
    if model_to_train:
        if (
            model_to_train in models_results
            and models_results[model_to_train].get("status") == "success"
        ):
            target_model_name = model_to_train
            target_hyperparameters = models_results[model_to_train].get(
                "best_hyperparameters"
            )
            logger.info(f"Training specified model: {target_model_name}")
        else:
            logger.error(
                f"Specified model '{model_to_train}' not found or did not tune successfully in {all_params_path}."
            )
            raise ValueError(
                f"Cannot train specified model '{model_to_train}'. Check tuning results."
            )
    elif (
        overall_best_model_name
        and overall_best_model_name in models_results
        and models_results[overall_best_model_name].get("status") == "success"
    ):
        target_model_name = overall_best_model_name
        target_hyperparameters = models_results[target_model_name].get(
            "best_hyperparameters"
        )
        logger.info(
            f"Training overall best model identified during tuning: {target_model_name}"
        )
    else:
        logger.error("Could not determine model to train.")
        raise ValueError("No valid model specified or found to train.")
    if not target_hyperparameters:
        logger.error(
            f"Hyperparameters for target model '{target_model_name}' are missing."
        )
        raise ValueError(f"Missing hyperparameters for {target_model_name}")
    logger.info(f"Using hyperparameters (with prefix): {target_hyperparameters}")

    # --- Load Data ---
    try:
        X_train, X_test, y_train, y_test = load_processed_data(data_cfg)
        logger.info(
            f"Loaded processed data for final training. Train shape: {X_train.shape}, Test shape: {X_test.shape}"
        )
    except FileNotFoundError:
        raise
    except Exception as e:
        logger.error(f"Failed to load processed data: {e}", exc_info=True)
        raise

    # --- Create Pipeline ---
    base_models = get_base_models(config)
    if target_model_name not in base_models:
        raise ValueError(f"Unknown base model type: {target_model_name}")
    base_model = base_models[target_model_name].__class__(
        **base_models[target_model_name].get_params()
    )
    logger.debug(f"Instantiated base model: {base_model}")
    data_processing_pipeline = create_data_processing_pipeline(
        numerical_vars=data_cfg.numerical_vars,
        categorical_vars=data_cfg.categorical_vars,
    )

    # --- MODIFICATION: Conditionally add Sanitizer ---
    pipeline_steps = [("data_processing", data_processing_pipeline)]
    if target_model_name == "XGBoost":
        logger.info("Adding feature name sanitizer step for XGBoost.")
        pipeline_steps.append(
            ("sanitize_names", SanitizeNamesTransformer)
        )  # Add sanitizer step
    pipeline_steps.append(("classifier", base_model))  # Add classifier last

    final_pipeline = Pipeline(steps=pipeline_steps)
    logger.debug(
        f"Created final pipeline structure with steps: {[s[0] for s in final_pipeline.steps]}"
    )
    # --- END MODIFICATION ---

    # --- Prepare and Set Hyperparameters ---
    classifier_params: Dict[str, Any] = {}
    for key, value in target_hyperparameters.items():
        if key.startswith("classifier__"):
            classifier_params[key.replace("classifier__", "")] = value
    if target_model_name == "XGBoost" and "scale_pos_weight" not in classifier_params:
        try:
            counts = np.bincount(y_train)
            scale_pos_weight = (
                float(counts[0]) / counts[1]
                if len(counts) > 1 and counts[1] > 0
                else 1.0
            )
            classifier_params["scale_pos_weight"] = scale_pos_weight
            logger.info(
                f"Dynamically added scale_pos_weight={scale_pos_weight:.2f} for final XGBoost model."
            )
        except Exception:
            logger.warning(
                "Could not calculate scale_pos_weight for final XGBoost model."
            )
    if target_model_name == "LogisticRegression":
        current_penalty = classifier_params.get("penalty")
        if current_penalty != "elasticnet":
            classifier_params.pop("l1_ratio", None)
            logger.debug("Removed l1_ratio for non-elasticnet Logistic Regression.")
    logger.debug(
        f"Parameters prepared for direct setting on classifier: {classifier_params}"
    )
    try:
        final_pipeline.named_steps["classifier"].set_params(**classifier_params)
        logger.info("Successfully set hyperparameters directly on the classifier step.")
    except ValueError as e:
        logger.error(
            f"Error setting final hyperparameters directly on classifier: {classifier_params}. Error: {e}",
            exc_info=True,
        )
        raise
    except Exception as e:
        logger.error(
            f"Unexpected error during direct set_params on classifier: {e}",
            exc_info=True,
        )
        raise

    # --- MLflow Logging ---
    mlflow.set_experiment(training_cfg.mlflow_experiment_name)
    with mlflow.start_run(run_name=f"Final_Training_{target_model_name}") as run:
        run_id = run.info.run_id
        logger.info(f"MLflow Run ID for final training: {run_id}")
        mlflow.log_param("trained_model_type", target_model_name)
        mlflow.log_params(classifier_params)
        mlflow.log_artifact(str(all_params_path))

        # --- Fit Model ---
        logger.info("Fitting final pipeline on full training data...")
        fit_start_time = time.time()
        try:
            final_pipeline.fit(X_train, y_train)

            fit_duration = time.time() - fit_start_time
            logger.success(
                f"Final pipeline fitting complete. Duration: {fit_duration:.2f}s"
            )
            mlflow.log_metric("training_duration_seconds", fit_duration)
            mlflow.set_tag("training_status", "success")
        except Exception as e:
            fit_duration = time.time() - fit_start_time
            logger.error(
                f"Failed to fit final pipeline after {fit_duration:.2f}s: {e}",
                exc_info=True,
            )
            mlflow.log_metric("training_duration_seconds", fit_duration)
            mlflow.set_tag("training_status", "failed")
            raise

        # --- Quick Evaluation on Test Set ---
        logger.info("Performing quick evaluation on test set...")
        eval_metrics = {}
        try:
            y_pred = final_pipeline.predict(X_test)
            if hasattr(final_pipeline, "predict_proba"):
                y_proba = final_pipeline.predict_proba(X_test)[:, 1]
                eval_metrics["test_roc_auc"] = roc_auc_score(y_test, y_proba)
            else:
                eval_metrics["test_roc_auc"] = np.nan
            eval_metrics["test_f1_score"] = f1_score(y_test, y_pred)
            eval_metrics["test_balanced_accuracy"] = balanced_accuracy_score(
                y_test, y_pred
            )
            mlflow.log_metrics(eval_metrics)
            logger.info(
                f"Quick Test ROC AUC: {eval_metrics.get('test_roc_auc', 'N/A'):.4f}"
            )
            logger.info(f"Quick Test F1 Score: {eval_metrics['test_f1_score']:.4f}")
            logger.info(
                f"Quick Test Balanced Accuracy: {eval_metrics['test_balanced_accuracy']:.4f}"
            )
            mlflow.set_tag("quick_evaluation_status", "success")
        except Exception as e:
            logger.warning(f"Quick evaluation failed: {e}", exc_info=True)
            mlflow.set_tag("quick_evaluation_status", "failed")

        # --- Save Final Model ---
        final_model_path = Path(
            f"{training_cfg.final_model_output_base_path}_{target_model_name}.joblib"
        )
        logger.info(f"Attempting to save final model to {final_model_path}...")
        try:
            save_pipeline_joblib(final_pipeline, final_model_path)
            logger.success(f"Final trained pipeline saved successfully.")
            mlflow.log_artifact(
                str(final_model_path), artifact_path="final_model_joblib"
            )
            try:
                from mlflow.models.signature import infer_signature

                # --- FIX for MLflow Signature ---
                try:
                    signature = infer_signature(
                        X_train.head(), final_pipeline.predict(X_train.head())
                    )
                except Exception as sig_inf_e:
                    logger.warning(
                        f"Could not infer signature: {sig_inf_e}. Proceeding without signature."
                    )
                    signature = None
                # --- End FIX ---
                mlflow.sklearn.log_model(
                    sk_model=final_pipeline,
                    artifact_path=f"final_model_sklearn_{target_model_name}",
                    signature=signature,
                    input_example=X_train.head(5).to_dict(orient="records"),
                )
                logger.info("Final model logged to MLflow using mlflow.sklearn.")
            except ImportError:
                mlflow.sklearn.log_model(
                    sk_model=final_pipeline,
                    artifact_path=f"final_model_sklearn_{target_model_name}",
                )
                logger.info(
                    "Final model logged to MLflow using mlflow.sklearn (signature requires newer MLflow or failed)."
                )
            except Exception as sig_log_e:
                logger.warning(
                    f"Failed to log model with signature/example: {sig_log_e}. Logging basic model."
                )
                mlflow.sklearn.log_model(
                    sk_model=final_pipeline,
                    artifact_path=f"final_model_sklearn_{target_model_name}",
                )
            mlflow.set_tag("saving_status", "success")
        except Exception as e:
            logger.error(f"Failed to save or log final model: {e}", exc_info=True)
            mlflow.set_tag("saving_status", "failed")
            raise

    total_duration = time.time() - start_time
    logger.info(
        f"--- Final Model Training Workflow Finished for {target_model_name}. Total Duration: {total_duration:.2f}s ---"
    )
    return final_model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the final churn prediction model."
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to the configuration file relative to project root.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Specific model name to train (e.g., RandomForest). If None, trains the overall best from tuning.",
    )
    args = parser.parse_args()
    try:
        from .config import PROJECT_ROOT, load_config
        from .utils import setup_logging

        setup_logging()
    except ImportError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    try:
        config_path = PROJECT_ROOT / args.config
        app_config = load_config(config_path)
        run_training(app_config, model_to_train=args.model_name)
    except FileNotFoundError:
        logger.critical("Config/params file not found.")
        sys.exit(1)
    except ValueError as e:
        logger.critical(f"Value error during training setup: {e}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Training script failed: {e}", exc_info=True)
        sys.exit(1)
