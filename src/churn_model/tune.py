# ============================================
# File: src/churn_model/tune.py
# ============================================
import argparse  # json, re
import sys
import time
from pathlib import Path

import mlflow
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import get_scorer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from xgboost import XGBClassifier

from .config import PROJECT_ROOT, AppConfig, load_config
from .pipeline import create_data_processing_pipeline, _sanitize_column_names
from .processing import load_processed_data, run_preprocess_workflow
from .utils import save_json, setup_logging

# --- Setup ---
setup_logging()


# --- Model Definitions ---
def get_base_models(config: AppConfig) -> dict:
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


# --- Optuna Hyperparameter Spaces ---
def get_optuna_params(
    trial: optuna.Trial, model_name: str, y_train_for_imbalance: np.ndarray = None
):
    """Defines the hyperparameter search space for Optuna."""
    logger.trace(f"Getting Optuna params for trial {trial.number}, model: {model_name}")

    if model_name == "LogisticRegression":
        penalty = trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet"])
        c_val = trial.suggest_float("C", 1e-3, 1e2, log=True)
        params = {
            "classifier__C": c_val,
            "classifier__solver": "saga",
            "classifier__penalty": penalty,
        }
        if penalty == "elasticnet":
            params["classifier__l1_ratio"] = trial.suggest_float("l1_ratio", 0.05, 0.95)
        return params

    # --- RandomForest, LightGBM, SVC remain the same as previous version ---
    elif model_name == "RandomForest":
        return {
            "classifier__n_estimators": trial.suggest_int(
                "n_estimators", 50, 500, step=50
            ),
            "classifier__max_depth": trial.suggest_int("max_depth", 5, 30),
            "classifier__min_samples_split": trial.suggest_int(
                "min_samples_split", 2, 20
            ),
            "classifier__min_samples_leaf": trial.suggest_int(
                "min_samples_leaf", 1, 20
            ),
            "classifier__max_features": trial.suggest_categorical(
                "max_features", ["sqrt", "log2", 0.5, 0.7, None]
            ),
        }
    elif model_name == "LightGBM":
        return {
            "classifier__n_estimators": trial.suggest_int(
                "n_estimators", 100, 1000, step=100
            ),
            "classifier__learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.2, log=True
            ),
            "classifier__num_leaves": trial.suggest_int("num_leaves", 10, 150),
            "classifier__max_depth": trial.suggest_int("max_depth", 3, 15),
            "classifier__reg_alpha": trial.suggest_float(
                "reg_alpha", 1e-8, 10.0, log=True
            ),
            "classifier__reg_lambda": trial.suggest_float(
                "reg_lambda", 1e-8, 10.0, log=True
            ),
            "classifier__colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.5, 1.0
            ),
            "classifier__subsample": trial.suggest_float("subsample", 0.5, 1.0),
        }
    elif model_name == "XGBoost":
        scale_pos_weight = 1.0
        if y_train_for_imbalance is not None and len(y_train_for_imbalance) > 0:
            try:
                counts = np.bincount(y_train_for_imbalance)
                scale_pos_weight = (
                    float(counts[0]) / counts[1]
                    if len(counts) > 1 and counts[1] > 0
                    else 1.0
                )
            except Exception:
                scale_pos_weight = 1.0
        return {
            "classifier__n_estimators": trial.suggest_int(
                "n_estimators", 100, 1000, step=100
            ),
            "classifier__learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.2, log=True
            ),
            "classifier__max_depth": trial.suggest_int("max_depth", 3, 15),
            "classifier__subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "classifier__colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.5, 1.0
            ),
            "classifier__gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
            "classifier__reg_alpha": trial.suggest_float(
                "reg_alpha", 1e-8, 10.0, log=True
            ),
            "classifier__reg_lambda": trial.suggest_float(
                "reg_lambda", 1e-8, 10.0, log=True
            ),
            "classifier__scale_pos_weight": scale_pos_weight,
        }
    elif model_name == "SVC":
        kernel = trial.suggest_categorical("kernel", ["rbf", "poly", "sigmoid"])
        params = {
            "classifier__C": trial.suggest_float("C", 1e-2, 1e3, log=True),
            "classifier__kernel": kernel,
            "classifier__gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
        }
        if kernel == "poly":
            params["classifier__degree"] = trial.suggest_int("degree", 2, 4)
        if kernel in ["poly", "sigmoid"]:
            params["classifier__coef0"] = trial.suggest_float("coef0", 0.0, 1.0)
        return params
    else:
        logger.warning(f"Model {model_name} not found in Optuna parameter definitions.")
        return {}


# --- Optuna Objective Function ---
def objective(
    trial: optuna.Trial,
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: AppConfig,
):
    """Optuna objective function for hyperparameter tuning."""
    start_time = time.time()
    base_models = get_base_models(config)
    tuning_cfg = config.tuning
    data_cfg = config.data

    # 1. Create Data Processing Pipeline
    data_processing_pipeline = create_data_processing_pipeline(
        numerical_vars=data_cfg.numerical_vars,
        categorical_vars=data_cfg.categorical_vars,
    )

    # 2. Get Base Model Instance
    if model_name not in base_models:
        raise optuna.exceptions.TrialPruned(f"Model {model_name} not defined.")
    base_model = base_models[model_name].__class__(
        **base_models[model_name].get_params()
    )

    # 3. Create Full Scikit-learn Pipeline
    pipeline = Pipeline(
        steps=[
            ("data_processing", data_processing_pipeline),
            ("classifier", base_model),
        ]
    )

    # 4. Get & Set Hyperparameters
    params_to_tune = get_optuna_params(
        trial, model_name, y_train_for_imbalance=y_train.to_numpy()
    )
    if model_name == "LogisticRegression":
        penalty = params_to_tune.get("classifier__penalty")
        if penalty != "elasticnet":
            params_to_tune.pop("classifier__l1_ratio", None)
    try:
        pipeline.set_params(**params_to_tune)
        logger.trace(f"Trial {trial.number}: Set params {params_to_tune}")
    except ValueError as e:
        logger.warning(
            f"Error setting parameters for {model_name} trial {trial.number}: {params_to_tune}. Error: {e}"
        )
        raise optuna.exceptions.TrialPruned(f"Incompatible parameters: {e}")

    # 5. Perform Cross-Validation
    cv = StratifiedKFold(
        n_splits=tuning_cfg.cv_folds, shuffle=True, random_state=data_cfg.random_state
    )
    optimization_metric = tuning_cfg.optimization_metric
    try:
        scorer = get_scorer(optimization_metric)
    except ValueError:
        raise optuna.exceptions.TrialPruned(f"Invalid scorer: {optimization_metric}")

    # 6. MLflow Logging (Nested Run)
    with mlflow.start_run(nested=True, run_name=f"Trial_{trial.number}") as trial_run:
        mlflow.log_params(
            {
                k.replace("classifier__", ""): v
                for k, v in trial.params.items()
                if v is not None
            }
        )
        mlflow.log_param("model_name", model_name)
        mlflow.set_tag("optuna_trial_number", trial.number)

        try:
            logger.debug(
                f"Trial {trial.number}: Starting CV with {tuning_cfg.cv_folds} folds..."
            )
            if model_name == "XGBoost":
                logger.debug("Applying feature name sanitization for XGBoost trial.")
                try:
                    X_processed = pipeline.named_steps["data_processing"].fit_transform(
                        X_train, y_train
                    )
                    X_processed_sanitized = _sanitize_column_names(X_processed.copy())
                    classifier_step = pipeline.named_steps["classifier"]
                    scores = cross_val_score(
                        classifier_step,
                        X_processed_sanitized,
                        y_train,
                        n_jobs=1,
                        cv=cv,
                        scoring=scorer,
                        error_score="raise",
                    )
                except Exception as xgb_e:
                    logger.error(
                        f"Error during sanitized XGBoost CV: {xgb_e}", exc_info=True
                    )
                    raise
            else:
                scores = cross_val_score(
                    pipeline,
                    X_train,
                    y_train,
                    n_jobs=1,
                    cv=cv,
                    scoring=scorer,
                    error_score="raise",
                )

            mean_score = float(np.mean(scores))
            std_score = float(np.std(scores))
            duration = time.time() - start_time
            logger.debug(
                f"Trial {trial.number}: CV scores = {scores}, Mean = {mean_score:.5f}, Std = {std_score:.5f}, Duration = {duration:.2f}s"
            )

            mlflow.log_metric(f"cv_{optimization_metric}_mean", mean_score)
            mlflow.log_metric(f"cv_{optimization_metric}_std", std_score)
            mlflow.log_metric("cv_duration_seconds", duration)
            mlflow.set_tag("status", "completed")

        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                f"CV failed for {model_name} trial {trial.number} after {duration:.2f}s: {e}",
                exc_info=True,
            )
            mlflow.log_metric(f"cv_{optimization_metric}_mean", -np.inf)
            mlflow.log_metric("cv_duration_seconds", duration)
            mlflow.set_tag("status", "failed_cv")
            raise optuna.exceptions.TrialPruned(f"Cross-validation failed: {e}")

    return mean_score


# --- Main Tuning Function ---
def run_tuning(config: AppConfig):
    """Orchestrates the hyperparameter tuning process."""
    logger.info("--- Starting Hyperparameter Tuning Workflow ---")
    tuning_cfg = config.tuning
    data_cfg = config.data

    # --- Load Data ---
    try:
        X_train, X_test, y_train, y_test = load_processed_data(data_cfg)
        logger.info("Loaded processed data for tuning.")
    except FileNotFoundError:
        logger.warning(
            "Processed data not found. Attempting to run preprocessing workflow..."
        )
        if run_preprocess_workflow(config):
            logger.info("Preprocessing workflow completed. Reloading processed data...")
            try:
                X_train, X_test, y_train, y_test = load_processed_data(data_cfg)
            except Exception as e:
                raise RuntimeError(
                    "Failed to load processed data even after running workflow."
                ) from e
        else:
            raise RuntimeError("Data preprocessing failed.")
    except Exception as e:
        raise RuntimeError("Failed to load data for tuning.") from e

    # --- MLflow Setup ---
    mlflow.set_experiment(tuning_cfg.mlflow_experiment_name)
    logger.info(f"MLflow Experiment set to: {tuning_cfg.mlflow_experiment_name}")

    overall_best_score = -np.inf
    overall_best_model_name = None
    all_models_best_results = {}

    # --- Loop Through Models ---
    for model_name in tuning_cfg.models_to_tune:
        logger.info(f"\n===== Tuning Model: {model_name} =====")
        with mlflow.start_run(run_name=f"Tune_{model_name}") as parent_run:
            parent_run_id = parent_run.info.run_id
            logger.info(f"Started MLflow parent run for {model_name}: {parent_run_id}")
            mlflow.log_param("model_type", model_name)
            mlflow.set_tag("tuning_status", "running")

            study = optuna.create_study(
                direction="maximize",
                study_name=f"{model_name}_tuning_{parent_run_id}",
                pruner=optuna.pruners.MedianPruner(n_warmup_steps=5, n_min_trials=10),
            )
            try:
                study.optimize(
                    lambda trial: objective(
                        trial, model_name, X_train, y_train, config
                    ),
                    n_trials=tuning_cfg.optuna_trials_per_model,
                    n_jobs=1,
                    show_progress_bar=True,
                )
                mlflow.set_tag("tuning_status", "completed")
                logger.info(f"Optuna tuning completed for {model_name}.")
            except Exception as e:
                logger.error(
                    f"Optuna study optimize call failed for {model_name}: {e}",
                    exc_info=True,
                )
                mlflow.set_tag("tuning_status", "failed")
                continue

            # --- Process Study Results ---
            try:
                completed_trials = [
                    t
                    for t in study.trials
                    if t.state == optuna.trial.TrialState.COMPLETE
                ]
                if not completed_trials:
                    raise ValueError("No trials completed successfully for this study.")

                best_trial = study.best_trial
                current_best_score = best_trial.value
                all_models_best_results[model_name] = {
                    "best_score": current_best_score,
                    "best_hyperparameters": best_trial.params,
                    "mlflow_run_id": parent_run_id,
                    "status": "success",
                }

                logger.info(f"Best Trial Number for {model_name}: {best_trial.number}")
                logger.info(
                    f"Best CV {tuning_cfg.optimization_metric} Score: {current_best_score:.5f}"
                )
                logger.info(f"Best Params: {best_trial.params}")

                best_params_cleaned = {
                    k.replace("classifier__", ""): v
                    for k, v in best_trial.params.items()
                    if v is not None
                }
                mlflow.log_params(best_params_cleaned)
                mlflow.log_metric(
                    f"best_cv_{tuning_cfg.optimization_metric}", current_best_score
                )
                mlflow.set_tag("best_trial_number", best_trial.number)
                if (
                    current_best_score > -np.inf
                    and current_best_score > overall_best_score
                ):
                    overall_best_score = current_best_score
                    overall_best_model_name = model_name
                    logger.success(
                        f"*** New Overall Best Model Found: {model_name} (CV Score: {overall_best_score:.5f}) ***"
                    )
                    mlflow.set_tag("is_current_best", "True")

            except ValueError as e:
                logger.warning(
                    f"Optuna study for {model_name} finished without any successful trials: {e}"
                )
                mlflow.set_tag("tuning_status", "no_successful_trials")
                all_models_best_results[model_name] = {
                    "status": "no_successful_trials",
                    "mlflow_run_id": parent_run_id,
                }
            except Exception as e:
                logger.error(
                    f"Error processing Optuna results for {model_name}: {e}",
                    exc_info=True,
                )
                mlflow.set_tag("tuning_status", "result_processing_error")
                all_models_best_results[model_name] = {
                    "status": "result_processing_error",
                    "mlflow_run_id": parent_run_id,
                }

    # --- MODIFICATION: Save ALL Best Parameters ---
    if all_models_best_results:
        logger.info(f"\n===== Tuning Results Summary =====")
        if overall_best_model_name:
            logger.info(f"Overall Best Model Type: {overall_best_model_name}")
            logger.info(f"Overall Best CV Score: {overall_best_score:.5f}")
        else:
            logger.warning("No overall best model identified among successful runs.")

        results_to_save = {
            "overall_best_model_name": overall_best_model_name,
            "overall_best_score": (
                overall_best_score if overall_best_score > -np.inf else None
            ),
            "models": all_models_best_results,
        }

        output_path = Path(tuning_cfg.all_best_params_output_path)
        try:
            save_json(results_to_save, output_path)
            logger.info(f"All models' best parameters saved to: {output_path}")

            with mlflow.start_run(run_name="Tuning_Summary") as summary_run:
                mlflow.log_param(
                    "overall_best_model",
                    overall_best_model_name if overall_best_model_name else "N/A",
                )
                if overall_best_score > -np.inf:
                    mlflow.log_metric("overall_best_cv_score", overall_best_score)
                for model, result in all_models_best_results.items():
                    mlflow.log_param(f"{model}_status", result.get("status", "unknown"))
                    if result.get("status") == "success":
                        mlflow.log_metric(
                            f"{model}_best_cv_score", result.get("best_score")
                        )
                mlflow.log_artifact(str(output_path))
                logger.info(
                    f"Tuning summary logged to MLflow Run ID: {summary_run.info.run_id}"
                )

        except Exception as e:
            logger.error(
                f"Failed to save or log all best parameters/summary: {e}", exc_info=True
            )

    else:
        logger.warning(
            "Tuning finished, but no models were successfully tuned or results recorded."
        )

    logger.info("--- Hyperparameter Tuning Workflow Finished ---")
    return (
        Path(tuning_cfg.all_best_params_output_path)
        if all_models_best_results
        else None
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run hyperparameter tuning for churn model."
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config file.")
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
        run_tuning(app_config)
    except FileNotFoundError:
        logger.critical("Config file not found.")
        sys.exit(1)
    except RuntimeError as e:
        logger.critical(f"Runtime error during tuning: {e}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Tuning script failed: {e}", exc_info=True)
        sys.exit(1)
