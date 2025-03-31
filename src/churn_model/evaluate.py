# ============================================
# File: src/churn_model/evaluate.py
# ============================================
import argparse  # re
import sys
import time
from pathlib import Path
from typing import Optional  # Any, Dict, List

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import shap
from loguru import logger
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from .config import PROJECT_ROOT, AppConfig, load_config
from .pipeline import _sanitize_column_names
from .processing import load_processed_data
from .utils import load_json, load_pipeline_joblib, setup_logging

# --- Setup ---
setup_logging()


def plot_roc_curve(y_true, y_proba, model_name, output_path: Path):
    """Generates and saves ROC curve plot."""
    try:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc_score = roc_auc_score(y_true, y_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(
            fpr,
            tpr,
            label=f"ROC Curve (AUC = {auc_score:.3f})",
            color="darkorange",
            lw=2,
        )
        plt.plot(
            [0, 1],
            [0, 1],
            "k--",
            label="Random Guess",
            color="navy",
            lw=2,
            linestyle="--",
        )
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{model_name} ROC Curve (Test Set)")
        plt.legend(loc="lower right")
        plt.grid(alpha=0.5)
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
        logger.info(f"ROC curve saved to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to generate ROC curve: {e}", exc_info=True)
        return False


def plot_confusion_matrix(y_true, y_pred, model_name, output_path: Path):
    """Generates and saves confusion matrix plot."""
    try:
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        fig, ax = plt.subplots(figsize=(7, 6))
        disp.plot(cmap=plt.cm.Blues, ax=ax)
        ax.set_title(f"{model_name} Confusion Matrix (Test Set)")
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
        logger.info(f"Confusion matrix saved to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to generate confusion matrix: {e}", exc_info=True)
        return False


def plot_precision_recall_curve(y_true, y_proba, model_name, output_path: Path):
    """Generates and saves Precision-Recall curve plot."""
    try:
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        avg_precision = average_precision_score(y_true, y_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(
            recall,
            precision,
            label=f"{model_name} (AP = {avg_precision:.3f})",
            color="teal",
            lw=2,
        )
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"{model_name} Precision-Recall Curve (Test Set)")
        plt.legend(loc="upper right")
        plt.grid(alpha=0.5)
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
        logger.info(f"Precision-Recall curve saved to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to generate Precision-Recall curve: {e}", exc_info=True)
        return False


# --- SHAP Analysis Function ---
def run_shap_analysis(pipeline, X_train, X_test, config: AppConfig):
    """Performs SHAP analysis and saves plots."""
    logger.info("--- Starting SHAP Analysis ---")
    shap_start_time = time.time()
    eval_cfg = config.evaluation
    data_cfg = config.data
    output_dir = Path(eval_cfg.shap_plots_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = "UnknownModel"
    try:
        preprocessor_step = pipeline.named_steps["data_processing"]
        model_step = pipeline.named_steps["classifier"]
        model_type = model_step.__class__.__name__
        model_name = model_type
        logger.info(f"Running SHAP for model type: {model_type}")
    except KeyError as e:
        logger.error(f"Pipeline steps not found: {e}. Cannot run SHAP.")
        return False
    except Exception as e:
        logger.error(f"Error accessing pipeline steps: {e}", exc_info=True)
        return False
    logger.info("Preprocessing data for SHAP using fitted preprocessor...")
    try:
        X_train_processed = preprocessor_step.transform(X_train)
        X_test_processed = preprocessor_step.transform(X_test)
        if isinstance(X_train_processed, pd.DataFrame):
            feature_names = X_train_processed.columns.tolist()
            logger.info(f"Data preprocessed for SHAP. Features: {len(feature_names)}")
        else:
            logger.warning(
                "Preprocessor output is not DataFrame. Attempting to get feature names."
            )
            try:
                feature_names = preprocessor_step.get_feature_names_out()
            except AttributeError:
                feature_names = [
                    f"feature_{i}" for i in range(X_train_processed.shape[1])
                ]
                logger.warning("Using generic feature names.")
    except Exception as e:
        logger.error(f"Failed to preprocess data for SHAP: {e}", exc_info=True)
        return False
    if model_type == "XGBClassifier":
        logger.debug("Sanitizing feature names for XGBoost SHAP analysis.")
        X_train_processed = _sanitize_column_names(X_train_processed.copy())
        X_test_processed = _sanitize_column_names(X_test_processed.copy())
        feature_names = X_test_processed.columns.tolist()
    explainer = None
    shap_values = None
    expected_value = None
    X_test_subset_for_plot = X_test_processed
    try:
        if model_type in ["RandomForestClassifier", "LGBMClassifier", "XGBClassifier"]:
            logger.info("Using shap.TreeExplainer.")
            explainer = shap.TreeExplainer(model_step, data=X_train_processed)
            logger.info("Calculating SHAP values (TreeExplainer)...")
            calc_start = time.time()
            shap_values = explainer.shap_values(X_test_processed)
            logger.info(
                f"SHAP values calculated (TreeExplainer) in {time.time() - calc_start:.2f}s"
            )
            expected_value = explainer.expected_value
        elif model_type in ["SVC", "LogisticRegression"]:
            logger.info("Using shap.KernelExplainer (can be slow).")

            def predict_proba_wrapper(data_as_np):
                data_df = pd.DataFrame(data_as_np, columns=feature_names)
                return model_step.predict_proba(data_df)

            num_bg_samples = min(
                eval_cfg.shap_kernel_background_samples, X_train_processed.shape[0]
            )
            logger.info(
                f"Creating KernelExplainer background dataset using kmeans ({num_bg_samples} samples)..."
            )
            background_data = shap.kmeans(
                X_train_processed, num_bg_samples, random_state=data_cfg.random_state
            )
            explainer = shap.KernelExplainer(predict_proba_wrapper, background_data)
            num_test_samples = min(100, X_test_processed.shape[0])
            X_test_subset_for_plot = X_test_processed.iloc[:num_test_samples, :]
            logger.info(
                f"Calculating SHAP values for {num_test_samples} test samples (KernelExplainer)..."
            )
            calc_start = time.time()
            shap_values = explainer.shap_values(X_test_subset_for_plot, nsamples="auto")
            logger.info(
                f"SHAP values calculated (KernelExplainer) in {time.time() - calc_start:.2f}s"
            )
            expected_value = explainer.expected_value
        else:
            logger.warning(
                f"SHAP analysis not implemented for model type: {model_type}."
            )
            return True
        logger.info("SHAP values calculation complete.")
    except Exception as e:
        logger.error(f"Failed during SHAP calculation: {e}", exc_info=True)
        return False
    plot_success = True
    if explainer and shap_values is not None:
        logger.info("Generating SHAP plots...")
        try:
            if isinstance(shap_values, list) and len(shap_values) == 2:
                shap_values_pos = shap_values[1]
                expected_value_pos = (
                    expected_value[1]
                    if isinstance(expected_value, (list, np.ndarray))
                    and len(expected_value) == 2
                    else expected_value
                )
            else:
                shap_values_pos = shap_values
                expected_value_pos = expected_value
            data_for_plots = X_test_subset_for_plot
            plot_tasks = {
                "summary_bar": lambda: shap.summary_plot(
                    shap_values_pos,
                    data_for_plots,
                    plot_type="bar",
                    show=False,
                    feature_names=feature_names,
                ),
                "summary_dot": lambda: shap.summary_plot(
                    shap_values_pos,
                    data_for_plots,
                    show=False,
                    feature_names=feature_names,
                ),
            }
            for plot_name, plot_func in plot_tasks.items():
                try:
                    plt.figure()
                    plot_func()
                    plt.title(
                        f"SHAP {plot_name.replace('_', ' ').title()} ({model_name})"
                    )
                    plt.tight_layout()
                    plt.savefig(
                        output_dir / f"{model_name}_shap_{plot_name}.png",
                        bbox_inches="tight",
                    )
                    plt.close()
                    logger.info(f"Saved SHAP {plot_name} plot.")
                except Exception as e_plt:
                    logger.error(
                        f"Failed to generate SHAP {plot_name} plot: {e_plt}",
                        exc_info=True,
                    )
                    plot_success = False
            try:
                mean_abs_shap = np.abs(shap_values_pos).mean(axis=0)
                feature_indices = np.argsort(mean_abs_shap)[::-1]
                num_dep_plots = min(10, len(feature_names))
                logger.info(
                    f"Generating SHAP dependence plots for top {num_dep_plots} features..."
                )
                for i in range(num_dep_plots):
                    idx = feature_indices[i]
                    if idx < len(feature_names):
                        name = feature_names[idx]
                        try:
                            plt.figure()
                            shap.dependence_plot(
                                idx,
                                shap_values_pos,
                                data_for_plots,
                                feature_names=feature_names,
                                interaction_index="auto",
                                show=False,
                            )
                            plt.title(f"SHAP Dependence: {name} ({model_name})")
                            plt.tight_layout()
                            plt.savefig(
                                output_dir / f"{model_name}_shap_dependence_{name}.png",
                                bbox_inches="tight",
                            )
                            plt.close()
                        except Exception as e_dep:
                            logger.warning(
                                f"Could not generate dependence plot for feature '{name}' (index {idx}): {e_dep}"
                            )
                    else:
                        logger.warning(
                            f"Feature index {idx} out of bounds for feature names list (length {len(feature_names)}). Skipping dependence plot."
                        )
            except Exception as e_dep_loop:
                logger.error(
                    f"Error during dependence plot loop: {e_dep_loop}", exc_info=True
                )
                plot_success = False
            instance_idx = 0
            logger.info(f"Generating SHAP force plot for instance {instance_idx}...")
            try:
                base_value_float = float(expected_value_pos)
                """ force_plot = shap.force_plot(
                    base_value_float,
                    shap_values_pos[instance_idx, :],
                    data_for_plots.iloc[instance_idx, :],
                    feature_names=feature_names,
                    matplotlib=True,
                    show=False,
                ) """
                plt.title(f"SHAP Force Plot: Instance {instance_idx} ({model_name})")
                plt.savefig(
                    output_dir
                    / f"{model_name}_shap_force_plot_instance_{instance_idx}.png",
                    bbox_inches="tight",
                )
                plt.close()
                logger.info(f"SHAP force plot saved for instance {instance_idx}.")
            except Exception as e_force:
                logger.error(
                    f"Failed to generate or save SHAP force plot: {e_force}",
                    exc_info=True,
                )
                plot_success = False
            logger.info(f"SHAP plots saved to {output_dir}")
        except Exception as e_plot_main:
            logger.error(
                f"An error occurred during SHAP plot generation: {e_plot_main}",
                exc_info=True,
            )
            plot_success = False
    else:
        logger.warning(
            "SHAP explainer or values not available. Skipping plot generation."
        )
        plot_success = True if explainer is None else False
    shap_duration = time.time() - shap_start_time
    logger.info(f"--- SHAP Analysis Finished. Duration: {shap_duration:.2f}s ---")
    return plot_success


# --- Main Evaluation Function ---
def run_evaluation(config: AppConfig, model_path_override: Optional[str] = None):
    """Loads the final model and evaluates it on the test set."""
    logger.info("--- Starting Model Evaluation Workflow ---")
    eval_start_time = time.time()
    eval_cfg = config.evaluation
    data_cfg = config.data
    training_cfg = config.training
    plots_dir = Path(eval_cfg.plots_output_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    model_path = None
    model_name = "UnknownModel"
    try:
        if model_path_override:
            model_path = Path(model_path_override)
            if not model_path.is_absolute():
                model_path = PROJECT_ROOT / model_path
            logger.info(f"Evaluating specified model file: {model_path}")
        else:
            all_params_path = Path(training_cfg.all_best_params_input_path)
            if not all_params_path.exists():
                raise FileNotFoundError(
                    f"Tuning results file not found at {all_params_path}."
                )
            all_params_data = load_json(all_params_path)
            overall_best_model_name = all_params_data.get("overall_best_model_name")
            if not overall_best_model_name:
                raise ValueError(
                    "Overall best model name not found in tuning results file."
                )
            model_path = Path(
                f"{training_cfg.final_model_output_base_path}_{overall_best_model_name}.joblib"
            )
            logger.info(
                f"Evaluating overall best model identified during tuning: {overall_best_model_name} from path: {model_path}"
            )
        if not model_path or not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found at determined path: {model_path}"
            )
        pipeline = load_pipeline_joblib(model_path)
        logger.info(f"Loaded final model pipeline from {model_path}")
        if "classifier" in pipeline.named_steps:
            model_name = pipeline.named_steps["classifier"].__class__.__name__
    except FileNotFoundError as e:
        logger.critical(f"Model or parameters file not found: {e}. Cannot evaluate.")
        raise
    except ValueError as e:
        logger.critical(f"Error determining model to evaluate: {e}")
        raise
    except Exception as e:
        logger.critical(f"Error loading model: {e}", exc_info=True)
        raise
    try:
        X_train, X_test, y_train, y_test = load_processed_data(data_cfg)
        logger.info(f"Loaded processed data for evaluation. Test shape: {X_test.shape}")
    except FileNotFoundError:
        logger.critical("Processed data not found.")
        raise
    except Exception as e:
        logger.critical(f"Error loading processed data: {e}", exc_info=True)
        raise
    logger.info("Making predictions on the test set...")
    y_pred, y_proba, has_proba = None, None, False
    try:
        pred_start = time.time()
        y_pred = pipeline.predict(X_test)
        if hasattr(pipeline, "predict_proba"):
            y_proba = pipeline.predict_proba(X_test)[:, 1]
            has_proba = True
        else:
            logger.warning(f"Model {model_name} does not support predict_proba.")
        pred_duration = time.time() - pred_start
        logger.info(f"Predictions completed in {pred_duration:.2f}s")
    except Exception as e:
        logger.error(f"Error during prediction on test set: {e}", exc_info=True)
        raise
    logger.info("Calculating evaluation metrics...")
    metrics = {}
    metrics_success = True
    try:
        if has_proba:
            metrics["roc_auc"] = roc_auc_score(y_test, y_proba)
        else:
            metrics["roc_auc"] = np.nan
        metrics["f1_score"] = f1_score(y_test, y_pred)
        metrics["balanced_accuracy"] = balanced_accuracy_score(y_test, y_pred)
        metrics["precision"] = precision_score(y_test, y_pred, zero_division=0)
        metrics["recall"] = recall_score(y_test, y_pred, zero_division=0)
        metrics["mcc"] = matthews_corrcoef(y_test, y_pred)
        logger.info("--- Test Set Metrics ---")
        [logger.info(f"{n}: {v:.4f}") for n, v in metrics.items()]
        logger.info("------------------------")
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}", exc_info=True)
        metrics_success = False
    logger.info("Generating evaluation plots...")
    plots_success = True
    plot_files = []
    try:
        if has_proba:
            roc_path = plots_dir / f"{model_name}_test_roc_curve.png"
            pr_path = plots_dir / f"{model_name}_test_pr_curve.png"
            if plot_roc_curve(y_test, y_proba, model_name, roc_path):
                plot_files.append(roc_path)
            else:
                plots_success = False

            if plot_precision_recall_curve(y_test, y_proba, model_name, pr_path):
                plot_files.append(pr_path)
            else:
                plots_success = False

        cm_path = plots_dir / f"{model_name}_test_confusion_matrix.png"
        if plot_confusion_matrix(y_test, y_pred, model_name, cm_path):
            plot_files.append(cm_path)
        else:
            plots_success = False
    except Exception as e:
        logger.error(f"Error generating plots: {e}", exc_info=True)
        plots_success = False
    logger.info("Logging evaluation results to MLflow...")
    mlflow_success = True
    try:
        mlflow.set_experiment(eval_cfg.mlflow_experiment_name)
        with mlflow.start_run(run_name=f"Evaluation_{model_name}") as run:
            run_id = run.info.run_id
            logger.info(f"MLflow Run ID for evaluation: {run_id}")
            mlflow.log_param("evaluated_model_name", model_name)
            mlflow.log_param("evaluated_model_path", str(model_path))
            if metrics_success:
                mlflow.log_metrics(metrics)
            else:
                mlflow.set_tag("metrics_status", "failed")
            if plots_success and plot_files:
                mlflow.log_artifacts(str(plots_dir), artifact_path="evaluation_plots")
                logger.info(f"Logged evaluation plots from {plots_dir} to MLflow.")
            else:
                mlflow.set_tag("plots_status", "failed or incomplete")
            shap_success = run_shap_analysis(pipeline, X_train, X_test, config)
            mlflow.set_tag("shap_status", "success" if shap_success else "failed")
    except Exception as e:
        logger.error(f"Failed to log results to MLflow: {e}", exc_info=True)
        mlflow_success = False
    eval_duration = time.time() - eval_start_time
    logger.info(
        f"--- Model Evaluation Workflow Finished. Duration: {eval_duration:.2f}s ---"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate the final churn prediction model."
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config file.")
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to specific .joblib model file to evaluate. If None, evaluates the overall best model based on tuning results.",
    )
    args = parser.parse_args()
    try:
        from typing import Optional

        from .config import PROJECT_ROOT, load_config
        from .utils import setup_logging

        setup_logging()
    except ImportError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    try:
        config_path = PROJECT_ROOT / args.config
        app_config = load_config(config_path)
        run_evaluation(app_config, model_path_override=args.model_path)
    except FileNotFoundError:
        logger.critical("Config/Model/Params file not found.")
        sys.exit(1)
    except ValueError as e:
        logger.critical(f"Value error during evaluation setup: {e}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Evaluation script failed: {e}", exc_info=True)
        sys.exit(1)
