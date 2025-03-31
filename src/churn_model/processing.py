# ============================================
# File: src/churn_model/processing.py
# ============================================
import argparse
import sys
from pathlib import Path
from typing import List, Tuple  # Any, Dict, import numpy as np

import pandas as pd
from loguru import logger
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split

from .config import PROJECT_ROOT, AppConfig, DataConfig
from .utils import setup_logging

# --- Setup Logging ---
# setup_logging() # Called explicitly in main scripts or __main__ block

# --- Custom Transformers ---


class FeatureRatioCalculator(BaseEstimator, TransformerMixin):
    """Calculates specified feature ratios."""

    def __init__(self, ratio_pairs: List[Tuple[str, str]], epsilon: float = 1e-6):
        if not isinstance(ratio_pairs, list) or not all(
            isinstance(pair, tuple) and len(pair) == 2 for pair in ratio_pairs
        ):
            raise ValueError(
                "ratio_pairs must be a list of tuples (numerator, denominator)"
            )
        self.ratio_pairs = ratio_pairs
        self.epsilon = epsilon
        self._feature_names_out_suffix = []

    def fit(self, X: pd.DataFrame, y=None):
        self._feature_names_out_suffix = []
        required_cols = set()
        for num, den in self.ratio_pairs:
            required_cols.add(num)
            required_cols.add(den)
            self._feature_names_out_suffix.append(f"{num}_to_{den}_Ratio")
        missing_cols = required_cols - set(X.columns)
        if missing_cols:
            raise ValueError(f"Missing columns required for ratios: {missing_cols}")
        self.feature_names_in_ = X.columns.tolist()
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_transformed = X.copy()
        logger.debug(f"Calculating ratios: {self.ratio_pairs}")
        for num, den in self.ratio_pairs:
            ratio_col_name = f"{num}_to_{den}_Ratio"
            try:
                num_col = pd.to_numeric(X_transformed[num], errors="coerce")
                den_col = pd.to_numeric(X_transformed[den], errors="coerce")
                X_transformed[ratio_col_name] = num_col / (den_col + self.epsilon)
                X_transformed[ratio_col_name] = X_transformed[ratio_col_name].fillna(0)
            except KeyError as e:
                logger.error(f"Column not found during ratio calculation: {e}")
                raise
            except Exception as e:
                logger.error(
                    f"Error calculating ratio {ratio_col_name}: {e}", exc_info=True
                )
                raise
        return X_transformed

    def get_feature_names_out(self, input_features=None):
        """Return feature names for output features."""
        if input_features is None:
            if hasattr(self, "feature_names_in_"):
                input_features_ = self.feature_names_in_
            else:
                raise ValueError(
                    "Cannot determine input feature names in get_feature_names_out. Ensure fit is called."
                )
        else:
            input_features_ = input_features
        return list(input_features_) + self._feature_names_out_suffix


class AgeBinner(BaseEstimator, TransformerMixin):
    """Bins the 'Age' column into categories."""

    def __init__(
        self,
        bins=[0, 30, 40, 50, 60, 100],
        labels=["<30", "30-39", "40-49", "50-59", "60+"],
    ):
        self.bins = bins
        self.labels = labels
        self.feature_name_out_suffix = "Age_Bin"

    def fit(self, X, y=None):
        if "Age" not in X.columns:
            raise ValueError("Input DataFrame must contain 'Age' column for AgeBinner.")
        self.feature_names_in_ = X.columns.tolist()
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X_transformed = X.copy()
        logger.debug("Binning Age feature...")
        try:
            X_transformed[self.feature_name_out_suffix] = pd.cut(
                X_transformed["Age"],
                bins=self.bins,
                labels=self.labels,
                right=False,
                include_lowest=True,
            )
            X_transformed[self.feature_name_out_suffix] = X_transformed[
                self.feature_name_out_suffix
            ].astype(object)
        except KeyError:
            logger.error("Column 'Age' not found during AgeBinner transform.")
            raise
        except Exception as e:
            logger.error(f"Error during Age binning: {e}", exc_info=True)
            raise
        return X_transformed

    def get_feature_names_out(self, input_features=None):
        """Return feature names for output features."""
        if input_features is None:
            if hasattr(self, "feature_names_in_"):
                input_features_ = self.feature_names_in_
            else:
                raise ValueError(
                    "Cannot determine input feature names in get_feature_names_out. Ensure fit is called."
                )
        else:
            input_features_ = input_features
        output_features = list(input_features_)
        if self.feature_name_out_suffix not in output_features:
            output_features.append(self.feature_name_out_suffix)
        return output_features


# --- Data Loading and Processing Functions ---
def load_raw_data(file_path: Path) -> pd.DataFrame:
    """Loads raw data from CSV, handling potential encoding issues."""
    logger.info(f"Loading raw data from {file_path}...")
    if not file_path.exists():
        logger.error(f"Data file not found at {file_path}")
        raise FileNotFoundError(f"Data file not found at {file_path}")
    try:
        df = pd.read_csv(file_path, encoding="utf-8")
    except UnicodeDecodeError:
        logger.warning("UTF-8 decoding failed, trying 'latin-1' encoding...")
        try:
            df = pd.read_csv(file_path, encoding="latin-1")
        except Exception as e:
            logger.error(
                f"Failed to load raw data with UTF-8 and latin-1 encoding from {file_path}: {e}"
            )
            raise
    except pd.errors.EmptyDataError:
        logger.error(f"No data found in file: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading raw data from {file_path}: {e}", exc_info=True)
        raise
    logger.info(f"Raw data loaded successfully. Shape: {df.shape}")
    return df


def drop_irrelevant_features(df: pd.DataFrame) -> pd.DataFrame:
    """Drops predefined irrelevant columns if they exist."""
    cols_to_drop = ["RowNumber", "CustomerId", "Surname"]
    cols_to_drop_existing = [col for col in cols_to_drop if col in df.columns]
    if cols_to_drop_existing:
        df = df.drop(columns=cols_to_drop_existing)
        logger.info(f"Dropped irrelevant columns: {cols_to_drop_existing}")
    else:
        logger.debug("No standard irrelevant columns found to drop.")
    return df


def select_initial_features(
    df: pd.DataFrame, initial_features: List[str], target_column: str
) -> pd.DataFrame:
    """Selects only the features specified in config + target, ensuring they exist."""
    logger.debug(f"Selecting initial features: {initial_features}")
    cols_to_keep = initial_features + [target_column]
    cols_present = [col for col in cols_to_keep if col in df.columns]
    missing_cols = set(cols_to_keep) - set(cols_present)
    if missing_cols:
        logger.error(
            f"Initial features/target missing from DataFrame after cleaning: {missing_cols}"
        )
        raise ValueError(
            f"Columns specified in config.data.initial_features or target_column not found: {missing_cols}"
        )
    if set(cols_present) != set(cols_to_keep):
        logger.warning(
            f"Not all requested initial features were found. Using: {cols_present}"
        )
    return df[cols_present]


def split_data(
    df: pd.DataFrame, config: DataConfig
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Splits data into training and testing sets with stratification."""
    target_column = config.target_column
    test_size = config.test_size
    random_state = config.random_state
    if target_column not in df.columns:
        logger.error(f"Target column '{target_column}' not found for splitting.")
        raise ValueError(f"Target column '{target_column}' not found.")
    if df[target_column].isnull().any():
        logger.error(
            f"Target column '{target_column}' contains missing values. Cannot stratify."
        )
        raise ValueError(f"Target column '{target_column}' contains missing values.")
    logger.info(
        f"Splitting data with test_size={test_size}, random_state={random_state}..."
    )
    try:
        X = df.drop(target_column, axis=1)
        y = df[target_column]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        logger.info(
            f"Data split complete. Train shape: {X_train.shape}, Test shape: {X_test.shape}"
        )
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logger.error(f"Error during data splitting: {e}", exc_info=True)
        raise


def save_processed_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    config: DataConfig,
):
    """Saves processed data splits to parquet files."""
    train_path = Path(config.processed_train_path)
    test_path = Path(config.processed_test_path)
    try:
        train_path.parent.mkdir(parents=True, exist_ok=True)
        test_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"Could not create directories for processed data: {e}")
        raise
    try:
        train_df = X_train.copy()
        train_df[config.target_column] = y_train
        test_df = X_test.copy()
        test_df[config.target_column] = y_test
        logger.info(f"Saving processed training data to {train_path}...")
        train_df.to_parquet(train_path, index=False, engine="pyarrow")
        logger.info(f"Saving processed testing data to {test_path}...")
        test_df.to_parquet(test_path, index=False, engine="pyarrow")
        logger.info("Processed data saved successfully.")
    except ImportError:
        logger.error(
            "`pyarrow` engine not installed. Cannot save to parquet. Please install it (`poetry add pyarrow`)."
        )
        raise
    except Exception as e:
        logger.error(f"Error saving processed data to parquet: {e}", exc_info=True)
        raise


def load_processed_data(
    config: DataConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Loads processed data splits from parquet files."""
    train_path = Path(config.processed_train_path)
    test_path = Path(config.processed_test_path)
    target_column = config.target_column
    if not train_path.exists():
        logger.error(f"Processed training data file not found at {train_path}")
        raise FileNotFoundError(
            f"Processed training data not found at {train_path}. Run data processing first."
        )
    if not test_path.exists():
        logger.error(f"Processed testing data file not found at {test_path}")
        raise FileNotFoundError(
            f"Processed testing data not found at {test_path}. Run data processing first."
        )
    try:
        logger.info(f"Loading processed training data from {train_path}...")
        train_df = pd.read_parquet(train_path, engine="pyarrow")
        logger.info(f"Loading processed testing data from {test_path}...")
        test_df = pd.read_parquet(test_path, engine="pyarrow")
        X_train = train_df.drop(columns=[target_column])
        y_train = train_df[target_column]
        X_test = test_df.drop(columns=[target_column])
        y_test = test_df[target_column]
        logger.info("Processed data loaded successfully.")
        return X_train, X_test, y_train, y_test
    except ImportError:
        logger.error(
            "`pyarrow` engine not installed. Cannot load from parquet. Please install it (`poetry add pyarrow`)."
        )
        raise
    except Exception as e:
        logger.error(f"Error loading processed data from parquet: {e}", exc_info=True)
        raise


def run_preprocess_workflow(config: AppConfig) -> bool:
    """Executes the full data loading, cleaning, splitting, and saving workflow."""
    logger.info("--- Starting Data Preprocessing Workflow ---")
    data_cfg = config.data
    success = False
    try:
        raw_df = load_raw_data(Path(data_cfg.raw_path))
        clean_df = drop_irrelevant_features(raw_df)
        selected_df = select_initial_features(
            clean_df, data_cfg.initial_features, data_cfg.target_column
        )
        X_train, X_test, y_train, y_test = split_data(selected_df, data_cfg)
        save_processed_data(X_train, X_test, y_train, y_test, data_cfg)
        success = True
        logger.info("--- Data Preprocessing Workflow Completed Successfully ---")
    except FileNotFoundError as e:
        logger.error(f"Data preprocessing failed: {e}")
    except ValueError as e:
        logger.error(f"Data preprocessing failed due to value error: {e}")
    except Exception as e:
        logger.error(
            f"An unexpected error occurred during data preprocessing: {e}",
            exc_info=True,
        )
    return success


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run data preprocessing workflow.")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to the configuration file relative to project root.",
    )
    parser.add_argument(
        "--run", action="store_true", help="Run the full preprocessing workflow."
    )
    args = parser.parse_args()
    try:
        from .config import PROJECT_ROOT, load_config
        from .utils import setup_logging

        setup_logging()
    except ImportError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    if args.run:
        logger.info("Executing preprocessing workflow via command line...")
        try:
            config_path = PROJECT_ROOT / args.config
            app_config = load_config(config_path)
            success = run_preprocess_workflow(app_config)
            if not success:
                sys.exit(1)
        except FileNotFoundError:
            logger.critical("Configuration file not found.")
            sys.exit(1)
        except Exception as e:
            logger.critical(f"Preprocessing script failed: {e}", exc_info=True)
            sys.exit(1)
    else:
        logger.info(
            "Processing module loaded. Use the --run flag to execute the preprocessing workflow."
        )
