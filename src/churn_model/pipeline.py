# ============================================
# File: src/churn_model/pipeline.py
# ============================================
import re

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from .processing import AgeBinner, FeatureRatioCalculator

def _sanitize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Internal function to sanitize column names for XGBoost compatibility."""
    original_columns = df.columns.tolist()
    sanitized_columns = []
    for col in original_columns:
        sanitized_col = re.sub(r"[\[\]<>]", "_", str(col))
        sanitized_columns.append(sanitized_col)
    if len(sanitized_columns) != len(set(sanitized_columns)):
        logger.warning("Duplicate column names generated after sanitization.")
    df.columns = sanitized_columns
    return df

SanitizeNamesTransformer = FunctionTransformer(_sanitize_column_names, validate=False)


def create_data_processing_pipeline(
    numerical_vars: list, categorical_vars: list
) -> Pipeline:
    """
    Creates the data preprocessing and feature engineering pipeline.
    Output is set to pandas DataFrame.
    """
    logger.info("Creating data processing pipeline...")
    ratio_creator = FeatureRatioCalculator(
        ratio_pairs=[("Balance", "EstimatedSalary"), ("CreditScore", "Age")]
    )
    age_binner = AgeBinner()
    numerical_imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    feature_engineering = Pipeline(
        steps=[("ratio_features", ratio_creator), ("age_binning", age_binner)]
    )
    try:
        feature_engineering.set_output(transform="pandas")
    except AttributeError:
        logger.warning("Could not set feature_engineering output to pandas.")

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                Pipeline(steps=[("imputer", numerical_imputer), ("scaler", scaler)]),
                make_column_selector(dtype_include=np.number),
            ),
            (
                "categorical",
                Pipeline(steps=[("onehot", encoder)]),
                make_column_selector(dtype_include=[object, "category"]),
            ),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    try:
        preprocessor.set_output(transform="pandas")
    except AttributeError:
        logger.warning("Could not set preprocessor output to pandas.")

    data_processing_pipeline = Pipeline(
        steps=[
            ("feature_engineering", feature_engineering),
            ("preprocessing", preprocessor),
        ]
    )
    try:
        data_processing_pipeline.set_output(transform="pandas")
    except AttributeError:
        logger.warning("Could not set data_processing_pipeline output to pandas.")

    logger.info("Data processing pipeline created successfully.")
    return data_processing_pipeline
