"""
inference.py

Loads the already-trained XGBoost model artifact and generates
predictions on an origin dataset built by feature_engineering.py.
No training happens here -- this module only consumes the artifacts
already produced by Favorita_GlobalModel.py.
"""

from pathlib import Path

import pandas as pd
from xgboost import XGBRegressor

from .feature_engineering import MODEL_FEATURES

CATEGORICAL_FEATURES = [
    "store_nbr", "family", "store_type", "city", "state", "store_cluster",
]


def load_model(model_artifact_path: str) -> XGBRegressor:
    """Load the trained XGBoost model from its saved JSON artifact."""
    path = Path(model_artifact_path)
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path}")

    model = XGBRegressor()
    model.load_model(path)
    return model


def load_categorical_reference(train_df_artifact_path: str) -> dict:
    """
    Load the categories used during training for each categorical
    feature, so inference-time data can be cast to the exact same
    categories (required by XGBoost's categorical support).
    """
    path = Path(train_df_artifact_path)
    if not path.exists():
        raise FileNotFoundError(f"Training dataset artifact not found: {path}")

    train_df = pd.read_pickle(path)
    reference = {}
    for column in CATEGORICAL_FEATURES:
        reference[column] = train_df[column].cat.categories
    return reference


def align_categoricals(origin_df: pd.DataFrame, categorical_reference: dict) -> pd.DataFrame:
    """Cast categorical columns to match the categories seen at training time."""
    df = origin_df.copy()
    for column, categories in categorical_reference.items():
        df[column] = pd.Categorical(df[column], categories=categories)
    return df


def predict(model: XGBRegressor, origin_df: pd.DataFrame,
            categorical_reference: dict) -> pd.DataFrame:
    """
    Run inference on an origin dataset (output of
    feature_engineering.build_origin_dataset) and attach a
    `forecast` column to it.
    """
    aligned = align_categoricals(origin_df, categorical_reference)
    X = aligned[MODEL_FEATURES].copy()

    predictions = model.predict(X)

    result = origin_df.copy()
    result["forecast"] = predictions
    return result
