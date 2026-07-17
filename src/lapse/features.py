"""Model pipeline: feature encoding + XGBoost classifier."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from lapse.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, SEED

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Select the modeling columns in canonical order."""
    return df[ALL_FEATURES]


def build_pipeline(seed: int = SEED) -> Pipeline:
    """One-hot categoricals, passthrough numerics, XGBoost on top.

    Hyperparameters are conservative - shallow trees with subsampling -
    because at 10k rows the risk is overfitting, not underfitting.
    """
    encoder = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            ("num", "passthrough", NUMERIC_FEATURES),
        ],
        verbose_feature_names_out=False,
    )
    model = XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_lambda=2.0,
        eval_metric="logloss",
        random_state=seed,
        n_jobs=-1,
    )
    return Pipeline([("encode", encoder), ("model", model)])


def encoded_feature_names(pipeline: Pipeline) -> list[str]:
    """Post-encoding feature names, aligned with SHAP value columns."""
    return list(pipeline.named_steps["encode"].get_feature_names_out())
