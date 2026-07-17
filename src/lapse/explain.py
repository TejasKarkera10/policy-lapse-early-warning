"""SHAP explanations for the fitted lapse pipeline."""

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from lapse.features import encoded_feature_names, feature_frame

# Encoded feature name -> plain-English driver label used in the app and
# in RAG queries.
_DRIVER_LABELS = {
    "premium_burden_pct": "Premium burden relative to income",
    "coverage_income_ratio": "Coverage size relative to income",
    "monthly_premium": "Monthly premium amount",
    "annual_income": "Annual income",
    "coverage_amount": "Coverage amount",
    "age": "Policyholder age",
    "number_of_dependents": "Number of dependents",
}


def humanize(feature: str) -> str:
    """Turn an encoded feature name into a readable driver label."""
    if feature in _DRIVER_LABELS:
        return _DRIVER_LABELS[feature]
    for col in (
        "policy_type",
        "smoking_status",
        "marital_status",
        "health_status",
        "occupation_sector",
        "gender",
    ):
        prefix = f"{col}_"
        if feature.startswith(prefix):
            value = feature.removeprefix(prefix)
            return f"{col.replace('_', ' ').capitalize()}: {value}"
    return feature.replace("_", " ").capitalize()


def shap_values(pipeline: Pipeline, df: pd.DataFrame) -> pd.DataFrame:
    """Per-policy SHAP values (log-odds space), encoded-feature columns."""
    X = pipeline.named_steps["encode"].transform(feature_frame(df))
    explainer = shap.TreeExplainer(pipeline.named_steps["model"])
    values = explainer.shap_values(X)
    return pd.DataFrame(values, columns=encoded_feature_names(pipeline), index=df.index)


def global_importance(shap_df: pd.DataFrame, top_k: int = 12) -> pd.Series:
    """Mean |SHAP| per feature, descending - the global driver ranking."""
    imp = shap_df.abs().mean().sort_values(ascending=False).head(top_k)
    imp.index = [humanize(f) for f in imp.index]
    return imp


def top_drivers(shap_row: pd.Series, top_k: int = 4) -> list[dict]:
    """The strongest risk-increasing drivers for one policy.

    Returns dicts of {feature, label, shap} sorted by SHAP descending,
    keeping only features that push risk *up* - those are what a
    retention play can act on.
    """
    pushing_up = shap_row[shap_row > 0].sort_values(ascending=False).head(top_k)
    return [
        {"feature": f, "label": humanize(f), "shap": round(float(v), 4)}
        for f, v in pushing_up.items()
    ]


def risk_tier(prob: float, watch: float = 0.20, high: float = 0.35) -> str:
    """Route policies by calibrated probability (human-in-the-loop tiers)."""
    if prob >= high:
        return "High - agent outreach"
    if prob >= watch:
        return "Watch - automated touchpoint"
    return "Stable - no action"


def expected_saves(probs: np.ndarray, contact_rate: float = 0.30) -> float:
    """Naive expected lapses averted if top-tier outreach converts 30%."""
    return float(np.sort(probs)[-len(probs) // 10 :].sum() * contact_rate)
