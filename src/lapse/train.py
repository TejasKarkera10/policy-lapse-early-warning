"""Train, calibrate, and evaluate the lapse model; persist artifacts.

Run with:  python -m lapse.train

Artifacts written to `artifacts/`:
- pipeline.joblib     fitted encoder + XGBoost pipeline (raw scores, used for SHAP)
- calibrator.joblib   Platt (sigmoid) map from raw score -> calibrated probability
- metrics.json        held-out test metrics + metadata
- scored_book.parquet full cleaned book with calibrated lapse probabilities
"""

import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from lapse.config import ARTIFACTS_DIR, SEED, TARGET
from lapse.data import load_clean
from lapse.features import build_pipeline, feature_frame
from lapse.labels import simulate_lapse


class PlattCalibrator:
    """Sigmoid (Platt) calibration on the model's raw score.

    Fits a logistic map on the raw score's log-odds. Chosen over isotonic
    here because it is smooth: no tied output scores and no saturation at
    exactly 0 or 1, which matters when probabilities are shown to people.
    """

    def __init__(self):
        self._lr = LogisticRegression(C=1e6)

    def fit(self, raw_prob: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        self._lr.fit(_logit(raw_prob).reshape(-1, 1), y)
        return self

    def predict(self, raw_prob: np.ndarray) -> np.ndarray:
        return self._lr.predict_proba(_logit(raw_prob).reshape(-1, 1))[:, 1]


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def top_decile_lift(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Lapse rate in the riskiest decile relative to the base rate."""
    n_top = max(1, len(y_prob) // 10)
    top_idx = np.argsort(y_prob)[-n_top:]
    return float(y_true[top_idx].mean() / y_true.mean())


def train(df: pd.DataFrame | None = None, seed: int = SEED) -> dict:
    """Full training run; returns the metrics dict."""
    if df is None:
        df = load_clean()
    df = df.copy()
    df[TARGET] = simulate_lapse(df, seed=seed)

    # 60/20/20 train / calibration / test split, stratified on the label.
    train_df, rest = train_test_split(
        df, test_size=0.4, stratify=df[TARGET], random_state=seed
    )
    calib_df, test_df = train_test_split(
        rest, test_size=0.5, stratify=rest[TARGET], random_state=seed
    )

    pipeline = build_pipeline(seed=seed)
    pipeline.fit(feature_frame(train_df), train_df[TARGET])

    # Calibration on a held-out slice: XGBoost scores rank well but are
    # not probabilities a retention team can budget against.
    calib_raw = pipeline.predict_proba(feature_frame(calib_df))[:, 1]
    calibrator = PlattCalibrator().fit(calib_raw, calib_df[TARGET].to_numpy())

    test_raw = pipeline.predict_proba(feature_frame(test_df))[:, 1]
    test_prob = calibrator.predict(test_raw)
    y_test = test_df[TARGET].to_numpy()

    metrics = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_policies": int(len(df)),
        "base_lapse_rate": round(float(df[TARGET].mean()), 4),
        "test": {
            "roc_auc": round(float(roc_auc_score(y_test, test_prob)), 4),
            "pr_auc": round(float(average_precision_score(y_test, test_prob)), 4),
            "brier": round(float(brier_score_loss(y_test, test_prob)), 4),
            "top_decile_lift": round(top_decile_lift(y_test, test_prob), 2),
        },
        "seed": seed,
    }

    # Score the whole book for the app. Policies the model trained on are
    # flagged so the demo can be read honestly.
    book = df.copy()
    book["lapse_prob"] = calibrator.predict(
        pipeline.predict_proba(feature_frame(book))[:, 1]
    )
    book["split"] = "train"
    book.loc[calib_df.index, "split"] = "calibration"
    book.loc[test_df.index, "split"] = "test"

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline, ARTIFACTS_DIR / "pipeline.joblib")
    joblib.dump(calibrator, ARTIFACTS_DIR / "calibrator.joblib")
    (ARTIFACTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    book.to_parquet(ARTIFACTS_DIR / "scored_book.parquet", index=False)

    return metrics


if __name__ == "__main__":
    m = train()
    print(json.dumps(m, indent=2))
