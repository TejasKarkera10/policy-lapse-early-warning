"""Central configuration: paths, seeds, and modeling constants."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge"

KAGGLE_DATASET = "ayushyajnik/life-insurance-retention-dataset"

SEED = 42

# Observation design: every policy in the dataset was issued in 2022.
# The target is "policy lapsed within 24 months of issue", so the
# observation window is fully closed for the whole book.
LABEL_WINDOW_MONTHS = 24

TARGET = "lapsed_24m"

NUMERIC_FEATURES = [
    "age",
    "number_of_dependents",
    "annual_income",
    "coverage_amount",
    "monthly_premium",
    "premium_burden_pct",
    "coverage_income_ratio",
]

CATEGORICAL_FEATURES = [
    "gender",
    "marital_status",
    "health_status",
    "smoking_status",
    "policy_type",
    "occupation_sector",
]

# Risk tiers used by the app to route work to a human. Thresholds are on
# the calibrated lapse probability.
TIER_WATCH = 0.20
TIER_HIGH = 0.35
