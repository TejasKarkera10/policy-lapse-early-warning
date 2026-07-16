"""Load and clean the Kaggle life-insurance retention dataset."""

import pandas as pd

from lapse.config import KAGGLE_DATASET

# Keyword → sector mapping for the free-text occupation field. The raw
# column has ~6,000 distinct values including typos ("Elemtrical
# Engineer"), so we bucket by keyword rather than exact match. Order
# matters: first hit wins.
_SECTOR_KEYWORDS = [
    ("engineer", "Engineering"),
    ("developer", "Technology"),
    ("software", "Technology"),
    ("data", "Technology"),
    ("analyst", "Finance & Analytics"),
    ("financ", "Finance & Analytics"),
    ("account", "Finance & Analytics"),
    ("teach", "Education"),
    ("professor", "Education"),
    ("school", "Education"),
    ("nurse", "Healthcare"),
    ("doctor", "Healthcare"),
    ("physician", "Healthcare"),
    ("medic", "Healthcare"),
    ("market", "Sales & Marketing"),
    ("sales", "Sales & Marketing"),
    ("manager", "Management"),
    ("director", "Management"),
    ("executive", "Management"),
    ("hr ", "Operations & Admin"),
    ("human resource", "Operations & Admin"),
    ("operations", "Operations & Admin"),
    ("admin", "Operations & Admin"),
    ("design", "Creative"),
    ("artist", "Creative"),
    ("writer", "Creative"),
    ("retired", "Retired"),
]


def occupation_sector(occupation: str) -> str:
    """Map a free-text occupation to a coarse sector bucket."""
    occ = str(occupation).lower()
    for keyword, sector in _SECTOR_KEYWORDS:
        if keyword in occ:
            return sector
    return "Other"


def load_raw() -> pd.DataFrame:
    """Download (cached) the dataset from Kaggle and return it raw."""
    import glob
    import os

    import kagglehub

    path = kagglehub.dataset_download(KAGGLE_DATASET)
    csvs = glob.glob(os.path.join(path, "*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV found in Kaggle download at {path}")
    return pd.read_csv(csvs[0])


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw dataset.

    - Drops the synthetic `name` column (PII-shaped, no signal).
    - Replaces the `_RARE_` placeholder values with "Unknown".
    - Parses the policy start date.
    - Buckets free-text occupations into sectors.
    - Adds affordability ratios (the strongest known lapse drivers).
    """
    df = df.copy()
    df = df.drop(columns=["name"], errors="ignore")

    for col in ["gender", "health_status", "marital_status", "smoking_status"]:
        df[col] = df[col].replace("_RARE_", "Unknown")

    df["policy_start_date"] = pd.to_datetime(df["policy_start_date"])
    df["occupation_sector"] = df["occupation"].map(occupation_sector)

    df["premium_burden_pct"] = 100 * (12 * df["monthly_premium"]) / df["annual_income"]
    df["coverage_income_ratio"] = df["coverage_amount"] / df["annual_income"]

    return df


def load_clean() -> pd.DataFrame:
    """Convenience: download + clean in one call."""
    return clean(load_raw())
