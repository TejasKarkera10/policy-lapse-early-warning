"""Simulate lapse outcomes for the label-agnostic Kaggle dataset.

The source dataset ships without an outcome column, so we generate one
from an actuarially-inspired logistic process. This is a *simulation*,
documented and seeded, not real experience data — see the README's
"Honest limitations" section. The effect directions follow published
lapse-experience studies (e.g. SOA persistency studies):

- Premium burden (annual premium / income) is the dominant driver of
  voluntary lapse.
- Term policies lapse more than permanent products; whole life has the
  stickiest persistency (cash value creates an incentive to keep paying).
- Younger policyholders lapse more.
- Current smokers and single policyholders lapse somewhat more;
  dependents make coverage stickier.
- A large unobserved-noise term keeps the problem realistically hard
  (life events, income shocks, and competitor offers are not in the
  data), so the model cannot simply invert the formula.
"""

import numpy as np
import pandas as pd

from lapse.config import SEED, TARGET

# Log-odds coefficients for the simulator.
_INTERCEPT = -2.35  # tuned for a ~13% 24-month lapse rate
_COEF = {
    "premium_burden_z": 0.90,
    "age_z": -0.55,  # younger -> higher lapse
    "term_life": 0.45,
    "variable_life": 0.30,
    "whole_life": -0.40,
    "current_smoker": 0.30,
    "single": 0.25,
    "per_dependent": -0.15,
}
_NOISE_SD = 1.0  # unobserved heterogeneity


def simulate_lapse(df: pd.DataFrame, seed: int = SEED) -> pd.Series:
    """Return a seeded binary `lapsed_24m` outcome for each policy.

    Expects a *cleaned* frame (see `lapse.data.clean`) with
    `premium_burden_pct` already computed.
    """
    rng = np.random.default_rng(seed)

    burden_z = _zscore(df["premium_burden_pct"])
    age_z = _zscore(df["age"])

    logit = (
        _INTERCEPT
        + _COEF["premium_burden_z"] * burden_z
        + _COEF["age_z"] * age_z
        + _COEF["term_life"] * (df["policy_type"] == "Term Life")
        + _COEF["variable_life"] * (df["policy_type"] == "Variable Life")
        + _COEF["whole_life"] * (df["policy_type"] == "Whole Life")
        + _COEF["current_smoker"] * (df["smoking_status"] == "Current smoker")
        + _COEF["single"] * (df["marital_status"] == "Single")
        + _COEF["per_dependent"] * df["number_of_dependents"]
        + rng.normal(0, _NOISE_SD, len(df))
    )

    prob = 1 / (1 + np.exp(-logit))
    lapsed = rng.uniform(size=len(df)) < prob
    return pd.Series(lapsed.astype(int), index=df.index, name=TARGET)


def _zscore(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std()
