import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def synthetic_book() -> pd.DataFrame:
    """A small raw-schema book so tests never hit the network."""
    rng = np.random.default_rng(0)
    n = 2000
    income = rng.integers(42_000, 190_000, n)
    return pd.DataFrame(
        {
            "customer_id": [f"c{i}" for i in range(n)],
            "name": ["Test Person"] * n,
            "age": rng.integers(22, 70, n),
            "gender": rng.choice(["Male", "Female", "_RARE_"], n, p=[0.47, 0.5, 0.03]),
            "marital_status": rng.choice(
                ["Married", "Single", "Divorced", "Widowed"], n, p=[0.6, 0.25, 0.08, 0.07]
            ),
            "number_of_dependents": rng.integers(0, 4, n),
            "annual_income": income,
            "occupation": rng.choice(
                ["Electrical Engineer", "Teacher", "Sales Manager", "Graphic Designer", "Retired"],
                n,
            ),
            "health_status": rng.choice(["Excellent", "Good", "Fair"], n),
            "smoking_status": rng.choice(
                ["Non-smoker", "Former smoker", "Current smoker"], n, p=[0.79, 0.14, 0.07]
            ),
            "policy_type": rng.choice(
                ["Term Life", "Universal Life", "Whole Life", "Variable Life"],
                n,
                p=[0.56, 0.2, 0.19, 0.05],
            ),
            "coverage_amount": rng.integers(100_000, 1_200_000, n),
            "monthly_premium": np.round(rng.uniform(24, 700, n), 2),
            "policy_start_date": "2022-06-15",
        }
    )
