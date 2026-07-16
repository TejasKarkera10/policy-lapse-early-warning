import numpy as np

from lapse.config import TARGET
from lapse.data import clean
from lapse.explain import risk_tier, shap_values, top_drivers
from lapse.features import build_pipeline, feature_frame
from lapse.labels import simulate_lapse


def test_end_to_end_model_beats_chance(synthetic_book, tmp_path, monkeypatch):
    import lapse.train as train_mod

    monkeypatch.setattr(train_mod, "ARTIFACTS_DIR", tmp_path)
    metrics = train_mod.train(df=clean(synthetic_book))
    assert metrics["test"]["roc_auc"] > 0.60
    assert 0 < metrics["base_lapse_rate"] < 1
    assert (tmp_path / "pipeline.joblib").exists()
    assert (tmp_path / "scored_book.parquet").exists()


def test_shap_shapes_and_drivers(synthetic_book):
    df = clean(synthetic_book)
    df[TARGET] = simulate_lapse(df)
    pipe = build_pipeline()
    pipe.fit(feature_frame(df), df[TARGET])

    sv = shap_values(pipe, df.head(50))
    assert sv.shape[0] == 50
    drivers = top_drivers(sv.iloc[0])
    assert all(d["shap"] > 0 for d in drivers)


def test_risk_tiers_partition_probability_space():
    assert risk_tier(0.05).startswith("Stable")
    assert risk_tier(0.25).startswith("Watch")
    assert risk_tier(0.50).startswith("High")
    assert {risk_tier(p) for p in np.linspace(0, 1, 101)} == {
        "Stable — no action",
        "Watch — automated touchpoint",
        "High — agent outreach",
    }
