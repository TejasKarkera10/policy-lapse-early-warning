from lapse.data import clean
from lapse.labels import simulate_lapse


def test_lapse_rate_is_plausible(synthetic_book):
    df = clean(synthetic_book)
    y = simulate_lapse(df)
    assert 0.05 < y.mean() < 0.30


def test_seeded_and_deterministic(synthetic_book):
    df = clean(synthetic_book)
    assert simulate_lapse(df, seed=7).equals(simulate_lapse(df, seed=7))
    assert not simulate_lapse(df, seed=7).equals(simulate_lapse(df, seed=8))


def test_premium_burden_raises_lapse_rate(synthetic_book):
    df = clean(synthetic_book)
    y = simulate_lapse(df)
    high = y[df.premium_burden_pct > df.premium_burden_pct.quantile(0.8)]
    low = y[df.premium_burden_pct < df.premium_burden_pct.quantile(0.2)]
    assert high.mean() > low.mean()
