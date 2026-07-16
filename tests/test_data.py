from lapse.data import clean, occupation_sector


def test_clean_drops_pii_and_rare_placeholders(synthetic_book):
    df = clean(synthetic_book)
    assert "name" not in df.columns
    assert "_RARE_" not in set(df["gender"])
    assert "Unknown" in set(df["gender"])


def test_affordability_features(synthetic_book):
    df = clean(synthetic_book)
    row = df.iloc[0]
    expected = 100 * 12 * row.monthly_premium / row.annual_income
    assert abs(row.premium_burden_pct - expected) < 1e-9
    assert (df.premium_burden_pct > 0).all()


def test_occupation_sector_handles_typos_and_unknowns():
    assert occupation_sector("Elemtrical Engineer") == "Engineering"
    assert occupation_sector("Senior Data Analyst") == "Technology"
    assert occupation_sector("Zookeeper") == "Other"
    assert occupation_sector("Retired") == "Retired"
