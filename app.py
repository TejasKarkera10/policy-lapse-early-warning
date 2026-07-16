"""Streamlit app: portfolio lapse-risk overview + per-policy review desk.

Run with:  streamlit run app.py   (after `python -m lapse.train`)
"""

import json
import sys
from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from lapse.config import ARTIFACTS_DIR, TARGET, TIER_HIGH, TIER_WATCH
from lapse.explain import global_importance, risk_tier, shap_values, top_drivers
from lapse.rag import RetentionAdvisor

# Chart tokens (light mode) — series colors from the validated palette,
# text/grid from the ink roles. Identity is never carried by color alone.
BLUE = "#2a78d6"   # categorical slot 1 / sequential hue
RED = "#e34948"    # diverging warm pole (risk-increasing)
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"

st.set_page_config(page_title="Lapse Early-Warning", page_icon="📉", layout="wide")


def _axis(**kw):
    return alt.Axis(
        labelColor=MUTED, titleColor=INK_2, gridColor=GRID,
        domainColor="#c3c2b7", tickColor="#c3c2b7", **kw,
    )


@st.cache_resource
def load_artifacts():
    if not (ARTIFACTS_DIR / "pipeline.joblib").exists():
        # First launch (e.g. a fresh Streamlit Cloud deploy): download the
        # dataset and train in place.
        from lapse.train import train as run_training

        with st.spinner("First run: downloading data and training the model (~1 min)…"):
            run_training()
    pipeline = joblib.load(ARTIFACTS_DIR / "pipeline.joblib")
    metrics = json.loads((ARTIFACTS_DIR / "metrics.json").read_text())
    book = pd.read_parquet(ARTIFACTS_DIR / "scored_book.parquet")
    return pipeline, metrics, book


@st.cache_resource
def load_advisor():
    return RetentionAdvisor()


@st.cache_data
def book_shap(_pipeline, book: pd.DataFrame) -> pd.DataFrame:
    return shap_values(_pipeline, book)


pipeline, metrics, book = load_artifacts()
shap_df = book_shap(pipeline, book)
advisor = load_advisor()

st.title("Policy Lapse Early-Warning System")
st.markdown(
    "Finds the policies most likely to **lapse (stop being paid) in the next 24 "
    "months**, explains *why* each one is at risk, and recommends a retention "
    "action a human agent can take — before the policy is lost."
)

s1, s2, s3 = st.columns(3)
s1.markdown(
    "**1 · Predict** \nAn **XGBoost** model scores every policy; **Platt "
    "calibration** turns scores into true probabilities you can budget against."
)
s2.markdown(
    "**2 · Explain** \n**SHAP** breaks each score into its drivers — e.g. "
    "*premium is 13% of this client's income* — so the score is never a black box."
)
s3.markdown(
    "**3 · Act (RAG + LLM)** \nThe top risk drivers become a search query over a "
    "**retention playbook**; an **LLM** (Claude if a key is set, else a "
    "deterministic template) composes a **cited** recommendation. A human decides."
)
st.caption(
    "Stack: Python · pandas · scikit-learn · XGBoost · SHAP · TF-IDF retrieval "
    "(RAG) · Anthropic Claude (optional) · Streamlit · pytest — data: Kaggle life-"
    "insurance retention dataset; outcomes **simulated** (no labels ship with it), "
    "see README."
)

tab_portfolio, tab_policy = st.tabs(["📊 Portfolio", "🔎 Policy review desk"])

# ---------------------------------------------------------------- portfolio
with tab_portfolio:
    m = metrics["test"]
    tiers = book["lapse_prob"].map(risk_tier)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Policies in force", f"{metrics['n_policies']:,}")
    c2.metric("24-mo lapse rate", f"{metrics['base_lapse_rate']:.1%}")
    c3.metric("Test ROC-AUC", f"{m['roc_auc']:.3f}")
    c4.metric("Top-decile lift", f"{m['top_decile_lift']}×")
    c5.metric("High-tier policies", f"{(tiers.str.startswith('High')).sum():,}")

    left, right = st.columns(2)

    with left:
        st.subheader("Where the book stands: policies by risk tier")
        tier_order = [
            "High — agent outreach",
            "Watch — automated touchpoint",
            "Stable — no action",
        ]
        tier_df = (
            tiers.value_counts().reindex(tier_order).fillna(0).astype(int)
            .rename_axis("tier").reset_index(name="policies")
        )
        tier_df["share"] = tier_df["policies"] / tier_df["policies"].sum()
        tier_df["label"] = tier_df.apply(
            lambda r: f"{r.policies:,} ({r.share:.0%})", axis=1
        )
        # Status colors (state, not series): critical / warning / good.
        tier_colors = ["#d03b3b", "#fab219", "#0ca30c"]
        tier_bars = (
            alt.Chart(tier_df)
            .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4, height=36)
            .encode(
                x=alt.X("policies:Q", title="Policies", axis=_axis(),
                        scale=alt.Scale(
                            domainMax=float(tier_df["policies"].max()) * 1.25)),
                y=alt.Y("tier:N", sort=tier_order, title=None,
                        axis=_axis(labelLimit=260)),
                color=alt.Color("tier:N", scale=alt.Scale(domain=tier_order,
                                                          range=tier_colors),
                                legend=None),
                tooltip=[alt.Tooltip("tier:N", title="Tier"),
                         alt.Tooltip("policies:Q", title="Policies", format=","),
                         alt.Tooltip("share:Q", title="Share of book", format=".1%")],
            )
        )
        tier_text = (
            alt.Chart(tier_df)
            .mark_text(align="left", dx=6, color=INK_2, fontSize=12)
            .encode(x="policies:Q", y=alt.Y("tier:N", sort=tier_order), text="label:N")
        )
        st.altair_chart(
            (tier_bars + tier_text).properties(height=220), width="stretch"
        )
        st.caption(
            f"Tiers cut on the calibrated lapse probability: Watch ≥ {TIER_WATCH:.0%}, "
            f"High ≥ {TIER_HIGH:.0%}. The retention team works the red bar first."
        )

    with right:
        st.subheader("Does the model actually work? (never-seen policies)")
        test = book[book["split"] == "test"].copy()
        # Rank first so qcut never hits duplicate bin edges if calibrated
        # scores tie.
        ranks = test["lapse_prob"].rank(method="first")
        test["decile"] = pd.qcut(ranks, 10, labels=False) + 1
        by_decile = (
            test.groupby("decile", as_index=False)
            .agg(lapse_rate=(TARGET, "mean"), policies=(TARGET, "size"))
        )
        base_rate = float(test[TARGET].mean())
        decile_bars = (
            alt.Chart(by_decile)
            .mark_bar(color=BLUE, cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                x=alt.X("decile:O",
                        title="Policies sorted into 10 groups by model score →",
                        axis=_axis(labelAngle=0)),
                y=alt.Y("lapse_rate:Q", title="How many actually lapsed",
                        axis=_axis(format=".0%"),
                        scale=alt.Scale(
                            domainMax=float(by_decile["lapse_rate"].max()) * 1.12)),
                tooltip=[alt.Tooltip("decile:O", title="Risk group (10 = riskiest)"),
                         alt.Tooltip("lapse_rate:Q", title="Actual lapse rate",
                                     format=".1%"),
                         alt.Tooltip("policies:Q", title="Policies")],
            )
        )
        avg_df = pd.DataFrame({"y": [base_rate], "label": [f"book average {base_rate:.0%}"]})
        avg_rule = (
            alt.Chart(avg_df).mark_rule(color=MUTED, strokeDash=[4, 3]).encode(y="y:Q")
        )
        avg_label = (
            alt.Chart(avg_df)
            .mark_text(align="left", baseline="bottom", dx=4, dy=-2,
                       color=INK_2, fontSize=11)
            .encode(y="y:Q", x=alt.value(4), text="label:N")
        )
        st.altair_chart(
            (decile_bars + avg_rule + avg_label).properties(height=300),
            width="stretch",
        )
        st.caption(
            "Bars rise left to right: the higher the model scored a policy, the "
            "more often it really lapsed. The riskiest group lapses at ~2.7× the "
            "book average — that's where outreach pays for itself."
        )

    st.subheader("What drives lapse risk across the book")
    imp = global_importance(shap_df, top_k=8).reset_index()
    imp.columns = ["driver", "mean_abs_shap"]
    imp["relative"] = imp["mean_abs_shap"] / imp["mean_abs_shap"].sum()
    imp_chart = (
        alt.Chart(imp)
        .mark_bar(color=BLUE, cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
        .encode(
            x=alt.X("relative:Q", title="Share of the model's attention",
                    axis=_axis(format=".0%")),
            y=alt.Y("driver:N", sort="-x", title=None, axis=_axis(labelLimit=260)),
            tooltip=[alt.Tooltip("driver:N", title="Driver"),
                     alt.Tooltip("relative:Q", title="Relative importance",
                                 format=".1%"),
                     alt.Tooltip("mean_abs_shap:Q", title="Mean |SHAP| (log-odds)",
                                 format=".3f")],
        )
        .properties(height=280)
    )
    st.altair_chart(imp_chart, width="stretch")
    st.caption(
        "Top 8 drivers, sized by their share of total SHAP importance. "
        "Affordability (premium vs. income) and age dominate — exactly the levers "
        "the retention playbook has plays for."
    )

    with st.expander("View data as table"):
        st.dataframe(imp, hide_index=True)

# ------------------------------------------------------------- policy desk
with tab_policy:
    st.subheader("Highest-risk policies")
    worklist = book.nlargest(50, "lapse_prob").copy()
    worklist["tier"] = worklist["lapse_prob"].map(risk_tier)

    chosen_id = st.selectbox(
        "Select a policy from the outreach worklist (top 50 by risk)",
        worklist["customer_id"],
        format_func=lambda cid: (
            f"{cid[:13]}…  ·  "
            f"{worklist.set_index('customer_id').loc[cid, 'lapse_prob']:.0%} risk  ·  "
            f"{worklist.set_index('customer_id').loc[cid, 'policy_type']}"
        ),
    )
    row = book[book["customer_id"] == chosen_id].iloc[0]
    row_shap = shap_df.loc[book["customer_id"] == chosen_id].iloc[0]

    info, chart_col = st.columns([1, 2])
    with info:
        prob = row["lapse_prob"]
        tier = risk_tier(prob)
        icon = "🔴" if tier.startswith("High") else "🟡" if tier.startswith("Watch") else "🟢"
        st.metric("Calibrated lapse probability", f"{prob:.0%}")
        st.markdown(f"**Tier:** {icon} {tier}")
        st.markdown(
            f"""
| | |
|---|---|
| Age | {row.age} |
| Policy | {row.policy_type} |
| Coverage | ${row.coverage_amount:,.0f} |
| Premium | ${row.monthly_premium:,.2f}/mo |
| Premium burden | {row.premium_burden_pct:.1f}% of income |
| Smoking | {row.smoking_status} |
| Marital / deps | {row.marital_status} / {row.number_of_dependents} |
"""
        )

    with chart_col:
        st.markdown("**Why the model scored this policy** (SHAP)")
        contrib = row_shap[row_shap.abs() > 0.01].sort_values()
        from lapse.explain import humanize

        contrib_df = pd.DataFrame(
            {
                "driver": [humanize(f) for f in contrib.index],
                "shap": contrib.values,
                "direction": ["Increases risk" if v > 0 else "Decreases risk"
                              for v in contrib.values],
            }
        )
        driver_chart = (
            alt.Chart(contrib_df)
            .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
            .encode(
                x=alt.X("shap:Q", title="← pushes risk down · pushes risk up →",
                        axis=_axis()),
                y=alt.Y("driver:N", sort=alt.EncodingSortField("shap", order="descending"),
                        title=None, axis=_axis(labelLimit=240)),
                color=alt.Color(
                    "direction:N",
                    scale=alt.Scale(domain=["Increases risk", "Decreases risk"],
                                    range=[RED, BLUE]),
                    legend=alt.Legend(title=None, orient="top", labelColor=INK_2),
                ),
                tooltip=[alt.Tooltip("driver:N", title="Driver"),
                         alt.Tooltip("shap:Q", title="SHAP", format="+.3f"),
                         alt.Tooltip("direction:N", title="Effect")],
            )
            .properties(height=max(180, 26 * len(contrib_df)))
        )
        st.altair_chart(driver_chart, width="stretch")

    st.divider()
    st.subheader("Recommended retention play")
    drivers = [d["label"] for d in top_drivers(row_shap)]
    context = (
        f"{row.policy_type} policy, age {row.age}, premium burden "
        f"{row.premium_burden_pct:.1f}% of income, {row.smoking_status.lower()}, "
        f"{row.marital_status.lower()}, {row.number_of_dependents} dependents"
    )
    result = advisor.recommend(context=context, drivers=drivers)
    st.markdown(result["recommendation"])
    st.caption(f"Composed by: {result['llm']} · grounded in the retention playbook")

    with st.expander("Retrieved playbook sections (RAG citations)"):
        for source in result["sources"]:
            st.markdown(f"#### {source['title']}  \n*{source['file']}*\n\n{source['text']}")

    st.info(
        "**Human-in-the-loop:** scores route work, people make the call. High-tier "
        "policies go to the assigned agent (contact within 5 business days); any "
        "mention of financial hardship escalates to a licensed specialist.",
        icon="🧑‍💼",
    )
