from lapse.rag import RetentionAdvisor, Retriever, TemplateLLM


def test_retriever_finds_affordability_section():
    chunks = Retriever().retrieve("premium burden relative to income affordability", k=2)
    assert chunks
    assert any("affordability" in c.title.lower() for c in chunks)


def test_advisor_recommendation_is_grounded_and_cited():
    advisor = RetentionAdvisor(llm=TemplateLLM())
    out = advisor.recommend(
        context="Term Life policy, age 28, premium burden 4.1% of income",
        drivers=["Premium burden relative to income", "Policy type: Term Life"],
    )
    assert out["sources"], "must cite at least one playbook section"
    for source in out["sources"]:
        assert source["title"] in out["recommendation"] or source["text"]
    assert "human" in out["recommendation"].lower() or "agent" in out["recommendation"].lower()


def test_retriever_returns_nothing_for_gibberish():
    assert Retriever().retrieve("zzzz qqqq xxxx", k=2) == []
