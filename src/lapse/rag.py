"""Retrieval-augmented retention recommendations.

Pipeline: the model's top SHAP risk drivers for a policy become the
retrieval query; TF-IDF retrieval pulls the most relevant playbook
sections; an LLM (or a deterministic template when no API key is
configured) composes the recommendation, citing its sources.

The LLM is behind a small protocol so swapping in Claude/Bedrock is a
one-class change - retrieval, prompting, and citation logic are shared.
"""

import os
import re
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from lapse.config import KNOWLEDGE_DIR


@dataclass
class Chunk:
    """One retrievable playbook section."""

    title: str
    text: str
    source: str


def load_chunks(knowledge_dir=KNOWLEDGE_DIR) -> list[Chunk]:
    """Split every markdown file in the knowledge dir on `## ` headings."""
    chunks = []
    for path in sorted(knowledge_dir.glob("*.md")):
        body = path.read_text()
        for match in re.split(r"\n(?=## )", body):
            if not match.startswith("## "):
                continue
            title, _, text = match.partition("\n")
            chunks.append(
                Chunk(title=title.removeprefix("## ").strip(), text=text.strip(), source=path.name)
            )
    return chunks


class Retriever:
    """TF-IDF retrieval over playbook chunks.

    Lexical retrieval is deliberate for a corpus this small: it is
    dependency-free and fully deterministic. The interface (query ->
    ranked chunks) is what an embedding store would also implement.
    """

    def __init__(self, chunks: list[Chunk] | None = None):
        self.chunks = chunks if chunks is not None else load_chunks()
        corpus = [f"{c.title}\n{c.text}" for c in self.chunks]
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform(corpus)

    def retrieve(self, query: str, k: int = 2) -> list[Chunk]:
        q = self.vectorizer.transform([query])
        scores = cosine_similarity(q, self.matrix)[0]
        order = scores.argsort()[::-1][:k]
        return [self.chunks[i] for i in order if scores[i] > 0]


class TemplateLLM:
    """Deterministic fallback composer used when no LLM key is set.

    Produces a grounded recommendation strictly from retrieved chunks, so
    the demo runs with zero secrets and never hallucinates.
    """

    name = "template (no API key - deterministic composition)"

    def compose(self, context: str, drivers: list[str], chunks: list[Chunk]) -> str:
        lines = [
            f"**Why this policy is flagged:** {', '.join(d.lower() for d in drivers)}.",
            "",
            "**Recommended plays (from the retention playbook):**",
        ]
        for chunk in chunks:
            first_play = _first_list_item(chunk.text)
            lines.append(f"- *{chunk.title}* - {first_play}")
        lines.append("")
        lines.append(
            "**Next step:** route per the escalation rules - a human agent "
            "makes the final call on any offer."
        )
        return "\n".join(lines)


class AnthropicLLM:
    """Live Claude composer, used when ANTHROPIC_API_KEY is set."""

    def __init__(self):
        import anthropic  # optional dependency, only needed with a key

        self.client = anthropic.Anthropic()
        self.name = "claude (live)"

    def compose(self, context: str, drivers: list[str], chunks: list[Chunk]) -> str:
        sources = "\n\n".join(f"[{c.title}]\n{c.text}" for c in chunks)
        msg = self.client.messages.create(
            model="claude-sonnet-5",
            max_tokens=400,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "You are a retention specialist assistant at a life "
                        "insurer. Using ONLY the playbook excerpts below, write "
                        "a short recommendation (<=120 words) for the agent "
                        "handling this policy. Cite section titles in brackets.\n\n"
                        f"Policy context: {context}\n"
                        f"Model risk drivers: {', '.join(drivers)}\n\n"
                        f"Playbook excerpts:\n{sources}"
                    ),
                }
            ],
        )
        return msg.content[0].text


def get_llm():
    """Pick the live LLM if configured, else the deterministic template."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return AnthropicLLM()
        except Exception:
            pass
    return TemplateLLM()


class RetentionAdvisor:
    """Drivers -> retrieval -> composed, cited recommendation."""

    def __init__(self, retriever: Retriever | None = None, llm=None):
        self.retriever = retriever or Retriever()
        self.llm = llm or get_llm()

    def recommend(self, context: str, drivers: list[str], k: int = 2) -> dict:
        query = " ".join(drivers) + " " + context
        chunks = self.retriever.retrieve(query, k=k)
        text = self.llm.compose(context, drivers, chunks)
        return {
            "recommendation": text,
            "sources": [{"title": c.title, "file": c.source, "text": c.text} for c in chunks],
            "llm": self.llm.name,
        }


def _first_list_item(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if re.match(r"^\d+\.\s|^-\s", stripped):
            item = re.sub(r"^\d+\.\s|-\s", "", stripped)
            return re.sub(r"\*\*", "", item)
    return text.splitlines()[0] if text else ""
