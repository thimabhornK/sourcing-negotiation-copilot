"""Embedding and hybrid search, and the evaluation that compares them. The real
model needs a key, so these tests use a bag-of-words stand-in with the same interface."""

import numpy as np
import pandas as pd

from copilot import config, tools
from copilot.contracts import embed
from copilot.eval import retrieval
from copilot.eval.cases import HELD_OUT_QUESTIONS, QUESTIONS


def test_held_out_questions_cover_the_same_topics_with_new_wording():
    assert HELD_OUT_QUESTIONS.keys() == QUESTIONS.keys()
    tuned = {q for qs in QUESTIONS.values() for q in qs}
    assert not tuned & {q for qs in HELD_OUT_QUESTIONS.values() for q in qs}


def test_embedding_and_hybrid_search_return_cited_clauses_for_the_vendor(trained):
    chunks = pd.read_parquet(config.CURATED / "contract_chunks.parquet")
    vendor = chunks["vendor_id"].iloc[0]
    for index in (embed.EmbeddingIndex(chunks, embed.HashingEmbedder()),
                  embed.HybridIndex(chunks, embed.HashingEmbedder())):
        found = index.search("warranty on goods against defects", vendor, 3)
        assert len(found) == 3 and {r["vendor_id"] for r in found} == {vendor}
        assert found[0]["title"] == "Warranty" and "clause" in found[0]["citation"]


def test_retrieval_report_scores_every_method_on_both_question_sets(trained):
    report = retrieval.run(embed.HashingEmbedder())
    for name in ("tuned", "held_out"):
        scores = report["sets"][name]
        assert scores["questions"] == 6 * 80 or scores["questions"] % 6 == 0
        for method in ("plain_tfidf", "with_vocabulary", "embeddings", "hybrid"):
            assert 0 <= scores[method]["hit_at_1"] <= scores[method]["hit_at_3"] <= 1


def test_vectors_are_unit_length():
    m = embed.HashingEmbedder().embed(["late delivery penalty", "payment terms"], "x")
    assert np.allclose(np.linalg.norm(m, axis=1), 1.0)


def test_copilot_uses_keyword_search_unless_asked(trained, monkeypatch):
    monkeypatch.delenv("COPILOT_RETRIEVAL", raising=False)
    assert tools._search_index() is tools.load_index()
