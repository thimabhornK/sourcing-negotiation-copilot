"""Find the contract clause that answers a question.

Contracts are split into clauses and indexed with TF-IDF, which runs anywhere
with no model download. On Google Cloud the same interface is backed by
embeddings and vector search. Every result carries a citation (contract and
clause number) so an answer can always be traced to its source.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from copilot import config

CLAUSE = re.compile(r"^## (\d+)\. (.+?)\n\n(.+?)(?=\n## |\Z)", re.S | re.M)
VENDOR = re.compile(r"vendor (V-\d{4})")


def chunk_contract(path: Path) -> list[dict]:
    """One row per numbered clause."""
    text = path.read_text()
    vendor = VENDOR.search(text)
    return [{"contract_id": path.stem, "vendor_id": vendor.group(1) if vendor else None,
             "clause_no": int(no), "title": title.strip(), "text": " ".join(body.split())}
            for no, title, body in CLAUSE.findall(text)]


def build_chunks(folder: Path) -> pd.DataFrame:
    rows = [row for path in sorted(folder.glob("*.md")) for row in chunk_contract(path)]
    return pd.DataFrame(rows)


# Buyers and contracts use different words for the same thing. Each word on the left
# is replaced by the word on the right in both the clauses and the question.
VOCABULARY = {
    "deliver": "delivery", "delivers": "delivery", "delivered": "delivery",
    "deliveries": "delivery", "shipment": "delivery", "shipments": "delivery",
    "delay": "late", "delayed": "late", "delays": "late", "overdue": "late",
    "penalty": "damages", "penalties": "damages", "compensation": "damages",
    "pay": "payment", "pays": "payment", "paid": "payment", "settle": "payment",
    "settled": "payment", "invoices": "invoice", "bill": "invoice", "bills": "invoice",
    "cancel": "terminate", "termination": "terminate", "exit": "terminate",
    "defects": "defect", "defective": "defect", "faulty": "defect", "guarantee": "warranty",
    "warrants": "warranty", "guaranteed": "warranty",
    "refund": "rebate", "rebates": "rebate", "cashback": "rebate", "bonus": "rebate",
    "prices": "price", "pricing": "price", "adjust": "adjustment", "adjusted": "adjustment",
    "renegotiate": "review", "reviewed": "review", "purchases": "spend", "buy": "spend",
    "buying": "spend", "days": "day", "months": "month", "weeks": "week",
}
STOP = {"the", "a", "an", "of", "to", "in", "is", "are", "be", "do", "does", "we", "us", "our",
        "they", "them", "their", "this", "that", "and", "or", "for", "on", "by", "with", "if",
        "what", "how", "can", "it", "any", "each", "shall", "may", "from", "at", "as", "have",
        "has", "there", "under", "will", "would"}


def normalise(text: str) -> list[str]:
    words = re.findall(r"[a-z]+|\d+", text.lower())
    return [VOCABULARY.get(w, w) for w in words if w not in STOP]


class ContractIndex:
    """TF-IDF over clauses. normalise_vocabulary=False gives the plain baseline that
    the evaluation compares against."""

    def __init__(self, chunks: pd.DataFrame, normalise_vocabulary: bool = True):
        self.chunks = chunks.reset_index(drop=True)
        if normalise_vocabulary:
            self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), tokenizer=normalise,
                                              token_pattern=None, lowercase=False,
                                              sublinear_tf=True)
        else:
            self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english",
                                              sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform(self.chunks["title"] + ". "
                                                    + self.chunks["text"])

    def search(self, question: str, vendor_id: str | None = None, k: int = 3) -> list[dict]:
        scores = linear_kernel(self.vectorizer.transform([question]), self.matrix).ravel()
        ranked = self.chunks.assign(score=scores)
        if vendor_id:
            ranked = ranked[ranked["vendor_id"] == vendor_id]
        ranked = ranked.sort_values("score", ascending=False).head(k)
        return [{"citation": f"{r.contract_id} clause {r.clause_no}", "contract_id": r.contract_id,
                 "vendor_id": r.vendor_id, "clause_no": r.clause_no, "title": r.title,
                 "text": r.text, "score": round(float(r.score), 4)}
                for r in ranked.itertuples()]


@lru_cache(maxsize=4)
def _cached_index(path: str, mtime: float) -> ContractIndex:
    return ContractIndex(pd.read_parquet(path))


def load_index() -> ContractIndex:
    path = config.CURATED / "contract_chunks.parquet"
    return _cached_index(str(path), path.stat().st_mtime)
