"""Find contract clauses by meaning, with embeddings.

Keyword search fails when a buyer's question shares no words with the clause
("Do we get anything for paying faster?" against "a discount of 1% applies to
any invoice settled within 10 days"). The vocabulary map in `search.py` fixes
the phrasings someone wrote down; embeddings handle the ones nobody did.

  EmbeddingIndex  cosine similarity between the question and every clause
  HybridIndex     reciprocal-rank fusion of the keyword index and the
                  embedding index, so each covers the other's misses

Embeddings come from the Gemini API (`gemini-embedding-001` by default, free
tier is enough) and are cached on disk by text, so each clause is embedded
once. On BigQuery the same vectors are searched with VECTOR_SEARCH, see
`copilot.eval.retrieval --bigquery`.

  python -m copilot.contracts.embed models    list embedding models for your key
  python -m copilot.contracts.embed build     embed every clause (about 10 API calls)
"""

from __future__ import annotations

import argparse
import hashlib
import os
import time
from functools import lru_cache

import numpy as np
import pandas as pd

from copilot import config
from copilot.contracts.search import ContractIndex

DEFAULT_MODEL = "gemini-embedding-001"
DIMENSIONS = 768
BATCH = 100
CACHE = config.DATA / "embeddings"


def _key(task: str, text: str) -> str:
    return hashlib.sha1(f"{task}\n{text}".encode()).hexdigest()


def _unit(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(norms == 0, 1.0, norms)


class GeminiEmbedder:
    """Gemini embeddings, cached in data/embeddings/<model>.parquet."""

    def __init__(self, model: str | None = None, dimensions: int = DIMENSIONS):
        from google import genai
        from google.genai import types

        self.name = model or os.environ.get("COPILOT_EMBED_MODEL", DEFAULT_MODEL)
        self.dimensions = dimensions
        self.client = genai.Client(
            api_key=os.environ.get("GEMINI_API_KEY") or os.environ["GOOGLE_API_KEY"],
            http_options=types.HttpOptions(timeout=60_000))
        self.calls = 0
        self.path = CACHE / f"{self.name}-{dimensions}.parquet"
        self._cache = {}
        if self.path.exists():
            cached = pd.read_parquet(self.path)
            self._cache = dict(zip(cached["key"], cached["vector"], strict=True))

    def _call(self, texts: list[str], task: str) -> list[list[float]]:
        from google.genai import types

        config_ = types.EmbedContentConfig(task_type=task,
                                           output_dimensionality=self.dimensions)
        for attempt in range(4):
            try:
                self.calls += 1
                result = self.client.models.embed_content(model=self.name, contents=texts,
                                                          config=config_)
                return [e.values for e in result.embeddings]
            except Exception as exc:
                text = str(exc)
                if attempt == 3 or not ("429" in text or "RESOURCE_EXHAUSTED" in text
                                        or "503" in text or "timeout" in text.lower()):
                    raise
                time.sleep(20 * (attempt + 1))
        raise RuntimeError("unreachable")

    def embed(self, texts: list[str], task: str) -> np.ndarray:
        """task: RETRIEVAL_DOCUMENT for clauses, RETRIEVAL_QUERY for questions."""
        keys = [_key(task, t) for t in texts]
        missing = list(dict.fromkeys(t for t, k in zip(texts, keys, strict=True)
                                     if k not in self._cache))
        for i in range(0, len(missing), BATCH):
            batch = missing[i:i + BATCH]
            for text, vector in zip(batch, self._call(batch, task), strict=True):
                self._cache[_key(task, text)] = list(vector)
        if missing:
            self.save()
        return _unit(np.array([self._cache[k] for k in keys], dtype=float))

    def save(self) -> None:
        CACHE.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"key": list(self._cache), "vector": list(self._cache.values())}
                     ).to_parquet(self.path, index=False)


class HashingEmbedder:
    """Offline stand-in with the same interface, for tests. It knows no meaning:
    it is a bag of words, so it only proves the plumbing works."""

    name = "hashing"
    dimensions = 256
    calls = 0

    def embed(self, texts: list[str], task: str) -> np.ndarray:
        from sklearn.feature_extraction.text import HashingVectorizer

        vec = HashingVectorizer(n_features=self.dimensions, alternate_sign=False)
        return _unit(vec.transform(texts).toarray())


def clause_text(chunks: pd.DataFrame) -> list[str]:
    return (chunks["title"] + ". " + chunks["text"]).tolist()


def _results(chunks: pd.DataFrame, scores: np.ndarray, vendor_id, k) -> list[dict]:
    ranked = chunks.assign(score=scores)
    if vendor_id:
        ranked = ranked[ranked["vendor_id"] == vendor_id]
    ranked = ranked.sort_values("score", ascending=False, kind="stable").head(k)
    return [{"citation": f"{r.contract_id} clause {r.clause_no}", "contract_id": r.contract_id,
             "vendor_id": r.vendor_id, "clause_no": r.clause_no, "title": r.title,
             "text": r.text, "score": round(float(r.score), 4)} for r in ranked.itertuples()]


class EmbeddingIndex:
    def __init__(self, chunks: pd.DataFrame, embedder):
        self.chunks = chunks.reset_index(drop=True)
        self.embedder = embedder
        self.matrix = embedder.embed(clause_text(self.chunks), "RETRIEVAL_DOCUMENT")

    def scores(self, question: str) -> np.ndarray:
        q = self.embedder.embed([question], "RETRIEVAL_QUERY")[0]
        return self.matrix @ q

    def search(self, question: str, vendor_id: str | None = None, k: int = 3) -> list[dict]:
        return _results(self.chunks, self.scores(question), vendor_id, k)


class HybridIndex:
    """Reciprocal-rank fusion: score = sum over both rankings of 1 / (60 + rank)."""

    def __init__(self, chunks: pd.DataFrame, embedder, keyword: ContractIndex | None = None):
        self.chunks = chunks.reset_index(drop=True)
        self.keyword = keyword or ContractIndex(self.chunks)
        self.semantic = EmbeddingIndex(self.chunks, embedder)

    def search(self, question: str, vendor_id: str | None = None, k: int = 3) -> list[dict]:
        from sklearn.metrics.pairwise import linear_kernel

        mask = (self.chunks["vendor_id"] == vendor_id).to_numpy() if vendor_id else None
        fused = np.zeros(len(self.chunks))
        kw = linear_kernel(self.keyword.vectorizer.transform([question]),
                           self.keyword.matrix).ravel()
        for s in (kw, self.semantic.scores(question)):
            s = np.where(mask, s, -np.inf) if mask is not None else s
            rank = np.empty(len(s))
            rank[np.argsort(-s, kind="stable")] = np.arange(1, len(s) + 1)
            fused += 1.0 / (60 + rank)
        return _results(self.chunks, fused, vendor_id, k)


@lru_cache(maxsize=2)
def load_semantic_index(mode: str = "hybrid"):
    """The index the copilot uses when COPILOT_RETRIEVAL is 'hybrid' or 'embeddings'."""
    if mode not in ("hybrid", "embeddings"):
        raise ValueError(f"COPILOT_RETRIEVAL must be keyword, hybrid or embeddings, not {mode}")
    chunks = pd.read_parquet(config.CURATED / "contract_chunks.parquet")
    embedder = GeminiEmbedder()
    return HybridIndex(chunks, embedder) if mode == "hybrid" else EmbeddingIndex(chunks, embedder)


def list_models() -> list[str]:
    from google import genai

    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY")
                          or os.environ["GOOGLE_API_KEY"])
    return sorted(m.name.removeprefix("models/") for m in client.models.list()
                  if "embedContent" in (getattr(m, "supported_actions", None) or []))


def main() -> None:
    ap = argparse.ArgumentParser(description="Contract clause embeddings")
    ap.add_argument("command", choices=["models", "build"])
    args = ap.parse_args()
    if args.command == "models":
        print("\n".join(list_models()) or "No embedding models available for this key.")
        return
    chunks = pd.read_parquet(config.CURATED / "contract_chunks.parquet")
    embedder = GeminiEmbedder()
    start = time.perf_counter()
    embedder.embed(clause_text(chunks), "RETRIEVAL_DOCUMENT")
    print(f"Embedded {len(chunks)} clauses with {embedder.name} in "
          f"{time.perf_counter() - start:.1f} s ({embedder.calls} API calls). "
          f"Cached in {embedder.path}")


if __name__ == "__main__":
    main()
