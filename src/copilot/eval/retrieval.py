"""Compare four ways of finding the contract clause that answers a buyer's question.

  python -m copilot.eval.retrieval              keyword methods only (no key needed)
  python -m copilot.eval.retrieval --embeddings add Gemini embeddings and hybrid
  python -m copilot.eval.retrieval --embeddings --bigquery
                                                also run the embedding search in
                                                BigQuery with VECTOR_SEARCH and
                                                check it returns the same clauses

Every vendor has its own contract, and each question is asked about every
vendor, so 6 topics x 80 vendors = 480 questions per set. Two sets are scored:
`tuned` (the questions the vocabulary map was written against) and `held_out`
(new phrasings written afterwards). Results go to reports/retrieval_eval.json.
"""

from __future__ import annotations

import argparse
import json
import time

import pandas as pd

from copilot import config
from copilot.contracts.search import ContractIndex
from copilot.eval.cases import HELD_OUT_QUESTIONS, QUESTIONS

SETS = {"tuned": QUESTIONS, "held_out": HELD_OUT_QUESTIONS}


def question_rows(chunks: pd.DataFrame, questions: dict) -> pd.DataFrame:
    """One row per (vendor, topic); phrasings rotate across vendors."""
    vendors = sorted(chunks["vendor_id"].unique())
    rows = [{"question_id": f"{vendor_id}|{title}", "vendor_id": vendor_id, "title": title,
             "question": variants[i % len(variants)]}
            for i, vendor_id in enumerate(vendors) for title, variants in questions.items()]
    return pd.DataFrame(rows)


def score(index, questions: pd.DataFrame) -> dict:
    top1 = top3 = 0
    misses = []
    start = time.perf_counter()
    for q in questions.itertuples():
        found = [r["title"] for r in index.search(q.question, q.vendor_id, 3)]
        top1 += found[:1] == [q.title]
        top3 += q.title in found
        if found[:1] != [q.title] and q.question not in [m["question"] for m in misses]:
            misses.append({"question": q.question, "expected": q.title, "got": found[0]})
    n = len(questions)
    return {"hit_at_1": round(top1 / n, 4), "hit_at_3": round(top3 / n, 4),
            "ms_per_question": round(1000 * (time.perf_counter() - start) / n, 2),
            "distinct_phrasings_missed": misses}


def run(embedder=None, bigquery: bool = False) -> dict:
    chunks = pd.read_parquet(config.CURATED / "contract_chunks.parquet")
    indexes = {"plain_tfidf": ContractIndex(chunks, normalise_vocabulary=False),
               "with_vocabulary": ContractIndex(chunks)}
    if embedder is not None:
        from copilot.contracts.embed import EmbeddingIndex, HybridIndex

        semantic = EmbeddingIndex(chunks, embedder)
        indexes["embeddings"] = semantic
        indexes["hybrid"] = HybridIndex(chunks, embedder, keyword=indexes["with_vocabulary"])
    report = {"clauses": len(chunks), "embedding_model": getattr(embedder, "name", None),
              "sets": {}}
    for set_name, questions in SETS.items():
        rows = question_rows(chunks, questions)
        report["sets"][set_name] = {"questions": len(rows), **{
            name: score(index, rows) for name, index in indexes.items()}}
        print(f"  {set_name:<9} " + "  ".join(
            f"{name} {report['sets'][set_name][name]['hit_at_1']:.1%}" for name in indexes))
    if bigquery and embedder is not None:
        report["bigquery_vector_search"] = vector_search_parity(chunks, semantic, embedder)
    config.REPORTS.mkdir(parents=True, exist_ok=True)
    (config.REPORTS / "retrieval_eval.json").write_text(json.dumps(report, indent=2),
                                                        encoding="utf-8")
    return report


VECTOR_SQL = """
SELECT query.question_id, base.contract_id, base.clause_no, base.title, distance
FROM VECTOR_SEARCH(
  TABLE `{dataset}.contract_clause_embeddings`, 'embedding',
  TABLE `{dataset}.question_embeddings`, 'embedding',
  top_k => {top_k}, distance_type => 'COSINE')
WHERE base.vendor_id = query.vendor_id
QUALIFY ROW_NUMBER() OVER (PARTITION BY query.question_id
                           ORDER BY distance, base.clause_no) = 1
"""


def vector_search_parity(chunks: pd.DataFrame, semantic, embedder) -> dict:
    """Load the vectors into BigQuery, search there, and compare the top clause
    for every question with the local search."""
    from google.cloud import bigquery

    from copilot.cloud import bigquery as bq

    s = bq.settings()
    client = bq.client()
    dataset = bq.ensure_dataset(s["dataset"])
    clauses = chunks[["contract_id", "vendor_id", "clause_no", "title"]].assign(
        embedding=[list(map(float, v)) for v in semantic.matrix])
    rows = pd.concat([question_rows(chunks, q) for q in SETS.values()]).reset_index(drop=True)
    rows["question_id"] = [f"{i}|{q}" for i, q in enumerate(rows["question_id"])]
    vectors = embedder.embed(rows["question"].tolist(), "RETRIEVAL_QUERY")
    queries = rows[["question_id", "vendor_id", "title"]].assign(
        embedding=[list(map(float, v)) for v in vectors])
    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    for name, frame in (("contract_clause_embeddings", clauses),
                        ("question_embeddings", queries)):
        client.load_table_from_dataframe(frame, f"{dataset}.{name}",
                                         job_config=job_config).result()

    start = time.perf_counter()
    job = client.query(VECTOR_SQL.format(dataset=dataset, top_k=len(chunks)),
                       job_config=bigquery.QueryJobConfig(use_query_cache=False))
    cloud = job.result().to_dataframe(create_bqstorage_client=False)
    seconds = time.perf_counter() - start

    local = [semantic.chunks.assign(s=semantic.matrix @ v)
             .query("vendor_id == @vid").sort_values(["s", "clause_no"],
                                                     ascending=[False, True]).iloc[0]
             for v, vid in zip(vectors, rows["vendor_id"], strict=True)]
    local = pd.DataFrame({"question_id": rows["question_id"],
                          "local_clause": [r["clause_no"] for r in local]})
    both = (local.merge(cloud, on="question_id", how="left")
            .merge(rows[["question_id", "title"]].rename(columns={"title": "expected"}),
                   on="question_id"))
    same = int((both["local_clause"] == both["clause_no"]).sum())
    hit = int((both["title"] == both["expected"]).sum())
    return {"questions": len(rows), "same_top_clause_as_local": same,
            "hit_at_1": round(hit / len(rows), 4), "seconds": round(seconds, 2),
            "bytes_processed": int(job.total_bytes_processed or 0),
            "match": same == len(rows)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare contract retrieval methods")
    ap.add_argument("--embeddings", action="store_true", help="needs GEMINI_API_KEY")
    ap.add_argument("--bigquery", action="store_true", help="needs COPILOT_BQ_PROJECT")
    args = ap.parse_args()
    embedder = None
    if args.embeddings:
        from copilot.contracts.embed import GeminiEmbedder

        embedder = GeminiEmbedder()
    report = run(embedder, bigquery=args.bigquery)
    if "bigquery_vector_search" in report:
        v = report["bigquery_vector_search"]
        print(f"  BigQuery VECTOR_SEARCH: same top clause for {v['same_top_clause_as_local']}"
              f" of {v['questions']} questions, {v['seconds']} s")
    print("Full report: reports/retrieval_eval.json")


if __name__ == "__main__":
    main()
