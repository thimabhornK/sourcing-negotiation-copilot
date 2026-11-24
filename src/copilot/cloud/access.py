"""Give the copilot's agents their own identity that can read curated data and
nothing else.

Three datasets, three levels of trust:

  <dataset>_raw    raw exports with tax IDs, bank accounts and card numbers in clear
  <dataset>        curated tables, de-identified (tokens instead of identifiers)
  <dataset>_agent  views over the curated tables, without even the tokens

The agents run as a service account that may read `<dataset>_agent` and run
query jobs, and has no access to the other two datasets. The views are
*authorized views*: BigQuery lets the view read the curated tables on the
agent's behalf, so the agent never needs (or gets) access to the tables
themselves. A prompt-injected or buggy agent therefore cannot query raw or
curated data, whatever SQL it writes.

  python -m copilot.cloud.access setup    load the raw zone, create the views,
                                          authorize them and grant the agent
  python -m copilot.cloud.access check    act as the agent and try every dataset;
                                          writes reports/access_check.json

The service account itself is created once with gcloud (see the README), and
`check` uses impersonation, so no key file is ever downloaded.

Settings: COPILOT_BQ_PROJECT, COPILOT_AGENT_SA (the service account's email).
"""

from __future__ import annotations

import argparse
import json
import os

import pandas as pd

from copilot import config
from copilot.cloud import bigquery as bq

# Columns the agents may see. Anything not listed (tax_id, bank_account,
# contact_email, even as tokens) is left out of the view.
AGENT_VIEWS = {
    "po_scored": ["po_number", "line_no", "vendor_id", "material_id", "qty", "unit_price_usd",
                  "po_date", "promised_date", "received_date", "qc_pass_rate",
                  "should_cost_p50"],
    "vendors": ["vendor_id", "vendor_name", "country", "currency", "categories",
                "credit_rating"],
    "materials": ["material_id", "category", "index_id", "base_uom", "kg_per_unit",
                  "conversion_cost_usd", "ref_qty", "description"],
    "market_indices": ["month", "steel_hrc", "aluminium", "copper", "pp_resin", "kraft_paper",
                       "labor_index", "thb_per_usd", "eur_per_usd"],
    "rfq_quotes": ["rfq_id", "vendor_id", "material_id", "qty", "quoted_price_usd",
                   "quote_date", "lead_time_days", "payment_terms"],
}
SENSITIVE = {"tax_id", "bank_account", "contact_email", "card_pan", "cardholder_name",
             "card_token", "cardholder_token"}
RAW_TABLES = {"vendor_master": "vendor_master.csv", "pcard_transactions": "pcard_transactions.csv"}


def agent_sa() -> str:
    sa = os.environ.get("COPILOT_AGENT_SA")
    if not sa:
        s = bq.settings()
        sa = f"copilot-agent@{s['project']}.iam.gserviceaccount.com"
    return sa


def view_sql(curated: str, table: str) -> str:
    return f"SELECT {', '.join(AGENT_VIEWS[table])}\nFROM `{curated}.{table}`"


def setup() -> dict:
    from google.cloud import bigquery

    s = bq.settings()
    client = bq.client()
    raw = bq.ensure_dataset(s["raw_dataset"])
    curated = bq.ensure_dataset(s["dataset"])
    agent = bq.ensure_dataset(s["agent_dataset"])

    # Raw zone: the sensitive source files as they arrive.
    for table, file in RAW_TABLES.items():
        frame = pd.read_csv(config.RAW / file, dtype=str)
        client.load_table_from_dataframe(
            frame, f"{raw}.{table}",
            job_config=bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")).result()
        print(f"  raw zone      {table:<20} {len(frame):>8,} rows")

    # Agent zone: one view per table the agents use.
    views = []
    for table in AGENT_VIEWS:
        view = bigquery.Table(f"{agent}.{table}")
        view.view_query = view_sql(curated, table)
        client.delete_table(view, not_found_ok=True)
        view = client.create_table(view)
        views.append(view)
        print(f"  agent view    {table}")

    # Authorize the views on the curated dataset, so they can read it for the agent.
    ds = client.get_dataset(curated)
    entries = [e for e in ds.access_entries
               if not (e.entity_type == "view" and e.entity_id.get("datasetId") ==
                       s["agent_dataset"])]
    entries += [bigquery.AccessEntry(None, "view", v.reference.to_api_repr()) for v in views]
    ds.access_entries = entries
    client.update_dataset(ds, ["access_entries"])

    # The agent may read the agent dataset, and only that one.
    sa = agent_sa()
    ds = client.get_dataset(agent)
    if not any(e.entity_id == sa for e in ds.access_entries):
        ds.access_entries = [*ds.access_entries,
                             bigquery.AccessEntry("READER", "userByEmail", sa)]
        client.update_dataset(ds, ["access_entries"])
    print(f"  granted       READER on {s['agent_dataset']} to {sa}")
    return {"raw": raw, "curated": curated, "agent": agent, "service_account": sa}


def agent_client():
    """A BigQuery client that acts as the agent's service account."""
    from google.cloud import bigquery

    s = bq.settings()
    return bigquery.Client(project=s["project"], location=s["location"],
                           credentials=bq.impersonate(agent_sa()))


def attempts() -> list[tuple[str, str, bool]]:
    """(what, SQL, should the agent be allowed?)"""
    s = bq.settings()
    p = s["project"]
    return [
        ("agent view: spend", f"SELECT COUNT(*) AS n FROM `{p}.{s['agent_dataset']}.po_scored`",
         True),
        ("agent view: vendors", f"SELECT * FROM `{p}.{s['agent_dataset']}.vendors` LIMIT 1",
         True),
        ("curated table: vendors (tokens)",
         f"SELECT tax_id FROM `{p}.{s['dataset']}.vendors` LIMIT 1", False),
        ("raw table: vendor master (bank accounts)",
         f"SELECT bank_account FROM `{p}.{s['raw_dataset']}.vendor_master` LIMIT 1", False),
        ("raw table: card transactions (card numbers)",
         f"SELECT card_pan FROM `{p}.{s['raw_dataset']}.pcard_transactions` LIMIT 1", False),
    ]


def check() -> dict:
    client = agent_client()
    # Acting as the agent must work before anything else is tested; otherwise every
    # attempt would fail for the wrong reason and "denied" would mean nothing.
    try:
        from google.auth.transport.requests import Request

        client._credentials.refresh(Request())
    except Exception as exc:
        raise SystemExit(
            f"Could not act as {agent_sa()}.\n  {type(exc).__name__}: {str(exc)[:200]}\n"
            "Grant yourself roles/iam.serviceAccountTokenCreator on the service account "
            "(see the README), wait a few minutes for it to take effect, and try again."
        ) from None
    results = []
    for what, sql, expected in attempts():
        try:
            rows = client.query(sql).result().to_dataframe(create_bqstorage_client=False)
            allowed, detail = True, f"{len(rows)} row(s), columns: {', '.join(rows.columns)}"
            leaked = sorted(SENSITIVE & set(rows.columns))
        except Exception as exc:  # Forbidden / AccessDenied is the expected outcome
            allowed, detail, leaked = False, type(exc).__name__ + ": " + str(exc)[:160], []
        ok = allowed == expected and not leaked
        results.append({"attempt": what, "expected": "allowed" if expected else "denied",
                        "result": "allowed" if allowed else "denied", "ok": ok,
                        "detail": detail})
        print(f"  {'OK ' if ok else 'BAD'}  {what:<45} {'allowed' if allowed else 'denied'}")
    # The report is meant to be published, so the project ID is replaced.
    project = bq.settings()["project"]
    for r in results:
        r["detail"] = r["detail"].replace(project, "<project>")
    report = {"service_account": agent_sa().replace(project, "<project>"),
              "all_ok": all(r["ok"] for r in results), "attempts": results}
    config.REPORTS.mkdir(parents=True, exist_ok=True)
    (config.REPORTS / "access_check.json").write_text(json.dumps(report, indent=2),
                                                      encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Least-privilege access for the agents")
    ap.add_argument("command", choices=["setup", "check"])
    args = ap.parse_args()
    if args.command == "setup":
        setup()
    else:
        report = check()
        print("The agent can read only its views." if report["all_ok"] else
              "Access is not as expected, see reports/access_check.json")


if __name__ == "__main__":
    main()
