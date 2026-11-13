# Walkthrough

This page explains each step of the pipeline, why it is built the way it is,
and what to look at in the code. Run `make all` first so the files mentioned
below exist on your machine.

## 1. The source data and its problems

`python -m copilot.datagen.generate` writes these files to `data/raw/`:

| File | What it stands for | State |
| --- | --- | --- |
| `erp_po_lines.csv` | Purchase-order lines exported from an ERP | Messy on purpose |
| `vendor_master.csv` | Vendor records, including tax ID and bank account | Clean but sensitive |
| `material_master.csv` | Materials, their category and the commodity index that drives them | Clean |
| `market_indices.csv` | Monthly commodity, labour and exchange-rate series | Clean |
| `rfq_quotes.csv` | Vendor quotes for each sourcing event | Clean |
| `negotiation_outcomes.csv` | The price finally agreed for each event | Clean |
| `pcard_transactions.csv` | Purchasing-card transactions with card numbers | Clean but sensitive |
| `eval_quote_labels.csv` | Which quotes were overpriced. Used only to score the model | Ground truth |

The generator builds a clean world first. Every price comes from a cost
formula:

```
price = (kg of commodity per unit x vendor's view of the index
         + conversion cost x labour index)
        x volume discount x (1 + vendor margin) x small random noise
```

"Vendor's view of the index" is the detail that makes the data interesting.
When the index rises, vendors pass the increase on in full. When it falls,
each vendor passes on only part of the drop (`sticky_index` in `generate.py`).
That is the behaviour buyers complain about, and it is what the should-cost
model is meant to expose.

`mess.py` then damages the purchase-order export:

| Problem | Example | Share of rows |
| --- | --- | --- |
| Vendor name variants | `Siam Steel Co., Ltd.`, `SIAM STEEL CO.,LTD`, `siam steel` | most rows |
| Typos in vendor names | `Siam Setel` | about 3% |
| Mixed units | a kg material entered in `ton`, `MT` or `g` | about 18% of weight-based lines |
| Mixed currencies and symbols | `THB`, `Baht`, `฿1,234.50`, `US$9.68` | all rows |
| Four date formats | `2023-04-05`, `05/04/2023`, `5 Apr 2023`, `05/04/2566` | all rows |
| Duplicated lines | the same line exported twice | 2% |
| Blank quantity or price | | about 1% |
| Negative quantity | returns booked as negative lines | 0.5% |
| Decimal slips | price 100 times too high | 0.3% |
| Placeholder vendor | `MISC VENDOR - SEE NOTE` | 0.2% |

`05/04/2566` is a Thai Buddhist-era year: 2566 BE is 2023 CE.

## 2. Cleaning

`src/copilot/pipeline/clean.py` has one small function per problem.

- **`normalize_name`** lower-cases, strips punctuation and removes legal words
  such as "Co" and "Ltd", so every variant of a name collapses to one key.
- **`resolve_vendors`** matches that key to the vendor master. Names that do
  not match exactly go through a fuzzy match (`difflib`, similarity 0.88 or
  higher) to recover typos. Anything still unmatched stays unknown.
- **`parse_number`** removes currency symbols and thousands separators.
- **`parse_date`** handles the four formats and converts Buddhist-era years.
- **Units and currency** are converted to a base unit (kg or pieces) and to
  USD at that month's exchange rate. Note that a price per ton must be divided
  by 1,000 as the quantity is multiplied by 1,000.
- **Outliers** are prices more than 20 times above or below the material's
  median, which catches decimal slips without touching normal variation.

Rows that fail a check are written to `data/quarantine/po_lines_rejected.csv`
with a `reason` column. The counts go to `reports/data_quality.json`.

Because the generator knows the truth, `tests/test_clean.py` can check that the
pipeline recovers the right vendor, quantity, price and date. That test is the
main evidence the cleaning works.

**Things to try**

- Lower the fuzzy-match cut-off in `resolve_vendors` to 0.6 and run the tests.
  What starts to go wrong?
- Add a fifth date format to `mess.py` and make `parse_date` handle it.

## 3. De-identification

`src/copilot/pipeline/deidentify.py` replaces sensitive values with tokens
before anything reaches `data/curated/`.

- Tax IDs, bank accounts, card numbers and cardholder names become tokens made
  with keyed hashing (HMAC-SHA256). The same input always produces the same
  token, so tables can still be joined and counted.
- The last four digits of a card are kept, which is what support staff
  normally need.
- Emails keep their first letter and domain.

The key comes from the `COPILOT_DEID_KEY` environment variable. Without the
key, a token cannot be linked back to the original value. On Google Cloud this
step is replaced by Sensitive Data Protection.

## 4. Should-cost model

`src/copilot/model/should_cost.py` answers: what should this material cost on
this date at this quantity?

**Features** (`src/copilot/features.py`): the material and its category, the
quantity, the commodity index now and one and three months ago, and the
labour index. The vendor is left out deliberately.

**Three approaches are compared:**

| Approach | Idea | Strength | Weakness |
| --- | --- | --- | --- |
| Baseline | Last price paid, scaled by the index change | What buyers do today | Inherits whatever was wrong with the last price |
| Model A: cost formula | A linear model per material: `a x index + b x labour`, with a volume term | Easy to explain; follows the index anywhere | One number, no range |
| Model B: boosted trees | Quantile regression for p10, p50 and p90 | Gives a range; finds interactions | Cannot extrapolate beyond index levels seen in training |

The final estimate uses model A for the middle value and model B for the width
of the range.

**Evaluation** uses a time split: train on everything before 2025, test on
2025. The report in `reports/should_cost_eval.json` contains:

- `mape_*`: mean absolute percentage error of each approach
- `band_coverage_p10_p90`: share of actual prices inside the range (80% would
  be perfect calibration)
- `flag_precision` and `flag_recall`: how well "more than X% above
  should-cost" identifies quotes that were really overpriced

The cut-off X is chosen on quotes from before 2025 and then applied unchanged
to 2025 quotes, so the reported precision and recall are not tuned on the test
data.

**Things to try**

- Change `TEST_START` in `config.py` to `2024-01-01`. Does the model get worse
  with a year less of training data?
- Add `vendor_id` as a feature. Accuracy on purchase orders goes up, but what
  happens to overpriced-quote detection, and why?

## 5. Vendor scorecard

`sql/vendor_score.sql` computes one row per vendor:

| Dimension | Weight | How it is measured |
| --- | --- | --- |
| Delivery | 30% | Share of lines received on or before the promised date |
| Quality | 25% | Quality pass rate, weighted by quantity |
| Price | 25% | Median of price paid divided by should-cost |
| Risk | 20% | Credit rating |

The weights live in `vendor_score.py` rather than in the SQL, because
different categories need different priorities. The SQL runs on DuckDB locally
and needs only small changes to run on BigQuery.

## 6. Quote check

`src/copilot/quote_check.py` puts the pieces together for one quote: it
converts the price to USD, gets the should-cost range, compares, and adds the
market trend, the vendor's score and recent purchases of the same material.

It is written as a plain function (`check_quote`) that returns a dictionary,
with a separate `render` function for the text output. That separation is what
lets the chat agents call it as a tool later: the agent gets structured data
and never has to calculate a price itself.

## Tests

```
make test
```

- `tests/test_clean.py`: each cleaning function, then the whole pipeline
  against ground truth
- `tests/test_deidentify.py`: tokens are stable, depend on the key, and no
  sensitive value survives
- `tests/test_model.py`: the model beats the baseline, the range is ordered,
  scores are in range, and the quote check gives the right verdicts

The tests build their own small dataset in a temporary folder, so they do not
touch `data/`.
