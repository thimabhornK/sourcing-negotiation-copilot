-- Clean the raw ERP purchase-order export inside BigQuery.
-- Same rules, in the same order, as src/copilot/pipeline/clean.py and
-- clean_spark.py; `python -m copilot.pipeline.benchmark --bigquery` checks
-- that all three return the same rows.
--
-- {raw}       dataset with the raw export (every column STRING) and the
--             vendor-name map produced by the resolver
-- {dataset}   curated dataset (market_indices)
-- {source}    table name of the raw export
-- The last statement is replaced by `summary` when only counts are needed.

CREATE TEMP FUNCTION be_to_ce(y INT64) AS (IF(y > 2400, y - 543, y));

CREATE TEMP FUNCTION month_number(m STRING) AS ((
  SELECT o + 1
  FROM UNNEST(['jan', 'feb', 'mar', 'apr', 'may', 'jun',
               'jul', 'aug', 'sep', 'oct', 'nov', 'dec']) AS mon WITH OFFSET o
  WHERE mon = LOWER(m)
));

-- ISO, day-first and '5 Apr 2023', with Thai Buddhist-era years.
CREATE TEMP FUNCTION parse_erp_date(raw STRING) AS (
  CASE
    WHEN REGEXP_CONTAINS(TRIM(raw), r'^\d{4}-\d{2}-\d{2}$')
      THEN SAFE.PARSE_DATE('%Y-%m-%d', TRIM(raw))
    WHEN REGEXP_CONTAINS(TRIM(raw), r'^\d{1,2}/\d{1,2}/\d{4}$')
      THEN SAFE.DATE(be_to_ce(CAST(REGEXP_EXTRACT(TRIM(raw), r'/(\d{4})$') AS INT64)),
                     CAST(REGEXP_EXTRACT(TRIM(raw), r'^\d{1,2}/(\d{1,2})/') AS INT64),
                     CAST(REGEXP_EXTRACT(TRIM(raw), r'^(\d{1,2})/') AS INT64))
    WHEN REGEXP_CONTAINS(TRIM(raw), r'^\d{1,2} [A-Za-z]{3} \d{4}$')
      THEN SAFE.DATE(be_to_ce(CAST(REGEXP_EXTRACT(TRIM(raw), r' (\d{4})$') AS INT64)),
                     month_number(REGEXP_EXTRACT(TRIM(raw), r' ([A-Za-z]{3}) ')),
                     CAST(REGEXP_EXTRACT(TRIM(raw), r'^(\d{1,2}) ') AS INT64))
  END
);

CREATE TEMP FUNCTION parse_number(raw STRING) AS (
  SAFE_CAST(REGEXP_REPLACE(raw, r'[^0-9.\-]', '') AS FLOAT64)
);

WITH deduplicated AS (
  SELECT DISTINCT * FROM `{raw}.{source}`
),

parsed AS (
  SELECT
    d.po_number,
    SAFE_CAST(d.line_no AS INT64) AS line_no,
    n.vendor_id,
    UPPER(TRIM(d.material_code)) AS material_id,
    parse_number(d.qty) AS qty_raw,
    parse_number(d.unit_price) AS price_raw,
    CASE LOWER(TRIM(d.currency))
      WHEN 'thb' THEN 'THB' WHEN 'baht' THEN 'THB' WHEN '฿' THEN 'THB'
      WHEN 'usd' THEN 'USD' WHEN 'us$' THEN 'USD'
      WHEN 'eur' THEN 'EUR' WHEN '€' THEN 'EUR' WHEN 'euro' THEN 'EUR'
    END AS currency,
    CASE LOWER(TRIM(d.uom))
      WHEN 'kg' THEN 'kg' WHEN 'kgs' THEN 'kg' WHEN 'ton' THEN 'ton' WHEN 'mt' THEN 'ton'
      WHEN 'g' THEN 'g' WHEN 'pcs' THEN 'pcs' WHEN 'pcs.' THEN 'pcs' WHEN 'ea' THEN 'ea'
    END AS uom,
    parse_erp_date(d.po_date) AS po_date,
    parse_erp_date(d.promised_date) AS promised_date,
    parse_erp_date(d.received_date) AS received_date,
    SAFE_CAST(d.qc_pass_rate AS FLOAT64) AS qc_pass_rate
  FROM deduplicated d
  LEFT JOIN `{raw}.vendor_name_map` n USING (vendor_name)
),

-- Units to kilograms or pieces; currency to USD at the month's rate.
converted AS (
  SELECT
    p.*,
    CASE p.uom WHEN 'kg' THEN 1.0 WHEN 'g' THEN 0.001 WHEN 'ton' THEN 1000.0
               WHEN 'pcs' THEN 1.0 WHEN 'ea' THEN 1.0 END AS factor,
    CASE p.currency WHEN 'THB' THEN fx.thb_per_usd WHEN 'EUR' THEN fx.eur_per_usd
                    ELSE 1.0 END AS rate
  FROM parsed p
  LEFT JOIN `{dataset}.market_indices` fx
    ON DATE(fx.month) = DATE_TRUNC(p.po_date, MONTH)
),

priced AS (
  SELECT
    c.*,
    c.qty_raw * c.factor AS qty,
    SAFE_DIVIDE(SAFE_DIVIDE(c.price_raw, c.factor), c.rate) AS unit_price_usd
  FROM converted c
),

with_median AS (
  SELECT p.*,
         PERCENTILE_CONT(p.unit_price_usd, 0.5) OVER (PARTITION BY p.material_id) AS median_price
  FROM priced p
),

-- First matching reason wins. NULL means the row can be trusted.
cleaned AS (
  SELECT
    po_number, line_no, vendor_id, material_id, qty, unit_price_usd,
    po_date, promised_date, received_date, qc_pass_rate,
    CASE
      WHEN vendor_id IS NULL THEN 'unknown_vendor'
      WHEN qty_raw IS NULL OR price_raw IS NULL THEN 'missing_qty_or_price'
      WHEN qty_raw <= 0 THEN 'negative_or_zero_qty'
      WHEN po_date IS NULL THEN 'unparseable_date'
      WHEN uom IS NULL OR currency IS NULL THEN 'unknown_unit_or_currency'
      WHEN COALESCE(unit_price_usd > 20 * median_price
                    OR unit_price_usd < median_price / 20, FALSE) THEN 'price_outlier'
    END AS reason
  FROM with_median
)

SELECT * FROM cleaned WHERE reason IS NULL;
