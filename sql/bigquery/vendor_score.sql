-- Vendor scorecard on BigQuery. Same logic as sql/vendor_score.sql (DuckDB).
-- Differences: BigQuery has no MEDIAN aggregate, so the median is taken with
-- PERCENTILE_CONT as a window function first; FILTER (WHERE ...) becomes IF().
WITH po AS (
  SELECT
    *,
    PERCENTILE_CONT(unit_price_usd / should_cost_p50, 0.5)
      OVER (PARTITION BY vendor_id) AS vendor_price_ratio
  FROM `{dataset}.po_scored`
),
per_vendor AS (
  SELECT
    vendor_id,
    COUNT(*)                                                        AS po_lines,
    SUM(qty * unit_price_usd)                                       AS spend_usd,
    AVG(IF(received_date IS NULL, NULL,
           IF(received_date <= promised_date, 1.0, 0.0)))           AS on_time_rate,
    SUM(qc_pass_rate * qty) / SUM(qty)                              AS quality_rate,
    ANY_VALUE(vendor_price_ratio)                                   AS price_ratio
  FROM po
  GROUP BY vendor_id
),
scored AS (
  SELECT
    p.*,
    v.vendor_name,
    v.credit_rating,
    100 * on_time_rate                                              AS delivery_score,
    100 * quality_rate                                              AS quality_score,
    GREATEST(0, LEAST(100, 75 - 250 * (price_ratio - 1)))           AS price_score,
    CASE v.credit_rating WHEN 'A' THEN 100 WHEN 'B' THEN 80
                         WHEN 'C' THEN 55 ELSE 25 END               AS risk_score
  FROM per_vendor p
  JOIN `{dataset}.vendors` v USING (vendor_id)
)
SELECT
  vendor_id, vendor_name, credit_rating, po_lines,
  ROUND(spend_usd, 0)      AS spend_usd,
  ROUND(on_time_rate, 3)   AS on_time_rate,
  ROUND(quality_rate, 3)   AS quality_rate,
  ROUND(price_ratio, 3)    AS price_ratio,
  ROUND(delivery_score, 1) AS delivery_score,
  ROUND(quality_score, 1)  AS quality_score,
  ROUND(price_score, 1)    AS price_score,
  ROUND(risk_score, 1)     AS risk_score,
  ROUND(@w_delivery * delivery_score + @w_quality * quality_score
        + @w_price * price_score + @w_risk * risk_score, 1) AS total_score
FROM scored
ORDER BY total_score DESC, vendor_id
