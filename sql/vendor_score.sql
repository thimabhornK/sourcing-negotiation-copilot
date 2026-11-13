-- Vendor scorecard: one row per vendor.
-- Inputs: po_scored (PO lines with a should-cost estimate), vendors.
-- Each dimension is scaled to 0-100, then combined with the weights passed in.
WITH per_vendor AS (
  SELECT
    vendor_id,
    COUNT(*)                                                        AS po_lines,
    SUM(qty * unit_price_usd)                                       AS spend_usd,
    AVG(CASE WHEN received_date <= promised_date THEN 1.0 ELSE 0.0 END)
      FILTER (WHERE received_date IS NOT NULL)                      AS on_time_rate,
    SUM(qc_pass_rate * qty) / SUM(qty)                              AS quality_rate,
    MEDIAN(unit_price_usd / should_cost_p50)                        AS price_ratio
  FROM po_scored
  GROUP BY vendor_id
),
scored AS (
  SELECT
    p.*,
    v.vendor_name,
    v.credit_rating,
    100 * on_time_rate                                              AS delivery_score,
    100 * quality_rate                                              AS quality_score,
    -- Paying exactly should-cost scores 75; every 1% above costs 2.5 points.
    GREATEST(0, LEAST(100, 75 - 250 * (price_ratio - 1)))           AS price_score,
    CASE v.credit_rating WHEN 'A' THEN 100 WHEN 'B' THEN 80
                         WHEN 'C' THEN 55 ELSE 25 END               AS risk_score
  FROM per_vendor p
  JOIN vendors v USING (vendor_id)
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
  ROUND($w_delivery * delivery_score + $w_quality * quality_score
        + $w_price * price_score + $w_risk * risk_score, 1) AS total_score
FROM scored
ORDER BY total_score DESC;
