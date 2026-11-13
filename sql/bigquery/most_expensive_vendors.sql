WITH lines AS (
  SELECT p.vendor_id, v.vendor_name, p.qty, p.unit_price_usd, p.should_cost_p50,
         PERCENTILE_CONT(p.unit_price_usd / p.should_cost_p50, 0.5)
           OVER (PARTITION BY p.vendor_id) AS median_ratio
  FROM `{dataset}.po_scored` p
  JOIN `{dataset}.vendors` v USING (vendor_id)
  JOIN `{dataset}.materials` m USING (material_id)
  WHERE (@year IS NULL OR EXTRACT(YEAR FROM p.po_date) = @year)
    AND (@category IS NULL OR m.category = @category)
    AND (@material IS NULL OR p.material_id = @material)
)
SELECT vendor_id, vendor_name,
       ROUND(100 * (ANY_VALUE(median_ratio) - 1), 1) AS pct_above_should_cost,
       ROUND(SUM(qty * (unit_price_usd - should_cost_p50)), 0) AS excess_spend_usd,
       COUNT(*) AS po_lines
FROM lines
GROUP BY vendor_id, vendor_name
HAVING COUNT(*) >= 5
ORDER BY pct_above_should_cost DESC, vendor_id
LIMIT {limit}
