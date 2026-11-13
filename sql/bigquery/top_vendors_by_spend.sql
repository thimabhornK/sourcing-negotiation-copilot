SELECT p.vendor_id, v.vendor_name, ROUND(SUM(p.qty * p.unit_price_usd), 0) AS spend_usd,
       COUNT(*) AS po_lines
FROM `{dataset}.po_scored` p
JOIN `{dataset}.vendors` v USING (vendor_id)
JOIN `{dataset}.materials` m USING (material_id)
WHERE (@year IS NULL OR EXTRACT(YEAR FROM p.po_date) = @year)
  AND (@category IS NULL OR m.category = @category)
GROUP BY p.vendor_id, v.vendor_name
ORDER BY spend_usd DESC, vendor_id
LIMIT {limit}
