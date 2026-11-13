SELECT m.category, ROUND(SUM(p.qty * p.unit_price_usd), 0) AS spend_usd,
       COUNT(DISTINCT p.vendor_id) AS vendors
FROM `{dataset}.po_scored` p
JOIN `{dataset}.materials` m USING (material_id)
WHERE (@year IS NULL OR EXTRACT(YEAR FROM p.po_date) = @year)
GROUP BY m.category
ORDER BY spend_usd DESC, category
