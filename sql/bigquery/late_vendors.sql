SELECT p.vendor_id, v.vendor_name,
       ROUND(100 * AVG(IF(p.received_date > p.promised_date, 1.0, 0.0)), 1) AS late_pct,
       COUNT(*) AS delivered_lines
FROM `{dataset}.po_scored` p
JOIN `{dataset}.vendors` v USING (vendor_id)
WHERE p.received_date IS NOT NULL
  AND (@year IS NULL OR EXTRACT(YEAR FROM p.po_date) = @year)
GROUP BY p.vendor_id, v.vendor_name
HAVING COUNT(*) >= 20
ORDER BY late_pct DESC, vendor_id
LIMIT {limit}
