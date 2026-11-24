"""The purchase-order cleaning step on Apache Spark.

Same rules as `copilot.pipeline.clean`, written with the Spark DataFrame API so
the step can run on a cluster (Dataproc Serverless) when the ERP exports of
several plants are combined and no longer fit one machine. `copilot.pipeline.
benchmark` checks that both versions return the same rows.

Vendor names are resolved in two parts. Fuzzy matching needs the whole vendor
master, so it runs once on the driver over the *distinct* names (a few hundred,
however many rows there are) and produces a small name-to-ID table. That table
is broadcast and joined to the rows, which is the only part that grows with
the data.

Needs: pip install -e ".[spark]" and Java 17 or newer.
"""

from __future__ import annotations

import pandas as pd

from copilot import config
from copilot.pipeline.clean import CURRENCY_MAP, MONTHS, UOM_MAP, resolve_vendors

RAW_COLUMNS = ["po_number", "line_no", "vendor_name", "material_code", "description", "qty",
               "uom", "unit_price", "currency", "po_date", "promised_date", "received_date",
               "qc_pass_rate"]
OUT_COLUMNS = ["po_number", "line_no", "vendor_id", "material_id", "qty", "unit_price_usd",
               "po_date", "promised_date", "received_date", "qc_pass_rate"]
REASONS = ["unknown_vendor", "missing_qty_or_price", "negative_or_zero_qty", "unparseable_date",
           "unknown_unit_or_currency", "price_outlier"]


def session(cores: str = "*", memory: str = "4g"):
    """A local Spark session. On Dataproc the cluster provides the session instead."""
    from pyspark.sql import SparkSession

    return (SparkSession.builder.master(f"local[{cores}]")
            .appName("sourcing-copilot-clean")
            .config("spark.ui.enabled", "false")
            .config("spark.ui.showConsoleProgress", "false")
            .config("spark.driver.memory", memory)
            .config("spark.sql.shuffle.partitions", "16")
            # Invalid numbers and dates become NULL and are quarantined, as in pandas.
            .config("spark.sql.ansi.enabled", "false")
            .config("spark.sql.session.timeZone", "UTC")
            .getOrCreate())


def read_raw(spark, path):
    """Read the ERP export with every column as text, blanks kept as ''."""
    from pyspark.sql import functions as F
    from pyspark.sql.types import StringType, StructField, StructType

    schema = StructType([StructField(c, StringType()) for c in RAW_COLUMNS])
    df = (spark.read.option("header", True).option("quote", '"').option("escape", '"')
          .option("multiLine", False).schema(schema).csv(str(path)))
    return df.select([F.coalesce(F.col(c), F.lit("")).alias(c) for c in RAW_COLUMNS])


def _strip(col):
    from pyspark.sql import functions as F

    return F.regexp_replace(col, r"^\s+|\s+$", "")


def _lookup(mapping: dict, col):
    """Map a column through a Python dict; values not in the dict become NULL."""
    from itertools import chain

    from pyspark.sql import functions as F

    m = F.create_map([F.lit(x) for x in chain.from_iterable(mapping.items())])
    return m[col]


def _number(col):
    from pyspark.sql import functions as F

    return F.expr(f"try_cast(regexp_replace({col}, '[^0-9.\\\\-]', '') AS DOUBLE)")


def _date(name: str):
    """ISO, day-first, '5 Apr 2023' and Thai Buddhist-era years, as in clean.parse_date."""
    from pyspark.sql import functions as F

    s = _strip(F.col(name))
    iso = r"^\d{4}-\d{2}-\d{2}$"
    dmy = r"^(\d{1,2})/(\d{1,2})/(\d{4})$"
    dmony = r"^(\d{1,2}) ([A-Za-z]{3}) (\d{4})$"

    def be(year):  # Buddhist era to Common era
        return F.when(year > 2400, year - 543).otherwise(year)

    month_num = _lookup(MONTHS, F.lower(F.regexp_extract(s, dmony, 2)))
    y_dmy = F.regexp_extract(s, dmy, 3).cast("int")
    y_mon = F.regexp_extract(s, dmony, 3).cast("int")
    return (F.when(s.rlike(iso), F.to_date(s, "yyyy-MM-dd"))
            .when(s.rlike(dmy), F.make_date(be(y_dmy), F.regexp_extract(s, dmy, 2).cast("int"),
                                            F.regexp_extract(s, dmy, 1).cast("int")))
            .when(s.rlike(dmony), F.make_date(be(y_mon), month_num,
                                              F.regexp_extract(s, dmony, 1).cast("int"))))


def vendor_name_map(spark, raw, vendors: pd.DataFrame) -> pd.DataFrame:
    """Resolve each distinct raw vendor name once, on the driver."""
    names = pd.Series([r[0] for r in raw.select("vendor_name").distinct().collect()], dtype=str)
    return pd.DataFrame({"vendor_name": names,
                         "vendor_id": resolve_vendors(names, vendors).astype(object)})


def clean_po_lines(spark, raw, vendors: pd.DataFrame, indices: pd.DataFrame):
    """Return one DataFrame with every deduplicated row and a `reason` column:
    NULL for rows that pass, otherwise the first check they fail."""
    from pyspark.sql import functions as F

    df = raw.dropDuplicates()

    names = vendor_name_map(spark, raw, vendors)
    names = names[names["vendor_id"].notna()]
    df = df.join(F.broadcast(spark.createDataFrame(names)), "vendor_name", "left")

    uom = _lookup(UOM_MAP, F.lower(_strip(F.col("uom"))))
    factor = _lookup(config.UOM_TO_BASE, uom)
    df = (df.withColumn("material_id", F.upper(_strip(F.col("material_code"))))
            .withColumn("qty_raw", _number("qty"))
            .withColumn("price_raw", _number("unit_price"))
            .withColumn("currency_std", _lookup(CURRENCY_MAP, F.lower(_strip(F.col("currency")))))
            .withColumn("uom_std", uom)
            .withColumn("factor", factor)
            .withColumn("po_date_d", _date("po_date"))
            .withColumn("promised_date_d", _date("promised_date"))
            .withColumn("received_date_d", _date("received_date")))

    fx = spark.createDataFrame(
        indices.assign(month=pd.to_datetime(indices["month"]).dt.date)
        [["month", "thb_per_usd", "eur_per_usd"]])
    df = df.withColumn("month", F.trunc("po_date_d", "month")).join(F.broadcast(fx), "month",
                                                                     "left")
    rate = (F.when(F.col("currency_std") == "THB", F.col("thb_per_usd"))
             .when(F.col("currency_std") == "EUR", F.col("eur_per_usd"))
             .otherwise(F.lit(1.0)))
    df = (df.withColumn("qty_base", F.col("qty_raw") * F.col("factor"))
            .withColumn("unit_price_usd", F.col("price_raw") / F.col("factor") / rate))

    # Median price per material over all rows, exact (same interpolation as pandas).
    med = df.groupBy("material_id").agg(F.expr("percentile(unit_price_usd, 0.5)").alias("med"))
    df = df.join(med, "material_id", "left")

    checks = [
        ("unknown_vendor", F.col("vendor_id").isNull()),
        ("missing_qty_or_price", F.col("qty_raw").isNull() | F.col("price_raw").isNull()),
        ("negative_or_zero_qty", F.col("qty_raw") <= 0),
        ("unparseable_date", F.col("po_date_d").isNull()),
        ("unknown_unit_or_currency", F.col("uom_std").isNull() | F.col("currency_std").isNull()),
        ("price_outlier", (F.col("unit_price_usd") > 20 * F.col("med"))
         | (F.col("unit_price_usd") < F.col("med") / 20)),
    ]
    reason = F.lit(None).cast("string")
    for name, cond in reversed(checks):  # first matching reason wins
        reason = F.when(F.coalesce(cond, F.lit(False)), F.lit(name)).otherwise(reason)

    return df.select(
        "po_number", F.col("line_no").cast("long").alias("line_no"), "vendor_id", "material_id",
        F.col("qty_base").alias("qty"), "unit_price_usd",
        F.col("po_date_d").alias("po_date"), F.col("promised_date_d").alias("promised_date"),
        F.col("received_date_d").alias("received_date"),
        F.col("qc_pass_rate").cast("double").alias("qc_pass_rate"),
        reason.alias("reason"))


def report(result) -> dict:
    """Counts in the same shape as the data-quality report of the pandas version."""
    from pyspark.sql import functions as F

    counts = {r["reason"]: r["n"] for r in
              result.groupBy("reason").agg(F.count("*").alias("n")).collect()}
    out = {f"quarantined_{name}": int(counts.get(name, 0)) for name in REASONS}
    out["rows_out"] = int(counts.get(None, 0))
    out["rows_quarantined"] = int(sum(v for k, v in counts.items() if k is not None))
    return out


def run(path=None, spark=None) -> tuple[pd.DataFrame, dict]:
    """Clean the ERP export with Spark and return (clean rows, counts)."""
    spark = spark or session()
    vendors = pd.read_csv(config.RAW / "vendor_master.csv", dtype=str)
    indices = pd.read_csv(config.RAW / "market_indices.csv", parse_dates=["month"])
    raw = read_raw(spark, path or config.RAW / "erp_po_lines.csv")
    result = clean_po_lines(spark, raw, vendors, indices).cache()
    counts = {"rows_in": raw.count(), **report(result)}
    counts["duplicates_removed"] = counts["rows_in"] - result.count()
    good = result.where("reason IS NULL").drop("reason").toPandas()
    return good, counts


if __name__ == "__main__":
    rows, counts = run()
    for key, value in counts.items():
        print(f"  {key:<40} {value:>8,}")
