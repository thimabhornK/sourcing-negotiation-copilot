"""Shared paths and constants."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
CURATED = DATA / "curated"
SERVING = DATA / "serving"
QUARANTINE = DATA / "quarantine"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

SEED = 42
START_MONTH = "2021-01"
END_MONTH = "2025-12"
# Quotes and purchase orders dated on or after this day are held out for evaluation.
TEST_START = "2025-01-01"

# Weight units converted to kilograms; piece units converted to pieces.
UOM_TO_BASE = {"kg": 1.0, "g": 0.001, "ton": 1000.0, "pcs": 1.0, "ea": 1.0}
