"""De-identify sensitive columns before data reaches the curated zone.

Local stand-in for Google Cloud Sensitive Data Protection. It uses keyed
hashing (HMAC-SHA256), so the same input always gives the same token and
tables can still be joined on it, but the original value cannot be read back
without the key. On Google Cloud this step is replaced by a de-identify
template (format-preserving encryption for card numbers).
"""

from __future__ import annotations

import hashlib
import hmac
import os

import pandas as pd

DEV_KEY = "dev-only-key-do-not-use-with-real-data"


def _key() -> bytes:
    return os.environ.get("COPILOT_DEID_KEY", DEV_KEY).encode()


def tokenize(value: str, prefix: str = "tok") -> str:
    digest = hmac.new(_key(), str(value).encode(), hashlib.sha256).hexdigest()
    return f"{prefix}_{digest[:16]}"


def mask_email(email: str) -> str:
    local, _, domain = str(email).partition("@")
    return f"{local[:1]}***@{domain}" if domain else "***"


def deidentify_vendors(vendors: pd.DataFrame) -> pd.DataFrame:
    out = vendors.copy()
    out["tax_id"] = out["tax_id"].astype(str).map(lambda v: tokenize(v, "tax"))
    out["bank_account"] = out["bank_account"].astype(str).map(lambda v: tokenize(v, "bank"))
    out["contact_email"] = out["contact_email"].map(mask_email)
    return out


def deidentify_pcard(pcard: pd.DataFrame) -> pd.DataFrame:
    """Keep the last four digits for support staff; tokenise the rest."""
    out = pcard.copy()
    pan = out.pop("card_pan").astype(str)
    out["card_token"] = pan.map(lambda v: tokenize(v, "card"))
    out["card_last4"] = pan.str[-4:]
    out["cardholder_token"] = out.pop("cardholder_name").map(lambda v: tokenize(v, "holder"))
    return out
