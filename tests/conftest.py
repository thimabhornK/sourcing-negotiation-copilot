"""Build a small end-to-end dataset once and point the package at a temp folder."""

import pytest

from copilot import config
from copilot.datagen import generate as gen


@pytest.fixture(scope="session")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("copilot")
    mp = pytest.MonkeyPatch()
    for name in ["RAW", "CURATED", "SERVING", "QUARANTINE"]:
        mp.setattr(config, name, root / "data" / name.lower())
    mp.setattr(config, "MODELS", root / "models")
    mp.setattr(config, "REPORTS", root / "reports")
    tables = gen.generate(n_po=8_000, n_rfq=400, seed=7)
    gen.write_raw(tables, seed=7)
    yield tables
    mp.undo()
