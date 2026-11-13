"""Build a small end-to-end dataset once and point the package at a temp folder."""

import pytest

from copilot import config
from copilot.datagen import generate as gen
from copilot.model import should_cost
from copilot.pipeline import clean
from copilot.scoring import vendor_score


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


@pytest.fixture(scope="session")
def trained(world):
    """Run the whole pipeline once: clean, train, score."""
    clean.run()
    report = should_cost.run()
    vendor_score.run()
    return report
