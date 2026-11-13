.PHONY: install data clean-data train score all test lint check-quote

install:
	pip install -e ".[dev]"

data:            ## Generate messy synthetic source files into data/raw
	python -m copilot.datagen.generate

clean-data:      ## Clean, standardise and de-identify into data/curated
	python -m copilot.pipeline.clean

train:           ## Train and evaluate the should-cost model
	python -m copilot.model.should_cost

score:           ## Build the vendor scorecard
	python -m copilot.scoring.vendor_score

all: data clean-data train score

test:
	python -m pytest -q

lint:
	ruff check src tests

check-quote:     ## Example: check one quote against should-cost
	python -m copilot.quote_check --material MAT-0001 --qty 5000 --price 1.62 --currency USD --vendor V-0001 --date 2025-06-15
