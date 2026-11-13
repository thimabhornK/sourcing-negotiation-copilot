.PHONY: install data clean-data train score all test lint check-quote chat app eval eval-live bq-load bq-parity

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
	ruff check src tests app adk

chat:            ## Chat with the copilot in the terminal
	python -m copilot.chat

app:             ## Chat in the browser (needs: pip install -e ".[app]")
	streamlit run app/streamlit_app.py

eval:            ## Evaluate routing, retrieval, the reviewer and negotiation strategies
	python -m copilot.eval.run

eval-live:       ## Evaluate the Gemini path (needs GEMINI_API_KEY)
	python -m copilot.eval.live

bq-load:         ## Upload the curated tables to BigQuery (needs COPILOT_BQ_PROJECT)
	python -m copilot.cloud.bigquery load

bq-parity:       ## Run every query on BigQuery and DuckDB and compare
	python -m copilot.cloud.bigquery parity

check-quote:     ## Example: check one quote against should-cost
	python -m copilot.quote_check --material MAT-0001 --qty 5000 --price 1.62 --currency USD --vendor V-0001 --date 2025-06-15
