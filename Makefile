.PHONY: install test lint demo clean

install:
	python -m pip install -e ".[dev]"

test:
	pytest --cov=image_rerank

lint:
	ruff check src tests scripts

demo:
	python scripts/run_synthetic_demo.py

clean:
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov
