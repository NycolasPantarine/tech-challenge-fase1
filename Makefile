.PHONY: install lint test run

install:
	uv sync

lint:
	uv run ruff check src/ tests/

test:
	uv run pytest tests/ -v

run:
	uv run uvicorn src.api.main:app --reload
	