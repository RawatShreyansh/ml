.PHONY : lint typecheck test check

lint :
	uv run ruff check
	uv run ruff format --check

typecheck :
	uv run mypy

test :
	uv run pytest

check : lint typecheck test
