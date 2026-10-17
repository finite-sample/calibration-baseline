.PHONY: check paper reproduce ci-docker
check:
	python -m black --check calibration.py make_tables.py tests
	python -m isort --check-only calibration.py make_tables.py tests
	python -m flake8 calibration.py make_tables.py tests
	python -m pytest -q

paper:
	latexmk -pdf -interaction=nonstopmode -halt-on-error paper.tex

reproduce:
	./build.sh

ci-docker:
	COPYFILE_DISABLE=1 tar --exclude=.git --exclude=.venv --exclude=__pycache__ --exclude=.pytest_cache -cf - . | docker run --rm -i -e UV_PROJECT_ENVIRONMENT=/tmp/calibration-ci -w /work python:3.14-slim sh -c 'tar -xf - && apt-get update && apt-get install -y --no-install-recommends make && pip install uv && uv sync --extra dev --frozen && uv run make check'
