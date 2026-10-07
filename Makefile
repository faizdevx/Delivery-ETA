PY ?= python

.PHONY: install data train evaluate notebook test lint typecheck readme serve all

install:
	$(PY) -m pip install -r requirements-dev.txt

data:
	$(PY) scripts/download_data.py

train:
	$(PY) scripts/train_model.py

evaluate:
	$(PY) scripts/evaluate_model.py

notebook:
	$(PY) scripts/build_eda_notebook.py

readme:
	$(PY) scripts/update_readme.py

test:
	$(PY) -m pytest -q

lint:
	ruff check .

typecheck:
	mypy

serve:
	uvicorn app.main:app --reload

all: data train evaluate notebook readme test
