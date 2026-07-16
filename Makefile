PY ?= .venv/bin/python

.PHONY: setup train app test

setup:
	python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

train:
	PYTHONPATH=src $(PY) -m lapse.train

app:
	$(PY) -m streamlit run app.py

test:
	$(PY) -m pytest -q
