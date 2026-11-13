# ML Monitoring Dashboard — shortcuts
#
#   make setup      install dependencies from requirements.txt
#   make test       run the pytest suite
#   make demo       train + simulate 12 batches + print the summary table
#   make dashboard  launch the Streamlit dashboard
#   make clean      remove caches (artifacts/ are kept: they are committed)
#
# Override the interpreter when needed, e.g.:
#   make test PY=/path/to/venv/bin/python

PY ?= python3

.PHONY: setup test demo dashboard clean

setup:
	$(PY) -m pip install -r requirements.txt

test:
	$(PY) -m pytest tests -q

demo:
	$(PY) run_demo.py

dashboard:
	streamlit run dashboard/app.py

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache
