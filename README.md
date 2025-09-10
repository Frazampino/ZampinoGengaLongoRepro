# -SLRZampinoGengaLongoRepro
This repository contains the code and PNML models used to compute process model comparison metrics, including fitness, precision, generalization, simplicity, PES, PSP, TAR similarity, and F1-scores for events and relations.

## Repository structure
- `models/`: all PNML files to be compared
- `run_experiment.py`: main script to compute all metrics
- `aggregate_results.py`: (optional) aggregates multiple experiments into a single CSV
- `results/`: folder where CSVs with metrics are saved
- `requirements.txt`: Python dependencies with fixed versions
- `reproducibility_protocol.tex`: document describing the 5-step reproducibility protocol

## Installation
Create a Python 3.11 virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
pip install -r requirements.txt
