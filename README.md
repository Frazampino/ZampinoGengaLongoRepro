# -SLRZampinoGengaLongoRepro
This repository contains the code and PNML models used to compute process model comparison metrics, including fitness, precision, generalization, simplicity, PES, PSP, TAR similarity, and F1-scores for events and relations.

## Dataset benchmark
The PMMC 2015 dataset was downloaded from the official contest website (ai.wu.ac.at/emisa2015).
The dataset was revised from the 2013 version with format fixes and an improved gold standard.

## Repository structure
- `models/`: all PNML files to be compared
- `run_experiment.py`: main script to compute all metrics
- `results/`: folder where CSVs with metrics are saved
- `requirements.txt`: Python dependencies with fixed versions
- `reproducibility_protocol.tex`: document describing the 5-step reproducibility protocol
### Variant Generation

The `models/` folder contains both original PNML models and their synthetic variants.
The `scripts/` folder contains scripts to generate additional synthetic variants:
- `generate_variants.py`: generates new synthetic variants by inserting tasks, adding loops, or renaming activities.  
## Execution Modes

1. **Sequential execution**  
Experiments are currently executed one after another (sequentially). This is the default mode and requires no additional setup.

2. **Parallel execution (optional)**  
Parallel execution is not implemented in the current code. However, it can be achieved by wrapping the experiment loop using Python's `multiprocessing` or `joblib` to process multiple model pairs simultaneously, reducing overall computation time.


## Installation
Create a Python 3.11 virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
pip install -r requirements.txt

## Running with Docker

This project can be executed in a fully reproducible environment using Docker, avoiding the need to manually install Python or dependencies.  

Build the Docker image and run the container:

```bash
docker build -t zgl-repro .
docker run --rm -v "$(pwd)/results:/app/results" zgl-repro

