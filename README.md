# ZampinoGengaLongoRepro

Reproducibility package for *Reproducing and Benchmarking Behavioral Similarity Methods for Business Process Models* (Zampino, Genga, Longo), companion to *Behavioral similarity in business process models: A perspective that needs more attention*, Information Systems, 102608.

The package compares pairs of Petri nets (PNML) with three behavioral similarity methods and reports an F1-score for each pair:

- **PES**: F1-score between the sets of visible events of the two models.
- **TAR**: F1-score between the sets of transition adjacency relations observed in traces simulated from each model.
- **Trace alignment**: harmonic mean of alignment-based fitness and precision, obtained by simulating a log from model A and replaying it on model B with PM4Py.

## Dataset

The process models come from the Process Model Matching Contest 2015 (ai.wu.ac.at/emisa2015). The repository includes the original models and their synthetic variants, obtained by renaming a transition with a semantically related label, inserting a transition with an intermediate place into an existing sequence, and adding a loop to an existing transition.

The dataset and resources are archived on Mendeley Data: https://data.mendeley.com/datasets/xt9gch8nzx/1

## Repository structure

| Path | Content |
|---|---|
| `assets/models/` | PNML models, original and variants |
| `assets/pairs-catalog-models.csv` | The 7 model pairs of the PMMC 2015 experiment |
| `assets/pairs-catalog-var.csv` | The 8 model pairs of the variant experiment |
| `run_experiment.py` | Computes the three methods for every pair of a catalogue |
| `statistical_analysis.py` | Bootstrap confidence intervals, Friedman and Wilcoxon tests |
| `requirements.txt` | Python dependencies with fixed versions |
| `Dockerfile`, `docker-compose.yaml` | Containerized execution |
| `results/` | Output folder, created at the first run |

## Reproducing the experiments with Docker (recommended)

Requirements:

- Docker Desktop (Windows, macOS) or Docker Engine (Linux), version 23 or later.
- Git, to download the repository. It is not installed by default on Windows: install it from https://git-scm.com/download/win, or with `winget install --id Git.Git -e --source winget`, then close and reopen the terminal. Check the installation with `git --version`.

No other software is needed. The commands below are identical on Windows (PowerShell or Command Prompt), macOS and Linux.

```
git --version
docker --version
docker run --rm hello-world
git clone https://github.com/frazampino/ZampinoGengaLongoRepro.git
cd ZampinoGengaLongoRepro
docker build -t zgl-repro .
mkdir results
docker run --rm -v "./results:/app/results" -e PAIRS_CATALOG_CSV=assets/pairs-catalog-models.csv zgl-repro
docker run --rm -v "./results:/app/results" -e PAIRS_CATALOG_CSV=assets/pairs-catalog-var.csv zgl-repro
docker run --rm -v "./results:/app/results" zgl-repro python statistical_analysis.py
```

The statistical analysis reads the results of the PMMC 2015 experiment, so the last command must follow the first `docker run`.

Without Git, the repository can be downloaded as a ZIP archive from this page (**Code**, then **Download ZIP**). After extraction the folder is named `ZampinoGengaLongoRepro-main`: replace the `git clone` command with the extraction, and use `cd ZampinoGengaLongoRepro-main` instead of `cd ZampinoGengaLongoRepro`.

### Notes for a fresh Windows installation

- **Git is not preinstalled.** Install it from https://git-scm.com/download/win or with `winget install --id Git.Git -e --source winget`, then close and reopen the terminal.
- **Docker Desktop must be running.** After installation, start Docker Desktop and wait until its status reports that the engine is running. `docker run --rm hello-world` must print a confirmation message before you build the image.
- **WSL 2 is required.** A fresh Docker Desktop installation enables it by default. If it is missing, run `wsl --install` and restart the computer.
- **Run every command from the repository folder**, that is after `cd ZampinoGengaLongoRepro`, so that `results` is created and read in the same place.

| Message | Cause | Solution |
|---|---|---|
| `git` is not recognised as a cmdlet, function or program | Git is not installed, or the terminal was opened before installing it | Install Git, then close and reopen the terminal |
| `failed to connect to the docker API at npipe:////./pipe/dockerDesktopLinuxEngine` | The Docker Desktop engine is not running yet | Start Docker Desktop and wait until the engine is running, then retry |
| `docker: 'docker run' requires at least 1 argument` | The image name is missing at the end of the command | Copy the whole command, including `zgl-repro` |
| `FileNotFoundError` from `statistical_analysis.py` | The statistical analysis was run before the PMMC 2015 experiment | Run the first `docker run` command, then the statistical analysis |

### Expected results

| File in `results/` | Content | Expected values |
|---|---|---|
| `pairs-catalog-models_results-summary.csv` | Table 4 of the paper | PES 0.385, TAR 0.104, trace alignment 0.267 |
| `pairs-catalog-var_results-summary.csv` | Table 5 of the paper | PES 0.328, TAR 0.027, trace alignment 0.173 |
| `statistical_analysis.csv` | Table 7 of the paper | Mean ranks: PES 1.429, trace alignment 1.857, TAR 2.714; Friedman p = 0.0498 |

Each run of `run_experiment.py` also writes `<catalogue>_results.csv`, with the metrics of every pair and its execution time. The pipeline is deterministic: repeated runs produce identical values.

### With Docker Compose

```
docker compose up --build
docker compose run --rm bpmn python statistical_analysis.py
```

The first command runs the PMMC 2015 experiment. To run the variant experiment, change `PAIRS_CATALOG_CSV` in `docker-compose.yaml` to `assets/pairs-catalog-var.csv`.

## Running without Docker

Create a Python 3.11 virtual environment and install the dependencies.

Linux and macOS:

```
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python run_experiment.py
PAIRS_CATALOG_CSV=assets/pairs-catalog-var.csv python run_experiment.py
python statistical_analysis.py
```

Windows (PowerShell):

```
py -3.11 -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
python run_experiment.py
$env:PAIRS_CATALOG_CSV = "assets/pairs-catalog-var.csv"; python run_experiment.py
Remove-Item Env:PAIRS_CATALOG_CSV
python statistical_analysis.py
```

## Configuration

| Setting | Where | Default |
|---|---|---|
| Pair catalogue | environment variable `PAIRS_CATALOG_CSV` | `assets/pairs-catalog-models.csv` |
| Random seed | `RANDOM_SEED` in `run_experiment.py` | 42 |
| Traces simulated for TAR | `TAR_NUMBER_OF_TRACES` | 100 |
| Traces simulated for trace alignment | `PM4PY_NUMBER_OF_TRACES` | 30 |
| Maximum length of those traces | `PM4PY_MAX_TRACE_LENGTH` | 15 |

A pair catalogue is a CSV file without a header row, with one pair per line: the reference model A in the first column, the compared model B in the second. The order matters for trace alignment. The models are read from `assets/models/`. To run the pipeline on new models without rebuilding the image, mount the folder:

```
docker run --rm -v "./results:/app/results" -v "./assets:/app/assets" -e PAIRS_CATALOG_CSV=assets/mydata.csv zgl-repro
```

To add a similarity method, compute it in `compute_pair_metrics` in `run_experiment.py`, add its name to `reported_metrics` to include it in the summary file, and to `METHOD_COLUMNS` in `statistical_analysis.py` to include it in the statistical analysis.
