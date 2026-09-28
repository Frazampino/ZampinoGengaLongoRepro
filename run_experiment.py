"""
run_experiment.py
-----------------
Compute similarity metrics between two PNML Petri nets and save results in `results/pairs.csv`.
"""

# Installazione, se necessaria:
# %pip install pm4py pandas numpy

import os
import sys
import random
import platform
import re
from pathlib import Path
import time
import numpy as np
import pandas as pd
import pm4py
from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.algo.simulation.playout.petri_net import algorithm as simulator
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments
from pm4py.algo.evaluation.precision import algorithm as precision_evaluator

# Deterministic playout. pm4py's basic playout draws the next transition with
# random.choice(list(enabled)), where `enabled` is a set of Transition objects
# hashed by memory address: its iteration order differs in every process, so a
# fixed seed alone does not make the simulated logs reproducible. Sorting the
# candidates by their stable PNML identifier removes that source of variation.
from pm4py.algo.simulation.playout.petri_net.variants import basic_playout as _basic_playout

def _deterministic_choice(candidates):
    ordered = sorted(candidates, key=lambda t: (t is None, "" if t is None else str(t.name)))
    return random.choice(ordered)

_basic_playout.choice = _deterministic_choice

print("Python:", sys.version.replace("\n", " "))
print("Platform:", platform.platform())
print("pandas:", pd.__version__)
print("numpy:", np.__version__)
print("pm4py:", pm4py.__version__)


# I PNML devono trovarsi nella stessa cartella del notebook.
MODELS_DIR = Path("assets/models/")

# Carica il catalogo delle coppie da un file CSV, se presente.
PAIRS_CATALOG_CSV = os.environ.get("PAIRS_CATALOG_CSV", "assets/pairs-catalog-models.csv")
print("Contenuto cartella modelli:", Path(PAIRS_CATALOG_CSV))


# Parametri dell'esperimento.
RANDOM_SEED = 42
TAR_NUMBER_OF_TRACES = 100
PM4PY_NUMBER_OF_TRACES = 30
PM4PY_MAX_TRACE_LENGTH = 15

RESULTS_CSV = Path(PAIRS_CATALOG_CSV).stem + '_results' + Path(PAIRS_CATALOG_CSV).suffix
SUMMARY_CSV = Path(PAIRS_CATALOG_CSV).stem + '_results-summary' + Path(PAIRS_CATALOG_CSV).suffix
RESULTS_DIR = Path("results/")
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

# Coppie dell'esperimento, definite esplicitamente prima dell'esecuzione.
# L'ordine A -> B è rilevante per PM4Py.
print(f"Caricamento coppie da {PAIRS_CATALOG_CSV}")
df_pairs = pd.read_csv(Path(PAIRS_CATALOG_CSV), header=None)
PAIRS = list(df_pairs.itertuples(index=False, name=None))



# PAIRS = [('birthCertificate_p248.pnml', 'birthCertificate_p249.pnml'), ('birthCertificate_p249.pnml', 'birthCertificate_p250.pnml'), ('birthCertificate_p31 (3).pnml', 'birthCertificate_p33.pnml'), ('birthCertificate_p248.pnml', 'birthCertificate_p250.pnml'), ('birthCertificate_p248.pnml', 'birthCertificate_p247.pnml'), ('birthCertificate_p249.pnml', 'birthCertificate_p248.pnml'), ('birthCertificate_p249.pnml', 'birthCertificate_p247.pnml')]

# print("Numero di coppie:", len(PAIRS))
# for index, (model_a, model_b) in enumerate(PAIRS, start=1):
#     print(f"{index:2d}. {model_a} -> {model_b}")

def precision_recall_f1(reference_set, candidate_set):
    true_positives = len(reference_set & candidate_set)
    precision = true_positives / len(candidate_set) if candidate_set else 0.0
    recall = true_positives / len(reference_set) if reference_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def harmonic_f1(value_a, value_b):
    return 2 * value_a * value_b / (value_a + value_b) if value_a + value_b else 0.0


def extract_visible_events(net):
    return {transition.label for transition in net.transitions if transition.label}


def extract_tar(pnml_path, number_of_traces):
    net, initial_marking, _ = pnml_importer.apply(str(pnml_path))
    log = simulator.apply(
        net,
        initial_marking,
        variant=simulator.Variants.BASIC_PLAYOUT,
        parameters={"noTraces": number_of_traces},
    )

    tar = set()
    for trace in log:
        events = [event["concept:name"] for event in trace]
        tar.update(zip(events, events[1:]))
    return tar


def compute_pair_metrics(model_a_path, model_b_path):
    net_a, initial_a, _ = pnml_importer.apply(str(model_a_path))
    net_b, initial_b, final_b = pnml_importer.apply(str(model_b_path))

    # PES Behavioral: F-score tra gli insiemi degli eventi visibili.
    events_a = extract_visible_events(net_a)
    events_b = extract_visible_events(net_b)
    pes_precision, pes_recall, pes_f1 = precision_recall_f1(events_a, events_b)

    # TAR Similarity Behavioral: F-score tra le relazioni di adiacenza osservate.
    tar_a = extract_tar(model_a_path, TAR_NUMBER_OF_TRACES)
    tar_b = extract_tar(model_b_path, TAR_NUMBER_OF_TRACES)
    tar_precision, tar_recall, tar_f1 = precision_recall_f1(tar_a, tar_b)

    # PM4Py Behavioral: log simulato da A e valutato sul modello B.
    simulated_log = simulator.apply(
        net_a,
        initial_a,
        variant=simulator.Variants.BASIC_PLAYOUT,
        parameters={
            "maxTraceLength": PM4PY_MAX_TRACE_LENGTH,
            "noTraces": PM4PY_NUMBER_OF_TRACES,
        },
    )
    alignment_results = alignments.apply(simulated_log, net_b, initial_b, final_b)
    fitness = (
        sum(result["fitness"] for result in alignment_results) / len(alignment_results)
        if alignment_results else 0.0
    )
    conformance_precision = precision_evaluator.apply(
        simulated_log, net_b, initial_b, final_b
    )
    pm4py_f1 = harmonic_f1(fitness, conformance_precision)

    return {
        "pes_precision": pes_precision,
        "pes_recall": pes_recall,
        "PES Behavioral": pes_f1,
        "tar_precision": tar_precision,
        "tar_recall": tar_recall,
        "TAR Similarity Behavioral": tar_f1,
        "pm4py_fitness": fitness,
        "pm4py_precision": conformance_precision,
        "PM4Py Behavioral": pm4py_f1,
    }


#notebook ha trovato il file corretto nella cartella, anche se il nome contiene suffissi aggiunti automaticamente

def normalized_pnml_name(filename):
    # I suffissi (1), (2), ecc. possono essere aggiunti automaticamente
    # durante download multipli e non identificano un modello diverso.
    stem = re.sub(r"\(\d+\)", "", Path(filename).stem)
    return re.sub(r"[^a-z0-9]", "", stem.lower())


def resolve_pnml(filename):
    exact_path = MODELS_DIR / filename
    if exact_path.is_file():
        return exact_path

    expected_key = normalized_pnml_name(filename)
    matches = sorted(
        path for path in MODELS_DIR.glob("*.pnml")
        if normalized_pnml_name(path.name) == expected_key
    )
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise FileExistsError(
            f"Più file corrispondono a {filename}: "
            + ", ".join(path.name for path in matches)
        )
    raise FileNotFoundError(
        f"PNML non trovato per {filename}. Cartella controllata: {MODELS_DIR.resolve()}"
    )

# print("contenuto cartella modelli:", os.listdir(Path("./models-variants")))


resolved_pairs = [
    (model_a, model_b, resolve_pnml(model_a), resolve_pnml(model_b))
    for model_a, model_b in PAIRS
]

print("File PNML risolti:")
for model_a, model_b, path_a, path_b in resolved_pairs:
    print(f"- {model_a} -> {path_a.name}")
    print(f"  {model_b} -> {path_b.name}")

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

rows = []
for pair_number, (model_a, model_b, path_a, path_b) in enumerate(resolved_pairs, start=1):
    start_time = time.time()
    print(f"[{pair_number}/{len(PAIRS)}] {model_a} -> {model_b}")
    metrics = compute_pair_metrics(path_a, path_b)
    end_time = time.time()
    rows.append({
        "pair": pair_number,
        "model_a": model_a,
        "model_b": model_b,
        **metrics,
        "duration_seconds": end_time - start_time,
    })

results_df = pd.DataFrame(rows)
# display(results_df.round(6))


reported_metrics = [
    "PES Behavioral",
    "TAR Similarity Behavioral",
    "PM4Py Behavioral",
    "duration_seconds",
]

means = results_df[reported_metrics].mean()
summary_df = means.rename("mean").to_frame().T

# print("Medie calcolate sulle coppie eseguite:")
# display(summary_df.round(6))


results_df.to_csv(os.path.join(RESULTS_DIR, RESULTS_CSV), index=False, float_format="%.9f")
summary_df.to_csv(os.path.join(RESULTS_DIR, SUMMARY_CSV), index=False, float_format="%.9f")

total_duration = end_time - start_time

print("Risultati individuali:", Path(RESULTS_DIR / RESULTS_CSV).resolve())
print("Medie:", Path(RESULTS_DIR / SUMMARY_CSV).resolve())
