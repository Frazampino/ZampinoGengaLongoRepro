"""
run_experiment.py
-----------------
Compute similarity metrics between two PNML Petri nets and save results in `results/pairs.csv`.
"""

import os
import pandas as pd
from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.algo.simulation.playout.petri_net import algorithm as simulator
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments
from pm4py.algo.evaluation.precision import algorithm as precision_evaluator
from pm4py.algo.evaluation.generalization import algorithm as generalization_evaluator
from pm4py.algo.evaluation.simplicity import algorithm as simplicity_evaluator
import matplotlib.pyplot as plt

# -----------------------------
# Utility functions
# -----------------------------
def jaccard_similarity(set1, set2):
    inter = len(set1 & set2)
    union = len(set1 | set2)
    return inter / union if union > 0 else 0

def precision_recall_f1(gold_set, predicted_set):
    tp = len(gold_set & predicted_set)
    precision = tp / len(predicted_set) if predicted_set else 0
    recall = tp / len(gold_set) if gold_set else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return precision, recall, f1

def harmonic_f1(a, b):
    return 2 * a * b / (a + b) if (a + b) else 0

def extract_events_and_relations(net):
    events = {t.label for t in net.transitions if t.label}
    relations = set()
    for arc in net.arcs:
        if hasattr(arc.source, "label") and hasattr(arc.target, "label"):
            if arc.source.label and arc.target.label:
                relations.add((arc.source.label, arc.target.label))
    return events, relations

def extract_tar_from_pnml(pnml_path, no_traces=100):
    net, im, fm = pnml_importer.apply(pnml_path)
    log = simulator.apply(net, im, variant=simulator.Variants.BASIC_PLAYOUT,
                          parameters={"no_traces": no_traces})
    tar = set()
    for trace in log:
        events = [e["concept:name"] for e in trace]
        for i in range(len(events) - 1):
            tar.add((events[i], events[i+1]))
    return tar

def dab_similarity(tar1, tar2):
    inter = tar1 & tar2
    union = tar1 | tar2
    return len(inter) / len(union) if union else 1.0

# -----------------------------
# Main computation
# -----------------------------

# Crea la cartella results se non esiste
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

# Percorsi modelli
model_a_path = "models-variants/dataset2/models/birthCertificate_p31.pnml"
model_b_path = "models-variants/dataset2/models/birthCertificate_p32.pnml"

# Importa modelli
net_a, im_a, fm_a = pnml_importer.apply(model_a_path)
net_b, im_b, fm_b = pnml_importer.apply(model_b_path)

# --- PES
events_a, relations_a = extract_events_and_relations(net_a)
events_b, relations_b = extract_events_and_relations(net_b)

pes_events_jaccard = jaccard_similarity(events_a, events_b)
pes_relations_jaccard = jaccard_similarity(relations_a, relations_b)
pes_events_precision, pes_events_recall, pes_events_f1 = precision_recall_f1(events_a, events_b)
pes_relations_precision, pes_relations_recall, pes_relations_f1 = precision_recall_f1(relations_a, relations_b)

# --- TAR
tar_a = extract_tar_from_pnml(model_a_path)
tar_b = extract_tar_from_pnml(model_b_path)
tar_similarity = dab_similarity(tar_a, tar_b)
tar_precision, tar_recall, tar_f1 = precision_recall_f1(tar_a, tar_b)

# --- PM4Py metrics
simulated_log = simulator.apply(
    net_a, im_a, variant=simulator.Variants.BASIC_PLAYOUT,
    parameters={"max_trace_length": 15, "no_traces": 30}
)
results_align = alignments.apply(simulated_log, net_b, im_b, fm_b)
avg_fitness = sum(r["fitness"] for r in results_align) / len(results_align)
pm4py_precision = precision_evaluator.apply(simulated_log, net_b, im_b, fm_b)
pm4py_generalization = generalization_evaluator.apply(simulated_log, net_b, im_b, fm_b)
pm4py_simplicity = simplicity_evaluator.apply(net_b)
pm4py_f1 = harmonic_f1(avg_fitness, pm4py_precision)

# --- Save all metrics in CSV inside results/
csv_path = os.path.join(results_dir, "pairs.csv")
df = pd.DataFrame([{
    "model_a": model_a_path,
    "model_b": model_b_path,
    "pes_events_jaccard": pes_events_jaccard,
    "pes_events_precision": pes_events_precision,
    "pes_events_recall": pes_events_recall,
    "pes_events_f1": pes_events_f1,
    "pes_relations_jaccard": pes_relations_jaccard,
    "pes_relations_precision": pes_relations_precision,
    "pes_relations_recall": pes_relations_recall,
    "pes_relations_f1": pes_relations_f1,
    "tar_similarity": tar_similarity,
    "tar_precision": tar_precision,
    "tar_recall": tar_recall,
    "tar_f1": tar_f1,
    "pm4py_fitness": avg_fitness,
    "pm4py_precision": pm4py_precision,
    "pm4py_f1": pm4py_f1,
    "pm4py_generalization": pm4py_generalization,
    "pm4py_simplicity": pm4py_simplicity
}])
df.to_csv(csv_path, index=False)
print(f"✅ All metrics saved to {csv_path}")

# Optional: quick plot of F1 scores
df["pair"] = df["model_a"].str.extract(r'(\d+)') + "-" + df["model_b"].str.extract(r'(\d+)')
metrics_f1 = ["pes_events_f1", "pes_relations_f1", "tar_f1", "pm4py_f1"]

plt.figure(figsize=(12, 6))
for metric in metrics_f1:
    plt.plot(df["pair"], df[metric], marker='o', label=metric)

plt.xlabel("Model Pair")
plt.ylabel("F1 Score")
plt.title("Comparison of F1 Scores for BirthCertificate Model Pairs")
plt.xticks(rotation=45)
plt.ylim(0, 0.5)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()
