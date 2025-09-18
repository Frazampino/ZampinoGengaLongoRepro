# *1. Environment setup: create a Python 3.11 virtual environment and install the dependencies from requirements.txt*
# *2. Model import and pre-processing: load PNML models, extract events, relations, and TAR*
# *3. Metric computation: measure PES, PSP, TAR similarity, fitness, precision, generalization, simplicity, F1-score*
# *4. Result storage and analysis: save all metrics in a CSV file for each pair of models*
# *5. Reproducibility verification: fully reproducible pipeline with scripts available in the repository.*

"""
run_experiment.py
-----------------
Compute similarity metrics between pairs of PNML process models.

Three complementary methods are used:

1) PES (Process Element Similarity):
   - Jaccard similarity on events and relations
   - Precision, Recall, F1-score on events and relations

2) TAR (Transition Adjacency Relations):
   - Extract consecutive activity pairs via playout simulation
   - DAB similarity (Jaccard)
   - Precision, Recall, F1-score on TARs

3) PM4Py-based metrics:
   - Fitness, Precision, Generalization, Simplicity via replay/conformance
"""

import argparse
import pandas as pd
from pm4py.objects.petri.importer import importer as pnml_importer
from pm4py.algo.simulation.playout.petri_net import algorithm as simulator
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments
from pm4py.algo.evaluation.precision import algorithm as precision_evaluator
from pm4py.algo.evaluation.generalization import algorithm as generalization_evaluator
from pm4py.algo.evaluation.simplicity import algorithm as simplicity_evaluator

# ----------------------------
# Utility functions
# ----------------------------
def jaccard_similarity(set1, set2):
    """Jaccard index |A∩B| / |A∪B|"""
    inter = len(set1 & set2)
    union = len(set1 | set2)
    return inter / union if union > 0 else 0

def precision_recall_f1(gold_set, predicted_set):
    """Compute Precision, Recall, F1-score on sets"""
    tp = len(gold_set & predicted_set)
    precision = tp / len(predicted_set) if predicted_set else 0
    recall = tp / len(gold_set) if gold_set else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return precision, recall, f1

def extract_events_and_relations(net):
    """Extract visible transitions (events) and direct relations from a Petri net"""
    events = {t.label for t in net.transitions if t.label}
    relations = set()
    for arc in net.arcs:
        if arc.source.label and arc.target.label:
            relations.add((arc.source.label, arc.target.label))
    return events, relations

def extract_tar_from_pnml(pnml_path, no_traces=100):
    """Extract Transition Adjacency Relations (TAR) from PNML via playout simulation"""
    net, im, fm = pnml_importer.apply(pnml_path)
    log = simulator.apply(net, im, variant=simulator.Variants.BASIC_PLAYOUT,
                          parameters={"no_traces": no_traces})
    tar = set()
    for trace in log:
        events = [e["concept:name"] for e in trace]
        for i in range(len(events) - 1):
            tar.add((events[i], events[i + 1]))
    return tar

def dab_similarity(tar1, tar2):
    """Compute DAB similarity between TAR sets"""
    inter = tar1 & tar2
    union = tar1 | tar2
    return len(inter) / len(union) if union else 1.0

# ----------------------------
# Compute metrics
# ----------------------------
def compute_metrics(model_a_path, model_b_path):
    # Load Petri nets
    net_a, im_a, fm_a = pnml_importer.apply(model_a_path)
    net_b, im_b, fm_b = pnml_importer.apply(model_b_path)

    # --- PES metrics (events and relations)
    events_a, relations_a = extract_events_and_relations(net_a)
    events_b, relations_b = extract_events_and_relations(net_b)

    pes_events_jaccard = jaccard_similarity(events_a, events_b)
    pes_relations_jaccard = jaccard_similarity(relations_a, relations_b)
    pes_events_precision, pes_events_recall, pes_events_f1 = precision_recall_f1(events_a, events_b)
    pes_relations_precision, pes_relations_recall, pes_relations_f1 = precision_recall_f1(relations_a, relations_b)

    # --- TAR metrics
    tar_a = extract_tar_from_pnml(model_a_path)
    tar_b = extract_tar_from_pnml(model_b_path)
    tar_similarity = dab_similarity(tar_a, tar_b)
    tar_precision, tar_recall, tar_f1 = precision_recall_f1(tar_a, tar_b)

    # --- PM4Py metrics (conformance)
    simulated_log = simulator.apply(
        net_a, im_a, variant=simulator.Variants.BASIC_PLAYOUT,
        parameters={"max_trace_length": 15, "no_traces": 30}
    )
    results = alignments.apply(simulated_log, net_b, im_b, fm_b)
    avg_fitness = sum(r["fitness"] for r in results) / len(results)
    precision_val = precision_evaluator.apply(simulated_log, net_b, im_b, fm_b)
    generalization = generalization_evaluator.apply(simulated_log, net_b, im_b, fm_b)
    simplicity = simplicity_evaluator.apply(net_b)

    return {
        "model_a": model_a_path,
        "model_b": model_b_path,
        # PES
        "pes_events_jaccard": pes_events_jaccard,
        "pes_relations_jaccard": pes_relations_jaccard,
        "pes_events_precision": pes_events_precision,
        "pes_events_recall": pes_events_recall,
        "pes_events_f1": pes_events_f1,
        "pes_relations_precision": pes_relations_precision,
        "pes_relations_recall": pes_relations_recall,
        "pes_relations_f1": pes_relations_f1,
        # TAR
        "tar_similarity": tar_similarity,
        "tar_precision": tar_precision,
        "tar_recall": tar_recall,
        "tar_f1": tar_f1,
        # PM4Py
        "fitness": avg_fitness,
        "precision": precision_val,
        "generalization": generalization,
        "simplicity": simplicity
    }

# ----------------------------
# Main execution
# ----------------------------
def main(pairs_csv, output_csv):
    pairs = pd.read_csv(pairs_csv)
    results = []
    for _, row in pairs.iterrows():
        metrics = compute_metrics(row["model_a"], row["model_b"])
        results.append(metrics)
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"✅ Metrics saved to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute similarity metrics between PNML models.")
    parser.add_argument("--pairs", required=True, help="CSV file with model_a, model_b columns")
    parser.add_argument("--out", default="results/metrics.csv", help="Output CSV file")
    args = parser.parse_args()
    main(args.pairs, args.out)
