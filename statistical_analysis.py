"""Statistical analysis of the pair-level PMMC 2015 F1 scores.

Run after run_experiment.py. Reads the reproducible pair-level CSV from
results/ and writes statistical_analysis.csv/json to the same directory.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

RESULTS_DIR = Path("results")
INPUT_CSV = RESULTS_DIR / "risultati_senza_varianti_calcolati.csv"
OUTPUT_CSV = RESULTS_DIR / "statistical_analysis.csv"
OUTPUT_JSON = RESULTS_DIR / "statistical_analysis.json"
EXPECTED_PAIRS = 7
RANDOM_SEED = 42
N_BOOTSTRAP = 10_000
CI_LEVEL = 0.95

METHOD_COLUMNS = {
    "PES": "PES Behavioral",
    "PM4PY": "PM4Py Behavioral",
    "TAR": "TAR Similarity Behavioral",
}

if not INPUT_CSV.is_file():
    raise FileNotFoundError(f"Missing pair-level results: {INPUT_CSV}")

df = pd.read_csv(INPUT_CSV)
if len(df) != EXPECTED_PAIRS:
    raise ValueError(f"Expected {EXPECTED_PAIRS} result rows, found {len(df)} in {INPUT_CSV}.")

missing = [c for c in METHOD_COLUMNS.values() if c not in df.columns]
if missing:
    raise ValueError(f"Missing F1 columns: {missing}")

scores = pd.DataFrame({m: df[c].astype(float) for m, c in METHOD_COLUMNS.items()})
ranks = scores.rank(axis=1, ascending=False, method="average")

def bootstrap_mean_ci(values, seed):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(N_BOOTSTRAP, len(values)), replace=True)
    means = samples.mean(axis=1)
    alpha = (1.0 - CI_LEVEL) / 2.0
    lo, hi = np.quantile(means, [alpha, 1.0 - alpha])
    return float(lo), float(hi)

summary = []
for i, method in enumerate(scores.columns):
    values = scores[method].to_numpy()
    lo, hi = bootstrap_mean_ci(values, RANDOM_SEED + i)
    summary.append({
        "Method": method,
        "n": int(len(values)),
        "Mean F1": float(np.mean(values)),
        "Median F1": float(np.median(values)),
        "Min F1": float(np.min(values)),
        "Max F1": float(np.max(values)),
        "SD": float(np.std(values, ddof=1)),
        "95% CI lower": lo,
        "95% CI upper": hi,
        "Mean rank": float(ranks[method].mean()),
    })
summary_df = pd.DataFrame(summary)

friedman_stat, friedman_p = friedmanchisquare(scores["PES"], scores["PM4PY"], scores["TAR"])

comparisons = [("PES", "PM4PY"), ("PES", "TAR"), ("PM4PY", "TAR")]
tests = []
for a, b in comparisons:
    try:
        stat, p = wilcoxon(scores[a].to_numpy(), scores[b].to_numpy(), alternative="two-sided", zero_method="wilcox")
    except ValueError:
        stat, p = np.nan, 1.0
    tests.append({"Comparison": f"{a} vs {b}", "Statistic": None if np.isnan(stat) else float(stat), "Raw p-value": float(p)})

order = sorted(range(len(tests)), key=lambda i: tests[i]["Raw p-value"])
adjusted = [None] * len(tests)
previous = 0.0
for pos, idx in enumerate(order):
    corrected = min(1.0, (len(tests) - pos) * tests[idx]["Raw p-value"])
    corrected = max(previous, corrected)
    adjusted[idx] = corrected
    previous = corrected
for i, row in enumerate(tests):
    row["Holm-adjusted p-value"] = float(adjusted[i])
    row["Significant at 0.05"] = bool(adjusted[i] < 0.05)

summary_df.to_csv(OUTPUT_CSV, index=False, float_format="%.9f")
payload = {
    "dataset": "PMMC 2015",
    "number_of_pairs": EXPECTED_PAIRS,
    "random_seed": RANDOM_SEED,
    "bootstrap_iterations": N_BOOTSTRAP,
    "confidence_level": CI_LEVEL,
    "descriptive_statistics": summary,
    "friedman_test": {"statistic": float(friedman_stat), "p_value": float(friedman_p)},
    "wilcoxon_holm": tests,
}
OUTPUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

print("\nFINAL STATISTICAL TABLE")
print(summary_df.round(3).to_string(index=False))
print(f"\nFRIEDMAN: chi-square={friedman_stat:.6f}, p={friedman_p:.6f}")
print("\nWILCOXON-HOLM")
for row in tests:
    print(f"{row['Comparison']}: raw p={row['Raw p-value']:.6f}, Holm p={row['Holm-adjusted p-value']:.6f}, significant={row['Significant at 0.05']}")
print(f"\nSaved: {OUTPUT_CSV}")
print(f"Saved: {OUTPUT_JSON}")
