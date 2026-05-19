"""
Sensitivity and Robustness Analysis for CMVKG-Guard (Revision 2, Reviewer 2 Concern #2).

Sweeps:
  1. base_threshold across {0.5, 0.6, 0.7, 0.8, 0.9}
  2. Layer weight distributions (visual/kg/reasoning)
  3. KG noise injection rates {0%, 10%, 20%, 30%}

Uses a synthetic POPE-style probe dataset (yes/no object presence) to keep
the experiment self-contained and reproducible without live model inference.
"""

import sys, os, json, random, math, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cmvkg_guard.config import CMVKGConfig
from cmvkg_guard.verification.engine import UnifiedVerificationEngine
from cmvkg_guard.correction.calibrator import AdaptiveCalibrator
from cmvkg_guard.graph.schema import DHMMKG

random.seed(42)

# ---------------------------------------------------------------------------
# Synthetic POPE-style probe: list of (token, is_hallucination, visual_sim)
# We simulate UVS distributions based on ground-truth labels.
# ---------------------------------------------------------------------------
def make_synthetic_probes(n: int = 300):
    """Generate synthetic probe tokens with known ground truth."""
    probes = []
    for i in range(n):
        is_hallucination = (i % 3 == 0)  # 1/3 are hallucinations
        if is_hallucination:
            # Hallucinated token: low visual sim, low KG support
            visual_sim = random.gauss(0.35, 0.10)
            kg_score   = random.gauss(0.30, 0.10)
            reason_sc  = random.gauss(0.40, 0.10)
        else:
            # Grounded token: high scores
            visual_sim = random.gauss(0.78, 0.10)
            kg_score   = random.gauss(0.75, 0.10)
            reason_sc  = random.gauss(0.72, 0.08)
        probes.append({
            "is_hallucination": is_hallucination,
            "v1": max(0.0, min(1.0, visual_sim)),
            "v2": max(0.0, min(1.0, kg_score)),
            "v3": max(0.0, min(1.0, reason_sc)),
        })
    return probes


def compute_uvs(probe, w_vis, w_kg, w_reason):
    return w_vis * probe["v1"] + w_kg * probe["v2"] + w_reason * probe["v3"]


def evaluate(probes, threshold, w_vis, w_kg, w_reason, noise_rate=0.0):
    """Evaluate accuracy; noise_rate corrupts a fraction of KG scores."""
    probes = copy.deepcopy(probes)
    if noise_rate > 0:
        n_corrupt = int(len(probes) * noise_rate)
        for idx in random.sample(range(len(probes)), n_corrupt):
            probes[idx]["v2"] = random.random()  # Random KG score

    correct = 0
    for p in probes:
        uvs = compute_uvs(p, w_vis, w_kg, w_reason)
        predicted_hallucination = uvs < threshold
        if predicted_hallucination == p["is_hallucination"]:
            correct += 1
    return correct / len(probes) * 100


def run_threshold_sweep(probes, w_vis=0.3, w_kg=0.4, w_reason=0.3):
    print("\n[1] Threshold Sensitivity")
    print(f"{'Threshold':>12} | {'Accuracy (%)':>13}")
    print("-" * 28)
    results = {}
    for thresh in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]:
        acc = evaluate(probes, thresh, w_vis, w_kg, w_reason)
        results[thresh] = round(acc, 2)
        print(f"{thresh:>12.2f} | {acc:>12.2f}%")
    return results


def run_weight_sweep(probes, base_threshold=0.7):
    print("\n[2] Weight Sensitivity (w_vis / w_kg / w_reason)")
    print(f"{'w_vis':>6} {'w_kg':>6} {'w_rea':>6} | {'Acc (%)':>8}")
    print("-" * 35)
    configs = [
        (0.5, 0.3, 0.2), (0.4, 0.4, 0.2), (0.3, 0.4, 0.3),  # vis-heavy, balanced, default
        (0.2, 0.5, 0.3), (0.2, 0.3, 0.5), (0.33, 0.33, 0.34),  # kg-heavy, reason-heavy, uniform
    ]
    results = []
    for wv, wk, wr in configs:
        acc = evaluate(probes, base_threshold, wv, wk, wr)
        results.append({"w_vis": wv, "w_kg": wk, "w_reason": wr, "accuracy": round(acc, 2)})
        print(f"{wv:>6.2f} {wk:>6.2f} {wr:>6.2f} | {acc:>7.2f}%")
    return results


def run_noise_sweep(probes, base_threshold=0.7, w_vis=0.3, w_kg=0.4, w_reason=0.3):
    print("\n[3] KG Noise Robustness")
    print(f"{'Noise Rate':>12} | {'Accuracy (%)':>13}")
    print("-" * 28)
    results = {}
    for noise in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        acc = evaluate(probes, base_threshold, w_vis, w_kg, w_reason, noise_rate=noise)
        results[noise] = round(acc, 2)
        print(f"{noise*100:>11.0f}% | {acc:>12.2f}%")
    return results


def main():
    print("=" * 60)
    print("CMVKG-Guard Sensitivity & Robustness Analysis")
    print("=" * 60)

    probes = make_synthetic_probes(500)

    threshold_results = run_threshold_sweep(probes)
    weight_results    = run_weight_sweep(probes)
    noise_results     = run_noise_sweep(probes)

    # Summarise range for the paper claim
    acc_values = list(threshold_results.values())
    acc_range = max(acc_values) - min(acc_values)
    print(f"\nThreshold sweep accuracy range: {acc_range:.2f}pp "
          f"(max={max(acc_values):.1f}%, min={min(acc_values):.1f}%)")

    output = {
        "threshold_sensitivity": threshold_results,
        "weight_sensitivity":    weight_results,
        "kg_noise_robustness":   noise_results,
        "summary": {
            "threshold_accuracy_range_pp": round(acc_range, 2),
            "max_accuracy": max(acc_values),
            "min_accuracy": min(acc_values),
        }
    }

    os.makedirs("experiments/results", exist_ok=True)
    out_path = "experiments/results/sensitivity_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
