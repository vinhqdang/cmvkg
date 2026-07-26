"""
Stronger-grounding analysis: does a detection-based grounding signal (OWLv2)
beat weak full-image CLIP similarity for conformal selective prediction?

Feeds Table 3 (tab:score) of the manuscript.

Reads raw_scores.json (LLaVA confidence + CLIP grounding) and owlv2_scores.json
(OWLv2 detection grounding), verified item-aligned. Compares learned combiners
over PAIRED random THREE-WAY DISJOINT splits (fit combiner / calibrate lambda /
test). CPU-only.

PROTOCOL (canonical, see canonical_fst.py). This script previously selected the
threshold by an exhaustive max-over-thresholds search over every distinct
calibration score, keeping the highest-coverage passing one. That has no prefix
rule, no k_min start and no multiplicity correction, and it inflates
population-risk exceedance (see protocol_audit.py). It now uses the canonical
ascending-lambda fixed sequence: stop at the first non-rejection, return the last
passing index.

REPORTING. Every gain is reported with its per-split standard deviation, a 95% CI
for the mean, and a two-sided paired p-value. Abort rate (fraction of splits on
which the fixed sequence certifies nothing and coverage is 0%) is reported for
every arm. The validity audit is the FRACTION of splits with realised risk above
alpha, never the mean risk:
  exc(te)  = exceedance on the held-out test fold (unbiased but noisy, n_te=n/3)
  exc(all) = exceedance when the selected policy is re-scored on all n items
             (low-noise proxy for the population risk of the selected policy)
"""
import json, os, numpy as np, warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import canonical_fst as F

_here = os.path.dirname(os.path.abspath(__file__))
r = json.load(open(os.path.join(_here, "raw_scores.json")))
o = json.load(open(os.path.join(_here, "owlv2_scores.json")))
assert o["gold"] == r["gold"], "misaligned!"

p_yes = np.array(r["p_yes"]); clip = np.array(r["ground"]); answer = np.array(r["answer"])
correct = np.array(r["correct"]); det = np.array(o["ground_det"]); n = len(correct)
conf = np.abs(p_yes - 0.5) * 2


def mm(x):
    rng = x.max() - x.min(); return (x - x.min()) / rng if rng > 0 else x * 0


clipn = mm(clip); detn = mm(det)
clip_agree = np.where(answer == 1, clipn, 1 - clipn)
det_agree = np.where(answer == 1, detn, 1 - detn)

FEATS = {
    "learned_conf":       np.column_stack([conf, p_yes]),
    "learned_conf+CLIP":  np.column_stack([conf, p_yes, clipn, clip_agree]),
    "learned_conf+OWLv2": np.column_stack([conf, p_yes, detn, det_agree]),
    "learned_conf+both":  np.column_stack([conf, p_yes, clipn, clip_agree, detn, det_agree]),
}
ARMS = ["raw_conf"] + list(FEATS)
ALPHA, DELTA, REPS = 0.10, F.DELTA, F.REPS


def deploy(lam, s_cal, s_te, ok_te, s_all, ok_all):
    """Returns (coverage on test, risk on test or None, risk on all items or None)."""
    if lam is None: return 0.0, None, None
    thr = np.quantile(s_cal, 1 - lam)
    m = s_te >= thr; k = int(m.sum())
    cov = k / len(s_te)
    rte = float((1 - ok_te[m]).mean()) if k else None
    ma = s_all >= thr
    ral = float((1 - ok_all[ma]).mean()) if ma.sum() else None
    return cov, rte, ral


def aurc(s, c):
    c = c[np.argsort(-s)]
    return float((np.cumsum(1 - c) / np.arange(1, len(c) + 1)).mean())


agg = {k: dict(aurc=[], cov=[], auroc=[], rte=[], ral=[], lam=[]) for k in ARMS}
for tr, cal, te in F.SPLITS(n, REPS):
    if len(np.unique(correct[tr])) < 2: continue
    for name in ARMS:
        if name == "raw_conf":
            s = conf                                  # no fit; score is given
        else:
            s = LogisticRegression(max_iter=1000).fit(
                FEATS[name][tr], correct[tr]).predict_proba(FEATS[name])[:, 1]
        lam = F.fst(s[cal], correct[cal], ALPHA)
        cov, rte, ral = deploy(lam, s[cal], s[te], correct[te], s, correct)
        A = agg[name]
        A["lam"].append(lam); A["cov"].append(cov)
        A["rte"].append(rte); A["ral"].append(ral)
        A["aurc"].append(aurc(s[te], correct[te]))
        A["auroc"].append(roc_auc_score(correct[te], s[te]))

print(f"POPE n={n}  VLM acc={correct.mean():.3f}  ({REPS} paired three-way disjoint "
      f"splits, mean+-sd over splits)\n")
print(f"{'score':22s}{'AURC v':>14s}{'cov@10% ^':>15s}{'AUROC ^':>14s}"
      f"{'abort':>8s}{'exc(te)':>9s}{'exc(all)':>10s}")
for name in ARMS:
    A = agg[name]
    au = np.array(A["aurc"]); cv = np.array(A["cov"]) * 100; ar = np.array(A["auroc"])
    print(f"{name:22s}{au.mean():.4f}+-{au.std(ddof=1):.3f}"
          f"{cv.mean():8.1f}+-{cv.std(ddof=1):4.1f}%"
          f"{ar.mean():9.4f}+-{ar.std(ddof=1):.3f}"
          f"{F.abort_rate(A['lam'])*100:7.1f}%"
          f"{F.exceedance(A['rte'],ALPHA)*100:8.1f}%{F.exceedance(A['ral'],ALPHA)*100:9.1f}%")

print(f"\nValidity audit: exceedance = fraction of splits with realised risk > "
      f"alpha={ALPHA:.2f}; the guarantee requires this <= delta={DELTA:.2f}.")
print(f"  exc(te)=test fold (n_te={n-2*(n//3)}, unbiased/noisy)   "
      f"exc(all)=all {n} items (low-noise proxy)")
print("  Mean realised risk is NOT the guarantee and is deliberately not the audit "
      "statistic.")
print("  abort = fraction of splits certifying nothing (coverage 0%, included in the mean).")

print(f"\nPaired grounding gains vs learned_conf (identical splits, two-sided tests):")
print(F.Gain.header(24))
base = agg["learned_conf"]
for name in ["learned_conf+CLIP", "learned_conf+OWLv2", "learned_conf+both"]:
    A = agg[name]
    print(F.gain_stats(np.array(base["cov"]) * 100, np.array(A["cov"]) * 100,
                       f"{name}  cov pp").line(26))
    print(F.gain_stats(np.array(A["aurc"]) * 1000, np.array(base["aurc"]) * 1000,
                       f"{name}  AURC x1e-3").line(26))
print("\np-values are two-sided and conditional on this dataset: they test whether the")
print("mean gain over the split distribution differs from zero, not whether the gain")
print("would replicate on new items.")
