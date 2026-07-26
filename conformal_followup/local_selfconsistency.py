"""
Self-consistency baseline head-to-head on POPE-adversarial (hardest split),
LLaVA-1.5-7B. Data: exp7_pope.json (real self-consistency over K=3 samples +
OWLv2 grounding).

Feeds Table 4 (tab:selfcons) of the manuscript.

Baseline: a learned combiner over model-INTERNAL signals only, including
self-consistency [p_yes, conf, sc_yesfrac, sc_conf]. This is the scoring
PHILOSOPHY of ConfLVLM, not ConfLVLM: ConfLVLM's scorer is CLIP image-text
similarity, which appears as a separate arm in local_analysis_owlv2.py. Labelled
here as "internal (conf+self-consistency)" for that reason.
Ours: internal + OWLv2 structured grounding.

PROTOCOL (canonical, see canonical_fst.py). Previously an exhaustive
max-over-thresholds search; now the canonical ascending-lambda fixed sequence
(stop at first non-rejection, return last passing index) on PAIRED three-way
disjoint splits.

CAVEAT that matters here: n=444 grounded items leaves a 148-item calibration
fold, and k_min=22 at alpha=delta=0.10. This is a genuinely underpowered cell;
the abort rate and the per-split sd below are the honest measure of that, and
they are large. Do not read the mean gain without them.
"""
import json, os, numpy as np, warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
import canonical_fst as F

_here = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(_here, "exp7_pope.json")))
p_yes = np.array(d["p_yes"]); answer = np.array(d["answer"]); correct = np.array(d["correct"])
sc = np.array(d["sc_yesfrac"]); owl = np.array(d["owl"]); n = len(correct)
conf = np.abs(p_yes - .5) * 2; sc_conf = np.abs(sc - .5) * 2
# ungrounded (-1) -> neutral median of grounded
g = owl.copy(); med = np.median(g[g >= 0]); g[g < 0] = med
mm = lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9)
gn = mm(g); g_ag = np.where(answer == 1, gn, 1 - gn)

SCORES = {
    "self-consistency only": sc_conf,                    # the literal single signal
    "internal (conf+self-cons)": np.column_stack([p_yes, conf, sc, sc_conf]),
    "internal + grounding (ours)": np.column_stack([p_yes, conf, sc, sc_conf, gn, g_ag]),
}
ALPHA, DELTA, REPS = .10, F.DELTA, F.REPS
GRID = np.linspace(.02, 1, 99)


def rc(s, c):
    c = c[np.argsort(-s)]; cov = np.arange(1, len(c) + 1) / len(c)
    return np.interp(GRID, cov, np.cumsum(1 - c) / np.arange(1, len(c) + 1))


agg = {k: dict(aurc=[], cov=[], rte=[], ral=[], lam=[]) for k in SCORES}
for tr, cal, te in F.SPLITS(n, REPS):
    if len(np.unique(correct[tr])) < 2: continue
    for name, X in SCORES.items():
        s = X if X.ndim == 1 else LogisticRegression(max_iter=1000).fit(
            X[tr], correct[tr]).predict_proba(X)[:, 1]
        A = agg[name]
        A["aurc"].append(rc(s[te], correct[te]).mean())
        lam = F.fst(s[cal], correct[cal], ALPHA)
        A["lam"].append(lam)
        if lam is None:
            A["cov"].append(0.0); A["rte"].append(None); A["ral"].append(None); continue
        thr = np.quantile(s[cal], 1 - lam)
        m = s[te] >= thr; k = int(m.sum())
        A["cov"].append(k / len(te))
        A["rte"].append(float((1 - correct[te][m]).mean()) if k else None)
        ma = s >= thr
        A["ral"].append(float((1 - correct[ma]).mean()) if ma.sum() else None)

print(f"POPE-adversarial (hardest split)  n={n}  LLaVA acc={correct.mean():.3f}  "
      f"({REPS} paired three-way disjoint splits)\n")
print(f"{'score':30s}{'AURC v':>14s}{'cov@10% ^':>15s}"
      f"{'abort':>8s}{'exc(te)':>9s}{'exc(all)':>10s}")
for name in SCORES:
    A = agg[name]
    a = np.array(A["aurc"]); c = np.array(A["cov"]) * 100
    print(f"{name:30s}{a.mean():.4f}+-{a.std(ddof=1):.3f}"
          f"{c.mean():8.1f}+-{c.std(ddof=1):4.1f}%"
          f"{F.abort_rate(A['lam'])*100:7.1f}%"
          f"{F.exceedance(A['rte'],ALPHA)*100:8.1f}%{F.exceedance(A['ral'],ALPHA)*100:9.1f}%")

print(f"\nValidity audit: exceedance = fraction of splits with realised risk > "
      f"alpha={ALPHA:.2f}, vs delta={DELTA:.2f}.")
print(f"  exc(te)=test fold (n_te={n-2*(n//3)}, unbiased/noisy)   "
      f"exc(all)=all {n} items (low-noise proxy).  Mean risk is NOT the guarantee.")
print("  abort = fraction of splits certifying nothing (coverage 0%, in the mean).")

print(f"\nPaired gains (identical splits, two-sided tests)")
print(F.Gain.header(34))
b = agg["internal (conf+self-cons)"]
o = agg["internal + grounding (ours)"]
print(F.gain_stats(np.array(b["cov"]) * 100, np.array(o["cov"]) * 100,
                   "grounding gain, cov@10% (pp)").line(34))
print(F.gain_stats(np.array(o["aurc"]) * 1000, np.array(b["aurc"]) * 1000,
                   "grounding gain, AURC x1e-3").line(34))
print(f"\nn={n} leaves a {n//3}-item calibration fold against k_min="
      f"{F.kmin_for(F.cp_upper, ALPHA, DELTA)}. The per-split sd above is the honest")
print("uncertainty on this cell; the mean alone overstates what the data supports.")
