"""
Cross-backbone check: does 'grounding improves conformal selective prediction'
hold on a second VLM? Reads exp8_qwen_pope.json (Qwen2-VL-2B, POPE-adversarial)
by default; any exp*.json with the same schema can be passed as argv[1].

Feeds the Qwen2-VL rows of Table 6 (tab:datasets) and the cross-backbone claim.

PROTOCOL (canonical, see canonical_fst.py). Previously an exhaustive
max-over-thresholds search; now the canonical ascending-lambda fixed sequence
(stop at first non-rejection, return last passing index) on PAIRED three-way
disjoint splits.

REPORTING. Per-split sd, 95% CI, two-sided paired p-value on the gain; abort rate
per arm; validity audited as the FRACTION of splits with realised risk above
alpha (test fold, and re-scored on all items as a low-noise proxy), never as mean
risk.
"""
import json, os, sys, numpy as np, warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
import canonical_fst as F

_here = os.path.dirname(os.path.abspath(__file__))
path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_here, "exp8_qwen_pope.json")
d = json.load(open(path))
p_yes = np.array(d["p_yes"]); answer = np.array(d["answer"])
correct = np.array(d["correct"]); owl = np.array(d["owl"])
n = len(correct); conf = np.abs(p_yes - .5) * 2
g = owl.copy(); g[g < 0] = np.median(g[g >= 0])
mm = lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9)
gn = mm(g); g_ag = np.where(answer == 1, gn, 1 - gn)
SC = {"Confidence only":               np.column_stack([p_yes, conf]),
      "Confidence + grounding (ours)": np.column_stack([p_yes, conf, gn, g_ag])}
ALPHAS = (0.10, 0.15)
DELTA, REPS = F.DELTA, F.REPS
GRID = np.linspace(.02, 1, 99)


def rc(s, c):
    c = c[np.argsort(-s)]
    return np.interp(GRID, np.arange(1, len(c) + 1) / len(c),
                     np.cumsum(1 - c) / np.arange(1, len(c) + 1))


A = {k: {"aurc": [], **{a: dict(cov=[], rte=[], ral=[], lam=[]) for a in ALPHAS}}
     for k in SC}
for tr, cal, te in F.SPLITS(n, REPS):
    if len(np.unique(correct[tr])) < 2: continue
    for name, X in SC.items():
        s = LogisticRegression(max_iter=1000).fit(X[tr], correct[tr]).predict_proba(X)[:, 1]
        A[name]["aurc"].append(rc(s[te], correct[te]).mean())
        for a in ALPHAS:
            S = A[name][a]
            lam = F.fst(s[cal], correct[cal], a)
            S["lam"].append(lam)
            if lam is None:
                S["cov"].append(0.0); S["rte"].append(None); S["ral"].append(None); continue
            thr = np.quantile(s[cal], 1 - lam)
            m = s[te] >= thr; k = int(m.sum())
            S["cov"].append(k / len(te))
            S["rte"].append(float((1 - correct[te][m]).mean()) if k else None)
            ma = s >= thr
            S["ral"].append(float((1 - correct[ma]).mean()) if ma.sum() else None)

print(f"{d.get('model','model')} on POPE-adversarial  n={n}  acc={correct.mean():.3f}  "
      f"({REPS} paired three-way disjoint splits)\n")
print(f"{'score':32s}{'AURC v':>10s}" + "".join(
    f"{f'cov@{a:.0%} ^':>16s}{'abort':>8s}{'exc(te)':>9s}{'exc(all)':>10s}" for a in ALPHAS))
for k in SC:
    au = np.array(A[k]["aurc"])
    row = f"{k:32s}{au.mean():10.4f}"
    for a in ALPHAS:
        S = A[k][a]; c = np.array(S["cov"]) * 100
        row += (f"{c.mean():9.1f}+-{c.std(ddof=1):4.1f}%"
                f"{F.abort_rate(S['lam'])*100:7.1f}%"
                f"{F.exceedance(S['rte'],a)*100:8.1f}%{F.exceedance(S['ral'],a)*100:9.1f}%")
    print(row)

print(f"\nValidity audit: exceedance = fraction of splits with realised risk > alpha, "
      f"vs delta={DELTA:.2f}.")
print(f"  exc(te)=test fold n_te={n-2*(n//3)} (unbiased/noisy)   "
      f"exc(all)=all {n} items (low-noise proxy).  Mean risk is NOT the guarantee.")
print("  abort = fraction of splits certifying nothing (coverage 0%, in the mean).")

print(f"\nPaired grounding gains (identical splits, two-sided tests)")
print(F.Gain.header(30))
for a in ALPHAS:
    print(F.gain_stats(np.array(A["Confidence only"][a]["cov"]) * 100,
                       np.array(A["Confidence + grounding (ours)"][a]["cov"]) * 100,
                       f"a={a:.2f} cov gain (pp)").line(30))
print(F.gain_stats(np.array(A["Confidence + grounding (ours)"]["aurc"]) * 1000,
                   np.array(A["Confidence only"]["aurc"]) * 1000,
                   "AURC gain x1e-3").line(30))
