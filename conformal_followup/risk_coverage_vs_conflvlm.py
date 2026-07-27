"""
Head-to-head risk-coverage comparison against a confidence-only baseline.
Writes risk_coverage.png (Figure 5, fig:riskcov).

BASELINE LABELLING (fixed). This baseline uses ONLY LLaVA's own output logits
[confidence-margin, p_yes] -- no external evidence of any kind. It was previously
labelled "ConfLVLM-style (internal uncertainty)" in the figure legend, which is
wrong twice over: ConfLVLM's scorer is CLIP image-text similarity, i.e. external
visual evidence, and it is not internal uncertainty at all. That arm appears
separately as "+ CLIP grounding" in make_comparison_figure.py and as its own row
of Table 3. The baseline here is now labelled "confidence only (model logits)"
and no longer names ConfLVLM.

Ours: the same conformal procedure with a STRUCTURED grounding signal added
(OWLv2 detection agreement).

PROTOCOL (canonical, see canonical_fst.py). The calibrated arm previously used an
exhaustive max-over-thresholds search; it now uses the canonical ascending-lambda
fixed sequence (stop at first non-rejection, return last passing index) on PAIRED
three-way disjoint splits.

TWO COVERAGE QUANTITIES, kept apart. "oracle cov@10%" is read off the test-fold
risk-coverage curve -- the largest coverage whose empirical test risk is <= 10%.
It is NOT calibrated and is systematically optimistic: it peeks at the test labels.
"calibrated cov@10%" is what the procedure actually certifies, and is the number
the tables report. Both are printed and both are in the legend, labelled.
"""
import json, os, numpy as np, warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import canonical_fst as F

_here = os.path.dirname(os.path.abspath(__file__))
r = json.load(open(os.path.join(_here, "raw_scores.json")))
o = json.load(open(os.path.join(_here, "owlv2_scores.json")))
assert o["gold"] == r["gold"]
p_yes = np.array(r["p_yes"]); answer = np.array(r["answer"]); correct = np.array(r["correct"])
det = np.array(o["ground_det"]); n = len(correct)
conf = np.abs(p_yes - 0.5) * 2


def mm(x):
    d = x.max() - x.min(); return (x - x.min()) / d if d > 0 else x * 0


detn = mm(det); det_agree = np.where(answer == 1, detn, 1 - detn)

BASE = "confidence only (model logits)"
OURS = "+ structured grounding (ours)"
ARMS = {BASE: np.column_stack([conf, p_yes]),
        OURS: np.column_stack([conf, p_yes, detn, det_agree])}

GRID = np.linspace(0.02, 1.0, 99)
ALPHA, DELTA, REPS = 0.10, F.DELTA, F.REPS


def rc_curve(score, corr):
    order = np.argsort(-score); c = corr[order]
    cov = np.arange(1, len(c) + 1) / len(c)
    risk = np.cumsum(1 - c) / np.arange(1, len(c) + 1)
    return np.interp(GRID, cov, risk)


A = {k: dict(curve=[], aurc=[], oracle=[], cov=[], rte=[], ral=[], lam=[]) for k in ARMS}
for tr, cal, te in F.SPLITS(n, REPS):
    if len(np.unique(correct[tr])) < 2: continue
    for name, X in ARMS.items():
        s = LogisticRegression(max_iter=1000).fit(X[tr], correct[tr]).predict_proba(X)[:, 1]
        S = A[name]
        curve = rc_curve(s[te], correct[te])
        S["curve"].append(curve); S["aurc"].append(curve.mean())
        below = GRID[curve <= ALPHA]
        S["oracle"].append(below.max() if len(below) else 0.0)
        lam = F.fst(s[cal], correct[cal], ALPHA)
        S["lam"].append(lam)
        if lam is None:
            S["cov"].append(0.0); S["rte"].append(None); S["ral"].append(None); continue
        thr = np.quantile(s[cal], 1 - lam)
        m = s[te] >= thr; k = int(m.sum())
        S["cov"].append(k / len(te))
        S["rte"].append(float((1 - correct[te][m]).mean()) if k else None)
        ma = s >= thr
        S["ral"].append(float((1 - correct[ma]).mean()) if ma.sum() else None)

# --------------------------------------------------------------------- plot
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})
fig, ax = plt.subplots(figsize=(7.6, 5.2), dpi=150)
COL = {BASE: "#8a94a6", OURS: "#4b57c8"}
for name in ARMS:
    S = A[name]
    M = np.array(S["curve"]); m = M.mean(0); sd = M.std(0, ddof=1)
    ax.fill_between(GRID, m - sd, m + sd, color=COL[name], alpha=.15, lw=0)
    ax.plot(GRID, m, color=COL[name], lw=2.4,
            label=(f"{name}"
                   f"\n   calibrated cov@10% = {np.mean(S['cov'])*100:.1f}%"
                   f"\n   oracle cov@10% = {np.mean(S['oracle'])*100:.1f}%  (not calibrated)"))
ax.axhline(ALPHA, ls="--", lw=1.2, color="#c8384f")
ax.text(0.015, ALPHA + 0.004, r"target risk $\alpha$ = 10%", color="#c8384f",
        fontsize=9.5, va="bottom")
ax.set_xlabel("Coverage  (fraction of questions answered)")
ax.set_ylabel("Risk  (error rate among answered)")
ax.set_title("Risk-coverage on POPE (LLaVA-1.5-7B): structured grounding dominates\n"
             "the confidence-only baseline"
             f"\n({REPS} paired three-way disjoint splits: fit / calibrate / test;"
             " fixed-sequence protocol)",
             fontsize=12, loc="left")
ax.set_xlim(0, 1); ax.set_ylim(0, 0.30)
ax.grid(alpha=.18)
ax.legend(loc="upper left", fontsize=9.0, frameon=False, bbox_to_anchor=(0, 0.94),
          labelspacing=1.1)
for s in ["top", "right"]: ax.spines[s].set_visible(False)
fig.tight_layout()
out = os.path.join(_here, "risk_coverage.png")
fig.savefig(out, bbox_inches="tight", facecolor="white")
print("saved", out)

# ------------------------------------------------------------------ numbers
Ao = np.array(A[OURS]["curve"]).mean(0); Ab = np.array(A[BASE]["curve"]).mean(0)
print(f"\nOurs risk <= baseline at {100*np.mean(Ao <= Ab + 1e-9):.0f}% of coverage levels")
print(f"AURC  baseline={np.mean(A[BASE]['aurc']):.4f}  ours={np.mean(A[OURS]['aurc']):.4f}")
print(f"\n{REPS} paired three-way splits (POPE n={n}), alpha={ALPHA:.2f}, delta={DELTA:.2f}")
print(f"{'arm':34s}{'calib cov':>16s}{'oracle cov':>12s}{'abort':>8s}"
      f"{'exc(te)':>9s}{'exc(all)':>10s}")
for name in ARMS:
    S = A[name]; cv = np.array(S["cov"]) * 100
    print(f"{name:34s}{cv.mean():9.1f}+-{cv.std(ddof=1):4.1f}%"
          f"{np.mean(S['oracle'])*100:11.1f}%{F.abort_rate(S['lam'])*100:7.1f}%"
          f"{F.exceedance(S['rte'],ALPHA)*100:8.1f}%{F.exceedance(S['ral'],ALPHA)*100:9.1f}%")
print("  oracle cov = read off the test-fold curve, NOT calibrated (peeks at test labels);")
print("  calib cov  = what the fixed sequence certifies. These are different quantities.")
print(f"  exceedance = fraction of splits with realised risk > alpha, vs delta={DELTA:.2f};")
print(f"  exc(te)=test fold n_te={n-2*(n//3)} (unbiased/noisy), exc(all)=all {n} items.")
print("  Mean realised risk is NOT the guarantee and is not reported here.")

print(f"\nPaired gains (identical splits, two-sided tests)")
print(F.Gain.header(30))
print(F.gain_stats(np.array(A[BASE]["cov"]) * 100, np.array(A[OURS]["cov"]) * 100,
                   "calibrated cov (pp)").line(30))
print(F.gain_stats(np.array(A[BASE]["oracle"]) * 100, np.array(A[OURS]["oracle"]) * 100,
                   "oracle cov (pp)").line(30))
print(F.gain_stats(np.array(A[OURS]["aurc"]) * 1000, np.array(A[BASE]["aurc"]) * 1000,
                   "AURC x1e-3 (lower better)").line(30))
