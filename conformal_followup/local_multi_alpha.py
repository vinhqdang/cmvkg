"""
Multi-alpha results: how much coverage each score buys at several risk targets.
Real POPE / LLaVA-1.5-7B, PAIRED THREE-WAY DISJOINT splits (fit combiner /
calibrate lambda / test). CPU-only. Also writes coverage_vs_alpha.png.

Feeds Table 14 (tab:multialpha) and Figure 6 (fig:covalpha).

PROTOCOL (canonical, see canonical_fst.py). Previously an exhaustive
max-over-thresholds search; now the canonical ascending-lambda fixed sequence
(stop at first non-rejection, return last passing index), with the k_min start
recomputed per alpha -- which matters here, because k_min is 45 at alpha=0.05
against 11 at alpha=0.20, so the grid itself moves with the risk target.

REPORTING. Per-split sd, 95% CI, two-sided paired p-values on every gain; abort
rate for every cell; validity audited as the FRACTION of splits with realised
risk above alpha (test fold and re-scored on all items), never as mean risk.
"""
import json, os, numpy as np, warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import canonical_fst as F

_here = os.path.dirname(os.path.abspath(__file__))
r = json.load(open(os.path.join(_here, "raw_scores.json")))
o = json.load(open(os.path.join(_here, "owlv2_scores.json")))
p_yes = np.array(r["p_yes"]); answer = np.array(r["answer"]); correct = np.array(r["correct"])
det = np.array(o["ground_det"]); n = len(correct); conf = np.abs(p_yes - .5) * 2
mm = lambda x: (x - x.min()) / (x.max() - x.min())
detn = mm(det); det_ag = np.where(answer == 1, detn, 1 - detn)
clipn = mm(np.array(r["ground"])); clip_ag = np.where(answer == 1, clipn, 1 - clipn)
SC = {"Confidence only":            np.column_stack([conf, p_yes]),
      "+ CLIP grounding":           np.column_stack([conf, p_yes, clipn, clip_ag]),
      "+ OWLv2 grounding (ours)":   np.column_stack([conf, p_yes, detn, det_ag])}
ALPHAS = [0.05, 0.10, 0.15, 0.20]
DELTA, REPS = F.DELTA, F.REPS

cov = {k: {a: [] for a in ALPHAS} for k in SC}
rte = {k: {a: [] for a in ALPHAS} for k in SC}
ral = {k: {a: [] for a in ALPHAS} for k in SC}
lam_ = {k: {a: [] for a in ALPHAS} for k in SC}

for tr, cal, te in F.SPLITS(n, REPS):
    if len(np.unique(correct[tr])) < 2: continue
    for name, X in SC.items():
        s = LogisticRegression(max_iter=1000).fit(X[tr], correct[tr]).predict_proba(X)[:, 1]
        for a in ALPHAS:
            lam = F.fst(s[cal], correct[cal], a)
            lam_[name][a].append(lam)
            if lam is None:
                cov[name][a].append(0.0); rte[name][a].append(None); ral[name][a].append(None)
                continue
            thr = np.quantile(s[cal], 1 - lam)
            m = s[te] >= thr; k = int(m.sum())
            cov[name][a].append(k / len(te))
            rte[name][a].append(float((1 - correct[te][m]).mean()) if k else None)
            ma = s >= thr
            ral[name][a].append(float((1 - correct[ma]).mean()) if ma.sum() else None)

print(f"Certified coverage (%) at each risk target alpha  (POPE, LLaVA-1.5-7B, "
      f"{REPS} paired three-way splits, mean+-sd over splits)\n")
print(f"{'score':26s}" + "".join(f"alpha={a:.0%}".rjust(16) for a in ALPHAS))
for name in SC:
    row = name.ljust(26)
    for a in ALPHAS:
        v = np.array(cov[name][a]) * 100
        row += f"{v.mean():5.1f}+-{v.std(ddof=1):4.1f}".rjust(16)
    print(row)

print(f"\nk_min per alpha (Clopper-Pearson, delta={DELTA:.2f}): " +
      "  ".join(f"a={a:.2f}: {F.kmin_for(F.cp_upper, a, DELTA)}" for a in ALPHAS))

print(f"\nAbort rate (%) -- fraction of splits certifying nothing (coverage 0%)\n")
print(f"{'score':26s}" + "".join(f"alpha={a:.0%}".rjust(12) for a in ALPHAS))
for name in SC:
    print(name.ljust(26) + "".join(
        f"{F.abort_rate(lam_[name][a])*100:11.1f}%" for a in ALPHAS))

print(f"\nValidity audit: exceedance = fraction of splits with realised risk > alpha, "
      f"vs delta={DELTA:.2f}\n(test fold / re-scored on all {n} items).  "
      f"Mean risk is NOT the guarantee.\n")
print(f"{'score':26s}" + "".join(f"alpha={a:.0%}".rjust(18) for a in ALPHAS))
for name in SC:
    row = name.ljust(26)
    for a in ALPHAS:
        row += f"{F.exceedance(rte[name][a],a)*100:6.1f}% /{F.exceedance(ral[name][a],a)*100:6.1f}%".rjust(18)
    print(row)
print(f"  exc(te)=test fold (n_te={n-2*(n//3)}, unbiased/noisy)   "
      f"exc(all)=all {n} items (low-noise proxy)")

print(f"\nPaired gains over 'Confidence only' (identical splits, two-sided tests)")
print(F.Gain.header(30))
for name in ["+ CLIP grounding", "+ OWLv2 grounding (ours)"]:
    for a in ALPHAS:
        print(F.gain_stats(np.array(cov["Confidence only"][a]) * 100,
                           np.array(cov[name][a]) * 100,
                           f"a={a:.2f} {name}").line(30))

# ------------------------------------------------------------------- plot
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
COL = {"Confidence only": "#9aa3b2", "+ CLIP grounding": "#e0a458",
       "+ OWLv2 grounding (ours)": "#4b57c8"}
fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=150)
for name in SC:
    m = [np.mean(cov[name][a]) * 100 for a in ALPHAS]
    sd = [np.std(cov[name][a], ddof=1) * 100 for a in ALPHAS]
    ax.errorbar([a * 100 for a in ALPHAS], m, yerr=sd, marker="o",
                lw=2.4 if "ours" in name else 1.8, capsize=3, color=COL[name], label=name)
ax.set_xlabel(r"Risk target $\alpha$ (%)")
ax.set_ylabel("Certified coverage (%)")
ax.set_title("Coverage bought at each guarantee level (POPE)", fontweight="bold", loc="left")
ax.set_ylim(bottom=0)
ax.grid(alpha=.2)
ax.legend(frameon=False, fontsize=10, loc="lower right")
ax.annotate(f"fixed-sequence protocol, {REPS} paired splits;\nbars are 1 sd over splits",
            xy=(0.02, 0.97), xycoords="axes fraction", va="top", ha="left",
            fontsize=8.5, color="#5b6577")
for s in ["top", "right"]: ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(_here, "coverage_vs_alpha.png"), bbox_inches="tight",
            facecolor="white")
print("\nsaved coverage_vs_alpha.png")
