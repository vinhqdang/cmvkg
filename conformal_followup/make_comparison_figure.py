"""
Comparison figure (Figure 8, fig:comparison). Writes comparison_figure.png.

Panel A: risk-coverage curves (real, POPE/LLaVA-1.5-7B, PAIRED THREE-WAY DISJOINT
         splits -- combiner fit on a train fold, threshold calibrated on a second
         fold, curve measured on a third held-out fold) for confidence vs +CLIP vs
         +OWLv2 structured grounding.
         Two coverage numbers appear per arm and must not be confused: ORACLE
         coverage read off the test-fold curve (peeks at test labels, optimistic)
         and CALIBRATED coverage (what the procedure certifies, comparable to the
         tables). Both are labelled in the legend and printed to stdout.
Panel B: capability matrix across recent hallucination methods (factual, sourced
         from each paper).

PROTOCOL (canonical, see canonical_fst.py). The calibrated arm previously used an
exhaustive max-over-thresholds search; it now uses the canonical ascending-lambda
fixed sequence (stop at first non-rejection, return last passing index).

LAYOUT FIX. Panel B's column headers previously collided illegibly: the row-label
gutter consumed 4 of the 9.2 x-units, leaving each of the 5 columns about 11% of
the axis width -- narrower than the header text. The gutter is now 2.9 units,
headers wrap one word per line at a smaller size, and there is explicit headroom
above the grid, so nothing overlaps or clips.
"""
import json, os, numpy as np, warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
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
SC = {"confidence only (model logits)": np.column_stack([conf, p_yes]),
      "+ CLIP grounding":               np.column_stack([conf, p_yes, clipn, clip_ag]),
      "+ OWLv2 grounding (ours)":       np.column_stack([conf, p_yes, detn, det_ag])}
GRID = np.linspace(.02, 1, 99)
ALPHA, DELTA, REPS = .10, F.DELTA, F.REPS


def rc(s, c):
    c = c[np.argsort(-s)]; cov = np.arange(1, len(c) + 1) / len(c)
    return np.interp(GRID, cov, np.cumsum(1 - c) / np.arange(1, len(c) + 1))


A = {k: dict(curve=[], oracle=[], cov=[], rte=[], ral=[], lam=[]) for k in SC}
for tr, cal, te in F.SPLITS(n, REPS):
    if len(np.unique(correct[tr])) < 2: continue
    for k, X in SC.items():
        s = LogisticRegression(max_iter=1000).fit(X[tr], correct[tr]).predict_proba(X)[:, 1]
        S = A[k]
        curve = rc(s[te], correct[te]); S["curve"].append(curve)
        b = GRID[curve <= ALPHA]; S["oracle"].append(b.max() if len(b) else 0.0)
        lam = F.fst(s[cal], correct[cal], ALPHA)
        S["lam"].append(lam)
        if lam is None:
            S["cov"].append(0.0); S["rte"].append(None); S["ral"].append(None); continue
        thr = np.quantile(s[cal], 1 - lam)
        m = s[te] >= thr; kk = int(m.sum())
        S["cov"].append(kk / len(te))
        S["rte"].append(float((1 - correct[te][m]).mean()) if kk else None)
        ma = s >= thr
        S["ral"].append(float((1 - correct[ma]).mean()) if ma.sum() else None)

# ------------------------------------------------------------------ figure
INK, MUT = "#1a1f2b", "#5b6577"
COL = {"confidence only (model logits)": "#9aa3b2", "+ CLIP grounding": "#e0a458",
       "+ OWLv2 grounding (ours)": "#4b57c8"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "text.color": INK,
                     "axes.edgecolor": "#c9cfd8", "axes.labelcolor": INK,
                     "xtick.color": MUT, "ytick.color": MUT})
fig = plt.figure(figsize=(16.0, 6.9), dpi=150)
gs = GridSpec(1, 2, width_ratios=[1.0, 1.15], wspace=.20)

# ---- Panel A
axA = fig.add_subplot(gs[0])
for k in SC:
    S = A[k]; M = np.array(S["curve"]); m = M.mean(0); sd = M.std(0, ddof=1)
    axA.fill_between(GRID, m - sd, m + sd, color=COL[k], alpha=.13, lw=0)
    axA.plot(GRID, m, color=COL[k], lw=2.8 if "ours" in k else 2.0,
             label=f"{k}\n     oracle cov@10% = {np.mean(S['oracle'])*100:.0f}%  ·  "
                   f"calibrated = {np.mean(S['cov'])*100:.0f}%")
axA.axhline(ALPHA, ls="--", lw=1.2, color="#c8384f")
axA.text(.03, ALPHA + .006, "target risk = 10%", color="#c8384f", fontsize=9.5)
axA.set_xlabel("Coverage - fraction of questions answered")
axA.set_ylabel("Risk - error rate among answered")
axA.set_title("A · Selective risk-coverage · POPE / LLaVA-1.5-7B\n"
              f"     {REPS} paired three-way disjoint splits (fit / calibrate / test)",
              fontweight="bold", loc="left", fontsize=12, pad=12)
axA.set_xlim(0, 1); axA.set_ylim(0, .30); axA.grid(alpha=.2)
axA.legend(loc="upper left", frameon=False, fontsize=9.3, labelspacing=.9)
for s in ["top", "right"]: axA.spines[s].set_visible(False)

# ---- Panel B: capability matrix
axB = fig.add_subplot(gs[1]); axB.axis("off")
methods = ["OPERA  (CVPR'24)", "Attention Lens  (CVPR'25)", "REVERSE  (NeurIPS'25)",
           "ConfLVLM  (EMNLP'25)", "Online abstention  (2025)", "BCEA  (2026, concurrent)",
           "Ours  (CCRC)"]
# one word per line: each column gets ~13% of the axis width, so multi-word lines
# are what used to collide.
caps = ["Training\nfree", "External\nvisual\nevidence", "Changes\nthe\nanswer",
        "Statistical\nguarantee", "Selective\ncoverage"]
M = np.array([
    [1, 0, 1, 0, 0],        # OPERA             (decoding-time; no external evidence)
    [1, 0, 1, 0, 0],        # Attention Lens    (internal attention only)
    [0, 0.5, 1, 0, 0.5],    # REVERSE           (needs finetuning; rejection sampling ~ partial)
    [1, 1, 0, 1, 1],        # ConfLVLM          (CLIP/BiomedCLIP scorer IS external evidence)
    [1, 0, 0, 1, 1],        # Online abstention (internal signals; abstain only)
    [1, 1, 0, 1, 1],        # BCEA              (acquires zoomed views; does NOT change answer)
    [1, 1, 1, 1, 1],        # Ours              (external detector; substitutes the answer)
])
nR, nC = M.shape
GUT = 2.9                                        # row-label gutter, in x-units
for i in range(nR):
    for j in range(nC):
        v = M[i, j]
        c = "#2f8f6b" if v == 1 else ("#d9a441" if v == 0.5 else "#dfe3e9")
        sym = "✓" if v == 1 else ("~" if v == 0.5 else "–")
        axB.add_patch(plt.Rectangle((j + .03, nR - 1 - i + .03), .94, .94,
                                    color=c, ec="white", lw=2))
        axB.text(j + .5, nR - 1 - i + .5, sym, ha="center", va="center",
                 color="white" if v else "#9aa3b2", fontsize=15, fontweight="bold")
for j, cap in enumerate(caps):
    axB.text(j + .5, nR + .12, cap, ha="center", va="bottom", fontsize=8.0, color=INK,
             fontweight="bold", linespacing=1.15)
for i, mth in enumerate(methods):
    fw = "bold" if "Ours" in mth else "normal"
    axB.text(-.14, nR - 1 - i + .5, mth, ha="right", va="center", fontsize=9.0,
             color=INK, fontweight=fw)
axB.set_xlim(-GUT, nC + .1)
axB.set_ylim(-0.95, nR + 1.35)                   # headroom for 3-line headers
axB.set_title("B · Capability vs recent methods", fontweight="bold", loc="left",
              fontsize=12, pad=10, x=0)
axB.text(-GUT + .05, -0.28,
         "✓ full    ~ partial    – none\n"
         "Mitigation methods (OPERA / Attention Lens / REVERSE) operate only at full "
         "coverage; ours adds a\ndistribution-free guarantee + coverage control and "
         "composes on top of them.",
         fontsize=8.2, color=MUT, va="top", linespacing=1.35)

fig.subplots_adjust(left=.05, right=.985, top=.88, bottom=.10)
out = os.path.join(_here, "comparison_figure.png")
fig.savefig(out, bbox_inches="tight", facecolor="white")
print("saved", out)

# ----------------------------------------------------------------- numbers
print(f"\nPanel A numbers, {REPS} paired three-way disjoint splits (POPE n={n}), "
      f"alpha={ALPHA:.2f}, delta={DELTA:.2f}:")
print(f"{'score':32s}{'AURC':>9s}{'oracleCov':>11s}{'calibCov':>16s}"
      f"{'abort':>8s}{'exc(te)':>9s}{'exc(all)':>10s}")
for k in SC:
    S = A[k]; M_ = np.array(S["curve"]); cv = np.array(S["cov"]) * 100
    print(f"{k:32s}{M_.mean(1).mean():9.4f}{np.mean(S['oracle'])*100:10.1f}%"
          f"{cv.mean():9.1f}+-{cv.std(ddof=1):4.1f}%{F.abort_rate(S['lam'])*100:7.1f}%"
          f"{F.exceedance(S['rte'],ALPHA)*100:8.1f}%{F.exceedance(S['ral'],ALPHA)*100:9.1f}%")
print("  oracleCov = coverage read off the test-fold risk-coverage curve (NOT calibrated);")
print("  calibCov  = conformally calibrated coverage (CP bound at delta), matches tables.")
print(f"  exceedance = fraction of splits with realised risk > alpha, vs delta={DELTA:.2f};")
print(f"  exc(te)=test fold n_te={n-2*(n//3)} (unbiased/noisy), exc(all)=all {n} items.")
print("  Mean realised risk is NOT the guarantee and is not reported here.")

print(f"\nPaired gains over the confidence-only arm (identical splits, two-sided tests)")
print(F.Gain.header(34))
base = A["confidence only (model logits)"]
for k in ["+ CLIP grounding", "+ OWLv2 grounding (ours)"]:
    print(F.gain_stats(np.array(base["cov"]) * 100, np.array(A[k]["cov"]) * 100,
                       f"{k} calib cov").line(34))
    print(F.gain_stats(np.array(base["oracle"]) * 100, np.array(A[k]["oracle"]) * 100,
                       f"{k} oracle cov").line(34))
