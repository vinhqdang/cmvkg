"""
Figures for the TMLR manuscript.
  fig_method.png       : CCRC decision regions in the (s, m) plane  (schematic)
  fig_monotonicity.png : paired matched-prefix downstream-hallucination result
  fig_precondition.png : coverage gain vs the abstention floor (mu - alpha)
  fig_dilution.png     : leverage -- coverage gain vs repaired mass
"""
import json, os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from scipy.stats import wilcoxon, ttest_rel

_here = os.path.dirname(os.path.abspath(__file__))
INK, MUT, IND, OK, BAD, AMB = "#1a1f2b", "#5b6577", "#4b57c8", "#1f7d5c", "#c8384f", "#d9a441"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5, "text.color": INK,
                     "axes.edgecolor": "#c9cfd8", "axes.labelcolor": INK,
                     "xtick.color": MUT, "ytick.color": MUT})

# ---------------------------------------------------------------- fig 1: method
fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=200)
tau, q = 0.62, 0.72
ax.add_patch(Rectangle((tau, 0), 1 - tau, 1, color=OK, alpha=.16, lw=0))
ax.add_patch(Rectangle((0, q), tau, 1 - q, color=IND, alpha=.20, lw=0))
ax.add_patch(Rectangle((0, 0), tau, q, color=MUT, alpha=.10, lw=0))
ax.axvline(tau, color=OK, lw=1.8, ls="-")
ax.plot([0, tau], [q, q], color=IND, lw=1.8)
ax.text(0.80, 0.50, "ACCEPT\nmodel's answer", ha="center", va="center",
        fontsize=11, fontweight="bold", color=OK)
ax.text(0.31, 0.86, "REPAIR\nchannel-2 answer", ha="center", va="center",
        fontsize=11, fontweight="bold", color=IND)
ax.text(0.31, 0.34, "ABSTAIN", ha="center", va="center",
        fontsize=11, fontweight="bold", color=MUT)
ax.text(tau + .012, 0.035, r"$Q_s(1-\lambda)$  (calibrated)", color=OK, fontsize=9.5, rotation=90)
ax.text(0.015, q + .022, r"$Q_m(1-q)$   (fixed, strict)", color=IND, fontsize=9.5)
ax.set_xlabel(r"channel 1: correctness score $s(x)$  $\rightarrow$")
ax.set_ylabel(r"channel 2: evidence margin $m(x)$  $\rightarrow$")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_xticks([]); ax.set_yticks([])
ax.set_title("CCRC decision regions", fontweight="bold", loc="left", fontsize=12)
for s in ["top", "right"]: ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(_here, "fig_method.png"),
                                bbox_inches="tight", facecolor="white")
print("fig_method.png")

# ---------------------------------------------------------------- fig 2: monotonicity
# LAYOUT FIX: panel A's single-line title was ~3.9in of text over a ~3.6in axes at
# fontsize 11 with loc="left", so it ran into panel B's title. Both titles are now
# two lines at a smaller size over a wider figure with explicit wspace, and the
# p-value is computed from the data instead of being hardcoded.
d = json.load(open(os.path.join(_here, "exp14_monotonicity.json")))
ho = np.array([x["n_hallu_orig"] for x in d], float)
hr = np.array([x["n_hallu_rep"] for x in d], float)
diff = hr - ho; n = len(d)
p_wil = float(wilcoxon(hr, ho, alternative="two-sided").pvalue)
p_tt = float(ttest_rel(hr, ho).pvalue)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 4.3), dpi=200,
                             gridspec_kw={"width_ratios": [1, 1.15], "wspace": .30})
m = [ho.mean(), hr.mean()]
se = [ho.std(ddof=1) / np.sqrt(n), hr.std(ddof=1) / np.sqrt(n)]
a1.bar([0, 1], m, yerr=se, capsize=6, width=.55,
       color=[MUT, BAD], edgecolor="white", lw=1.5)
a1.set_xticks([0, 1]); a1.set_xticklabels(["keep\nhallucination", "repair\n(substitute)"])
a1.set_ylabel("downstream hallucinated mentions")
a1.set_title("A · Continuation quality after intervention\n"
             f"     ({n} matched-prefix cases)",
             fontweight="bold", loc="left", fontsize=10.5)
a1.set_ylim(0, max(m) * 1.28)
a1.grid(alpha=.2, axis="y")
for i, v in enumerate(m):
    a1.text(i, v + se[i] + .05, f"{v:.3f}", ha="center", fontweight="bold")
for s in ["top", "right"]: a1.spines[s].set_visible(False)

bins = np.arange(diff.min() - .5, diff.max() + 1.5, 1.0)
a2.hist(diff, bins=bins, color=IND, alpha=.75, edgecolor="white")
a2.axvline(0, color=MUT, lw=1.2, ls="--")
a2.axvline(diff.mean(), color=BAD, lw=2.2)
a2.set_ylim(0, a2.get_ylim()[1] * 1.18)
a2.annotate(f"mean {diff.mean():+.3f}\nWilcoxon p={p_wil:.3f}\npaired t p={p_tt:.3f}",
            xy=(diff.mean(), a2.get_ylim()[1] * .96), xytext=(10, 0),
            textcoords="offset points", color=BAD, fontsize=9.0,
            fontweight="bold", va="top", ha="left", linespacing=1.3)
a2.set_xlabel("paired difference (repaired − original)")
a2.set_ylabel("cases")
a2.set_title("B · Repair makes the continuation worse\n"
             "     (absolute hallucinated mentions)",
             fontweight="bold", loc="left", fontsize=10.5)
a2.grid(alpha=.2, axis="y")
for s in ["top", "right"]: a2.spines[s].set_visible(False)
fig.savefig(os.path.join(_here, "fig_monotonicity.png"),
            bbox_inches="tight", facecolor="white")
print(f"fig_monotonicity.png   (mean {diff.mean():+.4f}, Wilcoxon p={p_wil:.4f}, "
      f"paired t p={p_tt:.4f}, n={n})")

# ------------------------------------------------- fig 3: precondition & leverage
# Numbers refreshed to the CANONICAL protocol at alpha=0.10 (ccrc_gains_stats.py,
# 400 paired splits): (name, mu %, certified coverage gain pp, repaired mass %,
# dilution-only part of the gain pp).
# The old hardcoded values were from the leaky max-over-thresholds rule at 60 reps
# and every one of them was more favourable than the data supports.
settings = [("POPE-adv LLaVA", 17.3, 3.19, 1.74, 1.84),
            ("POPE-1500 LLaVA", 18.1, 3.18, 1.82, 0.86),
            ("POPE-adv VCD", 19.6, 2.15, 0.97, 1.08),
            ("POPE-adv Qwen", 12.9, 1.60, 1.02, 1.01),
            ("AMBER(d)", 11.4, -10.20, 1.97, 0.20)]
alpha = 10.0
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 4.3), dpi=200,
                             gridspec_kw={"wspace": .28})
x = [s[1] - alpha for s in settings]; y = [s[2] for s in settings]
cols = [IND if v > 0 else BAD for v in y]
a1.axhline(0, color=MUT, lw=1)
a1.scatter(x, y, s=110, c=cols, edgecolor="white", zorder=3, lw=1.5)
for (nm, mu, g, rp, dil), xi, yi in zip(settings, x, y):
    a1.annotate(nm, (xi, yi), textcoords="offset points",
                xytext=(8, 6 if yi > 0 else -14), fontsize=8.6, color=MUT)
a1.set_xlabel(r"abstention floor headroom  $\mu-\alpha$  (points)")
a1.set_ylabel("certified coverage gain (pp)")
a1.set_title(r"A · Gain against headroom $\mu-\alpha$" "\n"
             r"     (a diagnostic, not a prediction)",
             fontweight="bold", loc="left", fontsize=10.5)
a1.grid(alpha=.2); a1.set_xlim(0, 11)
for s in ["top", "right"]: a1.spines[s].set_visible(False)

# Panel B: the gain SPLIT into the repaired items themselves (which are emitted
# automatically, whether or not lambda-hat moves) and the part attributable to
# dilution (lambda-hat actually moving). Only the second is leverage.
msk = [s for s in settings if s[2] > 0]
rp = np.array([s[3] for s in msk]); gg = np.array([s[2] for s in msk])
dl = np.array([s[4] for s in msk])
mult = gg / rp
a2.bar(np.arange(len(msk)) - .17, rp, width=.32, color=MUT, alpha=.55,
       edgecolor="white", label="(i) repaired items emitted")
a2.bar(np.arange(len(msk)) + .17, dl, width=.32, color=IND,
       edgecolor="white", label="(ii) dilution ($\\hat\\lambda$ moves)")
for i, (nm, mu, g, r_, d_) in enumerate(msk):
    a2.text(i, max(r_, d_) + .12, f"{mult[i]:.2f}×", ha="center", fontsize=8.6,
            color=INK, fontweight="bold")
a2.set_xticks(np.arange(len(msk)))
a2.set_xticklabels([s[0].replace("POPE-adv ", "").replace("POPE-1500 ", "POPE-1500\n")
                    for s in msk], fontsize=8.4)
a2.set_ylabel("contribution to the gain (pp)")
a2.set_ylim(0, max(rp.max(), dl.max()) * 1.32)
a2.set_title(f"B · Leverage decomposed: {mult.min():.1f}–{mult.max():.1f}× overall,\n"
             f"     of which dilution only +{dl.min():.2f} to +{dl.max():.2f} pp",
             fontweight="bold", loc="left", fontsize=10.5)
a2.grid(alpha=.2, axis="y")
a2.legend(frameon=False, fontsize=8.4, loc="upper left")
for s in ["top", "right"]: a2.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(_here, "fig_precondition.png"),
                                bbox_inches="tight", facecolor="white")
print("fig_precondition.png")
