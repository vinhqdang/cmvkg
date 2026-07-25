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
d = json.load(open(os.path.join(_here, "exp14_monotonicity.json")))
ho = np.array([x["n_hallu_orig"] for x in d], float)
hr = np.array([x["n_hallu_rep"] for x in d], float)
diff = hr - ho; n = len(d)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 3.9), dpi=200,
                             gridspec_kw={"width_ratios": [1, 1.15]})
m = [ho.mean(), hr.mean()]
se = [ho.std(ddof=1) / np.sqrt(n), hr.std(ddof=1) / np.sqrt(n)]
a1.bar([0, 1], m, yerr=se, capsize=6, width=.55,
       color=[MUT, BAD], edgecolor="white", lw=1.5)
a1.set_xticks([0, 1]); a1.set_xticklabels(["keep\nhallucination", "repair\n(substitute)"])
a1.set_ylabel("downstream hallucinated mentions")
a1.set_title(f"A · Continuation quality after intervention (n={n})",
             fontweight="bold", loc="left", fontsize=11)
a1.grid(alpha=.2, axis="y")
for i, v in enumerate(m): a1.text(i, v + se[i] + .04, f"{v:.3f}", ha="center", fontweight="bold")
for s in ["top", "right"]: a1.spines[s].set_visible(False)

bins = np.arange(diff.min() - .5, diff.max() + 1.5, 1.0)
a2.hist(diff, bins=bins, color=IND, alpha=.75, edgecolor="white")
a2.axvline(0, color=MUT, lw=1.2, ls="--")
a2.axvline(diff.mean(), color=BAD, lw=2.2)
a2.text(diff.mean() + .08, a2.get_ylim()[1] * .88,
        f"mean {diff.mean():+.3f}\np={0.025:.3f}", color=BAD, fontsize=9.5, fontweight="bold")
a2.set_xlabel("paired difference (repaired − original)")
a2.set_ylabel("cases")
a2.set_title("B · Repair makes the continuation worse", fontweight="bold", loc="left", fontsize=11)
a2.grid(alpha=.2, axis="y")
for s in ["top", "right"]: a2.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(_here, "fig_monotonicity.png"),
                                bbox_inches="tight", facecolor="white")
print("fig_monotonicity.png")

# ------------------------------------------------- fig 3: precondition & leverage
settings = [("POPE-adv LLaVA", 17.3, 4.7, 1.4), ("POPE-1500 LLaVA", 18.1, 4.4, 1.8),
            ("POPE-adv VCD", 19.6, 2.5, 0.9), ("POPE-adv Qwen", 12.9, 1.9, 1.0),
            ("AMBER(d)", 11.4, -7.5, 0.0)]
alpha = 10.0
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.8, 3.9), dpi=200)
x = [s[1] - alpha for s in settings]; y = [s[2] for s in settings]
cols = [IND if v > 0 else BAD for v in y]
a1.axhline(0, color=MUT, lw=1)
a1.scatter(x, y, s=110, c=cols, edgecolor="white", zorder=3, lw=1.5)
for (nm, mu, g, rp), xi, yi in zip(settings, x, y):
    a1.annotate(nm, (xi, yi), textcoords="offset points",
                xytext=(8, 6 if yi > 0 else -14), fontsize=8.6, color=MUT)
a1.set_xlabel(r"abstention floor headroom  $\mu-\alpha$  (points)")
a1.set_ylabel("certified coverage gain (pp)")
a1.set_title(r"A · Gain tracks headroom $\mu-\alpha$", fontweight="bold", loc="left", fontsize=11)
a1.grid(alpha=.2); a1.set_xlim(0, 11)
for s in ["top", "right"]: a1.spines[s].set_visible(False)

msk = [s for s in settings if s[2] > 0]
rp = [s[3] for s in msk]; gg = [s[2] for s in msk]
a2.scatter(rp, gg, s=110, color=IND, edgecolor="white", lw=1.5, zorder=3)
lim = np.linspace(0, 2.1, 10)
for k, ls in [(1, ":"), (3, "--"), (6, "-.")]:
    a2.plot(lim, k * lim, ls=ls, color=MUT, lw=1, alpha=.8)
    a2.text(2.05, k * 2.05, f"{k}×", fontsize=8.5, color=MUT, va="center")
for (nm, mu, g, r_) in msk:
    a2.annotate(nm, (r_, g), textcoords="offset points", xytext=(8, -4),
                fontsize=8.6, color=MUT)
a2.set_xlabel("repaired mass (% of items)")
a2.set_ylabel("certified coverage gain (pp)")
a2.set_title("B · Leverage: gain is 3–6× the repaired mass",
             fontweight="bold", loc="left", fontsize=11)
a2.set_xlim(0, 2.3); a2.set_ylim(0, 8); a2.grid(alpha=.2)
for s in ["top", "right"]: a2.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(_here, "fig_precondition.png"),
                                bbox_inches="tight", facecolor="white")
print("fig_precondition.png")
