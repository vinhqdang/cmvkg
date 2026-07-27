"""Regenerate fig_precondition.png so that panel B's title no longer asserts a leverage
multiplier (review blocker 6) and panel A's title no longer asserts that gain tracks headroom
(blocker 9), with n=5 stated and the overlapping point labels separated.

Values are the canonical alpha=0.10 numbers from ccrc_gains_stats.py at 400 paired splits, i.e.
identical to the hardcoded table in make_paper_figures.py:
    (name, mu %, unconditional gain pp, repaired mass %, dilution-only part pp)
Term (i) / term (ii) bars use the conditional decomposition of CANONICAL_NUMBERS.md section 13.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "/home/user/cmvkg/conformal_followup/manuscript_tmlr/fig_precondition.png"
INK, MUT, IND, BAD = "#1a1f2b", "#5b6577", "#4b57c8", "#c8384f"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": MUT,
                     "axes.labelcolor": INK, "text.color": INK,
                     "xtick.color": MUT, "ytick.color": MUT, "font.size": 10})

# name, mu(%), unconditional gain(pp), term (i) repairs(pp), term (ii) dilution(pp), label offset
settings = [("POPE-adv LLaVA",  17.3,  3.19, 1.74, 1.84, (-64, 10)),
            ("POPE-1500 LLaVA", 18.1,  3.18, 1.82, 0.86, (8, 8)),
            ("POPE-adv VCD",    19.6,  2.15, 0.97, 1.08, (8, 6)),
            ("POPE-adv Qwen",   12.9,  1.60, 1.02, 1.01, (8, 6)),
            ("AMBER(d)",        11.4, -10.20, 1.97, 0.20, (8, -14))]
alpha = 10.0

fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 4.3), dpi=200,
                             gridspec_kw={"wspace": .28})

x = [s[1] - alpha for s in settings]
y = [s[2] for s in settings]
cols = [IND if v > 0 else BAD for v in y]
a1.axhline(0, color=MUT, lw=1)
a1.scatter(x, y, s=110, c=cols, edgecolor="white", zorder=3, lw=1.5)
for (nm, mu, g, rp, dil, off), xi, yi in zip(settings, x, y):
    a1.annotate(nm, (xi, yi), textcoords="offset points", xytext=off,
                fontsize=8.6, color=MUT)
a1.set_xlabel(r"abstention floor headroom  $\mu-\alpha$  (points)")
a1.set_ylabel("certified coverage gain (pp)")
a1.set_title("A · Gain against headroom  " r"$\mu-\alpha$" "  ($n=5$ settings)\n"
             "     a diagnostic, not a prediction",
             fontweight="bold", loc="left", fontsize=10.5)
a1.grid(alpha=.2)
a1.set_xlim(0, 11)
for s in ["top", "right"]:
    a1.spines[s].set_visible(False)

msk = [s for s in settings if s[2] > 0]
rp = np.array([s[3] for s in msk])
dl = np.array([s[4] for s in msk])
a2.bar(np.arange(len(msk)) - .17, rp, width=.32, color=MUT, alpha=.55,
       edgecolor="white", label="(i) repaired items emitted (automatic)")
a2.bar(np.arange(len(msk)) + .17, dl, width=.32, color=IND,
       edgecolor="white", label="(ii) dilution ($\\hat\\lambda$ moves)")
a2.set_xticks(np.arange(len(msk)))
a2.set_xticklabels([s[0].replace("POPE-adv ", "POPE-adv\n").replace("POPE-1500 ", "POPE-1500\n")
                    for s in msk], fontsize=8.4)
a2.set_ylabel("contribution to the gain (pp)")
a2.set_ylim(0, max(rp.max(), dl.max()) * 1.34)
a2.set_title("B · The gain decomposed: repaired mass is emitted\n"
             "     automatically; dilution adds +0.20 to +1.84 pp",
             fontweight="bold", loc="left", fontsize=10.5)
a2.grid(alpha=.2, axis="y")
a2.legend(frameon=False, fontsize=8.4, loc="upper left")
for s in ["top", "right"]:
    a2.spines[s].set_visible(False)

fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight", facecolor="white")
print("wrote", OUT)
