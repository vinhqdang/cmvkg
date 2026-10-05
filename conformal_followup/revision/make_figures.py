"""Figures of the revised manuscript (matplotlib). Writes ../manuscript_revised/*.png"""
import json, os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
H = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(H, "..", "manuscript_revised")
L = lambda f: json.load(open(os.path.join(H, f)))
main, boot = L("results_main.json"), L("results_bootstrap.json")
NAMES = ["POPE-1500 LLaVA", "POPE-3000 LLaVA", "POPE-adv LLaVA", "POPE-adv Qwen2-VL", "POPE-adv LLaVA+VCD", "AMBER(d) LLaVA"]
C = {"ext": "#0072B2", "fitsel": "#D55E00", "filter": "#444444", "union": "#009E73"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

# ---- Fig 1: forest plot of certified gains with image-bootstrap intervals
fig, ax = plt.subplots(figsize=(7.2, 5.2)); y = 0; yt, yl = [], []
for n in NAMES:
    for a in ("0.1", "0.15"):
        b = boot[f"{n}|{a}"]
        for arm, off in (("ext", -0.17), ("fitsel", 0.17)):
            g = main[f"{n}|all|{a}"][arm]["gain"]; lo, hi = b[arm]["gain"][1:]
            ax.plot([lo, hi], [y + off, y + off], color=C[arm], lw=2, solid_capstyle="butt")
            ax.plot(g, y + off, "o", color=C[arm], ms=4.5)
        yt.append(y); yl.append(f"{n.replace('POPE-adv','POPE-adv,').replace('POPE-1500','POPE-1500,').replace('POPE-3000','POPE-3000,').replace('AMBER(d)','AMBER(d),')}  $\\alpha$={float(a):.2f}"); y -= 1
    y -= 0.35
ax.axvline(0, color="k", lw=0.8); ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=7.5)
ax.set_xlabel("certified-coverage gain over filtering (points); bar = 95% image-bootstrap interval")
ax.plot([], [], "o-", color=C["ext"], label="gate fixed on an external benchmark"); ax.plot([], [], "o-", color=C["fitsel"], label="gate chosen on the fit fold")
ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=2, fontsize=8); fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_forest.png"), dpi=200); plt.close(fig)

# ---- Fig 2: certified coverage against alpha, POPE-1500
fig, ax = plt.subplots(figsize=(4.8, 3.3)); al = [0.05, 0.10, 0.15, 0.20]
for arm, lab in (("filter", "filtering"), ("ext", "CCRC, external gate"), ("fitsel", "CCRC, fit-fold gate"), ("union", "CCRC, union of 3 gates")):
    m = [main[f"POPE-3000 LLaVA|all|{a}"][arm]["cov"] * 100 for a in al]; s = [main[f"POPE-3000 LLaVA|all|{a}"][arm]["cov_sd"] * 100 for a in al]
    ax.errorbar(al, m, yerr=s, marker="o", ms=4, capsize=2, color=C[arm], label=lab, lw=1.4)
ax.set_xlabel("risk target $\\alpha$"); ax.set_ylabel("certified coverage (%)"); ax.legend(frameon=False, fontsize=7.5)
ax.set_xticks(al); fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_alpha.png"), dpi=200); plt.close(fig)

# ---- Fig 3: validity audit of the synthetic study
if os.path.exists(os.path.join(H, "results_validity.json")):
    V = L("results_validity.json"); sig = {0.0: 0.0, 1.0: 0.08, 2.0: 0.26}
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True)
    for ax, ncal in zip(axs, (150, 450)):
        for (proc, arm, calib), col, ls, lab in [(("value", "repair", "full"), "#0072B2", "-", "value grid, repair"), (("rank", "repair", "full"), "#D55E00", "--", "rank-indexed, repair"),
                                                 (("value", "repair", "single"), "#009E73", ":", "value grid, 1 item/image")]:
            xs, ys = [], []
            for sg_, icc in sig.items():
                if calib == "single" and sg_ == 0.0: continue
                v = [V[f"{mu}|{al}|{ncal}|{sg_}|{proc}|{arm}|{calib}"]["exc_incl_abort"] for mu in (0.15, 0.25) for al in (0.10, 0.15)]
                xs.append(icc); ys.append(max(v))
            ax.plot(xs, ys, marker="o", ms=4, color=col, ls=ls, label=lab)
        ax.axhline(0.10, color="k", lw=0.8); ax.text(0.0, 0.103, "$\\delta$", fontsize=8); ax.set_title(f"$n_{{cal}}$={ncal}", fontsize=9); ax.set_xlabel("intra-image correlation of errors")
    axs[0].set_ylabel("worst-case exceedance $\\Pr[\\mathrm{Risk}>\\alpha]$"); axs[1].legend(frameon=False, fontsize=7.5)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_validity.png"), dpi=200); plt.close(fig)
print("figures written")
