"""Ablation ladder: previous protocol -> revised protocol, one change per rung.
Writes results_ladder.json and prints a compact table."""
import json, time, numpy as np, ccrc_rev as R
S = R.load_settings()
M = R.Mode
RUNGS = [
 ("L0 conventional protocol",             M.legacy()),
 ("L1 + image-grouped folds",         M(grouping="image", grid="rank",  gate_kind="rank",  e0=0, gate="fixed", missing="drop", legacy_norm=True)),
 ("L2 + fixed value grid and gate",   M(grouping="image", grid="value", gate_kind="value", e0=0, gate="fixed", missing="drop")),
 ("L3 + analytic start e0=1",         M(grouping="image", grid="value", gate_kind="value", e0=1, gate="fixed", missing="drop")),
 ("L4 + gate chosen on fit fold",     M(grouping="image", grid="value", gate_kind="value", e0=1, gate="fitsel", missing="drop")),
 ("L5 + all items (missing policy)",  M(grouping="image", grid="value", gate_kind="value", e0=1, gate="fitsel", missing="impute")),
]
out = {}; t0 = time.time()
for name in R.MAIN:
    for rname, mode in RUNGS:
        run = R.Runner(S[name], mode)
        for a in (.10, .15):
            f = R.summarise(run.run(a, gate=None), a)
            c = R.summarise(run.run(a), a)           # mode's gate
            fixed = R.summarise(run.run(a, gate="fixed"), a) if mode.gate == "fitsel" else None
            out[f"{name}|{rname}|{a}"] = dict(filter=f, ccrc=c, fixed=fixed)
            print(f"{name:20s}{rname:34s}a={a:.2f} filt {f['cov']*100:5.1f} (ab {f['abort']*100:4.1f}) "
                  f"ccrc {c['cov']*100:5.1f} (ab {c['abort']*100:4.1f}) gain {(c['cov']-f['cov'])*100:+5.1f} "
                  f"exc {c['exc_te']:.2f} cv {c['conf_viol']:.2f}")
    print()
json.dump(out, open("results_ladder.json", "w"), indent=1)
print(f"{time.time()-t0:.0f}s")
