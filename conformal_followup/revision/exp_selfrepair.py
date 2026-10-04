"""Self-repair (negating the model's own answer on the bottom-q of the score) under P4, at both starts
(e0=0: the previous start; e0=1: the analytic start). Writes results_selfrepair.json."""
import json, numpy as np, ccrc_rev as R
S = R.load_settings(); out = {}
for name in R.MAIN:
    for e0 in (0, 1):
        run = R.Runner(S[name], R.Mode(e0=e0))
        for a in (0.10, 0.15):
            f = run.run(a, gate=None); sf = R.summarise(f, a)
            row = dict(filter=sf)
            for q in (0.02, 0.05, 0.10):
                res = run.run(a, gate="fixed", q=q, rule="self"); sc = R.summarise(res, a); gm, gs = R.paired_gain(f, res)
                ok = [r for r in res if not r["abort"]]; kr = sum(r["k_rep"] for r in ok); er = sum(r["e_rep"] for r in ok)
                row[f"self{q}"] = dict(sc, gain=gm, gain_sd=gs, flip_region_risk=(er / kr if kr else None))
            for q in (0.10,):
                res = run.run(a, gate="fixed", q=q, rule="margin"); sc = R.summarise(res, a); gm, gs = R.paired_gain(f, res)
                row["ccrc10"] = dict(sc, gain=gm, gain_sd=gs)
            out[f"{name}|e0={e0}|{a}"] = row
    print(name, flush=True)
json.dump(out, open("results_selfrepair.json", "w"), indent=1)
