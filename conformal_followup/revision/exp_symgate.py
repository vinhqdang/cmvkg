"""One-directional vs polarity-symmetric repair gates under P4 (R1.8), with the error-type breakdown:
how many of the model's false-positive and false-negative errors does each rule actually correct?
Writes results_symgate.json."""
import json, numpy as np, ccrc_rev as R
S = R.load_settings(); out = {}
for name in R.MAIN:
    run = R.Runner(S[name], R.Mode())
    for a in (0.10, 0.15):
        f = run.run(a, gate=None); row = dict(filter=R.summarise(f, a))
        for rule in ("margin", "pos", "neg", "sym"):
            for lab, g, q in (("q10", "fixed", 0.10), ("fitsel", "fitsel", None)):
                res = run.run(a, gate=g, q=q, rule=rule); sc = R.summarise(res, a); gm, gs = R.paired_gain(f, res)
                ok = [r for r in res if not r["abort"]]; T = {k: int(sum(r[k] for r in ok)) for k in R.KEYS}; T["n"] = int(sum(r["n"] for r in ok))
                row[f"{rule}|{lab}"] = dict(sc, gain=gm, gain_sd=gs, acct=T)
        out[f"{name}|{a}"] = row
    print(name, flush=True)
json.dump(out, open("results_symgate.json", "w"), indent=1)
