"""Main results under protocol P4 (image folds, fixed value grid, analytic start e0=1,
gate chosen on the fit fold, all-item cohort). Writes results_main.json.

Arms per (setting, alpha):  filter | fixed q=0.10 (NOT certified: q chosen with knowledge of the
benchmarks) | fitsel (certified at full delta; the headline CCRC) | union (certified, delta/3 per gate).
Also stores repair accounting (R1.8) and the cohort variant 'grounded only' (R1.9)."""
import json, time, numpy as np, ccrc_rev as R
S = R.load_settings(); M = R.Mode
ALPHAS = (0.05, 0.10, 0.15, 0.20)
DEV = {n: "AMBER(d) LLaVA" for n in R.MAIN if not n.startswith("AMBER")}; DEV["AMBER(d) LLaVA"] = "POPE-3000 LLaVA"
out = {}; t0 = time.time()
for name in R.MAIN:
    for cohort, mode in (("all", M()), ("grounded", M(missing="drop"))):
        run = R.Runner(S[name], mode)
        for a in ALPHAS:
            f = run.run(a, gate=None); cell = {"filter": R.summarise(f, a)}
            qext = R.external_gate_matched(S[DEV[name]], a, S[name].n // 3)      # size-matched development choice
            qfull = R.external_gate(S[DEV[name]], a)                             # whole development set as one sample
            for lab, g in (("fixed10", "fixed"), ("fitsel", "fitsel"), ("union", "union"), ("ext", "ext"), ("extfull", "extfull")):
                res = run.run(a, gate="fixed", q=qext) if g == "ext" else run.run(a, gate="fixed", q=qfull) if g == "extfull" else run.run(a, gate=g)
                s = R.summarise(res, a); gm, gs = R.paired_gain(f, res)
                s.update(gain=gm, gain_sd=gs)
                ok = [r for r in res if not r["abort"]]
                T = {k: int(sum(r[k] for r in ok)) for k in R.KEYS}; T["n"] = int(sum(r["n"] for r in ok))
                s["acct"] = T
                if g == "ext": s["q_ext"] = qext
                if g == "extfull": s["q_ext"] = qfull
                if g == "fitsel":
                    qs = [r["q"] for r in res]
                    s["q_dist"] = {("filter" if v is None else str(v)): float(np.mean([(x is None) if v is None else (x is not None and np.isclose(x, v)) for x in qs]))
                                   for v in (None, .05, .10, .25, .50)}
                cell[lab] = s
            out[f"{name}|{cohort}|{a}"] = cell
        print(f"{name:20s}{cohort:9s} done +{time.time()-t0:.0f}s", flush=True)
json.dump(out, open("results_main.json", "w"), indent=1)
