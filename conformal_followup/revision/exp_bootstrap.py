"""Outer image-level bootstrap (R1.10). Each replicate resamples IMAGES with replacement, then runs R
image-grouped splits of the whole protocol; the spread over replicates is uncertainty about the dataset,
not about the split. Writes results_bootstrap.json."""
import json, sys, time, numpy as np, ccrc_rev as R
B = int(sys.argv[1]) if len(sys.argv) > 1 else 200
RS = int(sys.argv[2]) if len(sys.argv) > 2 else 20
S = R.load_settings(); M = R.Mode; out = {}; t0 = time.time()
for name in R.MAIN:
    rng = np.random.default_rng(12345)
    DEV = "POPE-1500 LLaVA" if name == "AMBER(d) LLaVA" else "AMBER(d) LLaVA"
    qext = {a: R.external_gate(S[DEV], a) for a in (0.10, 0.15)}
    ARMS = ("fitsel", "fixed10", "ext")
    rec = {a: dict(filt=[], ab_f=[], **{k: dict(cov=[], gain=[], ab=[]) for k in ARMS}) for a in (0.10, 0.15)}
    for b in range(B):
        T = R.resample_images(S[name], rng)
        run = R.Runner(T, M(), reps=RS, seed=b)
        for a in rec:
            f = run.run(a, gate=None)
            if not f: continue
            sf = R.summarise(f, a); rec[a]["filt"].append(sf["cov"] * 100); rec[a]["ab_f"].append(sf["abort"])
            for k in ARMS:
                res = run.run(a, gate="fitsel") if k == "fitsel" else run.run(a, gate="fixed", q=(0.10 if k == "fixed10" else qext[a]))
                sc = R.summarise(res, a)
                rec[a][k]["cov"].append(sc["cov"] * 100); rec[a][k]["gain"].append((sc["cov"] - sf["cov"]) * 100); rec[a][k]["ab"].append(sc["abort"])
    pc = lambda v: [float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    for a, r in rec.items():
        cell = dict(B=len(r["filt"]), filt=pc(r["filt"]), ab_f=float(np.mean(r["ab_f"])), q_ext=qext[a])
        for k in ARMS:
            g = np.array(r[k]["gain"])
            cell[k] = dict(gain=pc(g), frac_pos=float((g > 0).mean()), cov=pc(r[k]["cov"]), ab=float(np.mean(r[k]["ab"])))
        out[f"{name}|{a}"] = cell
    print(f"{name} +{time.time()-t0:.0f}s", flush=True)
json.dump(out, open("results_bootstrap.json", "w"), indent=1)
