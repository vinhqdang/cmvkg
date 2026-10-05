"""Additional analyses requested in review: (A) detector-free channel 1, (B) a delta-spending multi-start search,
(C) cross-dataset transfer (score, grid and gate fitted on another benchmark). Writes results_review2.json."""
import json, time, numpy as np, ccrc_rev as R
S = R.load_settings(); M = R.Mode(); out = {}; t0 = time.time()
CELLS = R.MAIN
DEV = {n: ("AMBER(d) LLaVA" if not n.startswith("AMBER") else "POPE-3000 LLaVA") for n in CELLS}
Xdf = lambda s: np.column_stack([s.p, np.abs(s.p - .5) * 2])          # channel 1 without any detector feature

# ---------------- (A) detector-free channel 1
for name in CELLS:
    s = S[name]; run = R.Runner(s, M, X=Xdf(s))
    for a in (0.10, 0.15):
        f = run.run(a, gate=None); sf = R.summarise(f, a)
        qext = R.external_gate_matched(S[DEV[name]], a, s.n // 3, X_of=Xdf)
        res = run.run(a, gate="fixed", q=qext); sx = R.summarise(res, a); gm, gsd = R.paired_gain(f, res)
        res2 = run.run(a, gate="fitsel"); s2 = R.summarise(res2, a); gm2, _ = R.paired_gain(f, res2)
        out[f"A|{name}|{a}"] = dict(filter=sf, ext=dict(sx, gain=gm, gain_sd=gsd, q=qext), fitsel=dict(s2, gain=gm2))
    print("A", name, f"+{time.time()-t0:.0f}s", flush=True)

# ---------------- (B) multi-start search spending delta over start points
def multistart(run, alpha, q, starts, delta=R.DELTA):
    Sx, m = run.S, run.mode; dl = delta / len(starts); res = []
    for (tr, cal, te), sc, ref in zip(run.splits, run.scores, run.refs):
        if sc is None: continue
        best = None
        for e0 in starts:
            ks = R.k_start(alpha, dl, e0)
            thr = np.quantile(ref, 1 - R._grid_lams(len(run.sub) // 3, ks, m.n_grid, m.c))
            rg = None if q is None else R.Region(Sx, "margin", q, tr, ref, m, Sx.m[cal])
            j = R.fst_region(Sx, sc, cal, thr, alpha, dl, rg)
            if j is None: continue
            kc = R.evaluate(Sx, sc, cal, thr[j], rg)["k"]
            if best is None or kc > best[0]: best = (kc, thr[j], rg)
        if best is None:
            res.append(dict(abort=True, cov=0.0, n=len(te), risk=np.nan, e=0, k=0, k_rep=0)); continue
        r = R.evaluate(Sx, sc, te, best[1], best[2]); r.update(abort=False, cov=r["k"] / r["n"], risk=(r["e"] / r["k"]) if r["k"] else np.nan); res.append(r)
    return res
for name in CELLS:
    run = R.Runner(S[name], M)
    for a in (0.10, 0.15):
        qext = R.external_gate_matched(S[DEV[name]], a, S[name].n // 3)
        row = {}
        for lab, e0 in (("e0=0", 0), ("e0=1", 1)):
            row[f"filter {lab}"] = R.summarise(run.run(a, gate=None, e0=e0), a)
            row[f"ccrc {lab}"] = R.summarise(run.run(a, gate="fixed", q=qext, e0=e0), a)
        row["filter multi"] = R.summarise(multistart(run, a, None, (0, 1, 2)), a)
        row["ccrc multi"] = R.summarise(multistart(run, a, qext, (0, 1, 2)), a)
        row["q"] = qext; out[f"B|{name}|{a}"] = row
    print("B", name, f"+{time.time()-t0:.0f}s", flush=True)

# ---------------- (C) transfer: everything except calibration and test comes from another benchmark
def transfer(dev, tgt, alpha, q, mode=M, reps=R.REPS, delta=R.DELTA):
    Xd, Xt = dev.features("impute"), tgt.features("impute")
    w = R.fit_logistic(Xd, dev.ok)
    rn = np.random.default_rng(3); uniq, inv = np.unique(dev.img, return_inverse=True)
    f = (rn.permutation(len(uniq)) % mode.k_folds)[inv]; ref = np.empty(dev.n)
    for k in range(mode.k_folds):
        tk, rk = f == k, f != k; ref[tk] = R.predict(R.fit_logistic(Xd[rk], dev.ok[rk]), Xd[tk])
    s = R.predict(w, Xt)
    ks = R.k_start(alpha, delta, mode.e0)
    thr = np.quantile(ref, 1 - R._grid_lams(tgt.n // 3, ks, mode.n_grid, mode.c))
    gm = dev.m[dev.m >= 0]; gv = np.quantile(gm, 1 - q) if q is not None else None
    def reg():
        if q is None: return None
        r = R.Region(tgt, "margin", q, np.arange(min(10, tgt.n)), None, mode, None); r.gv = gv; return r
    f_res, c_res = [], []
    for (tr, cal, te) in R.make_splits(tgt, reps, 0, "image"):
        for res, rg in ((f_res, None), (c_res, reg())):
            j = R.fst_region(tgt, s, cal, thr, alpha, delta, rg)
            if j is None: res.append(dict(abort=True, cov=0.0, n=len(te), risk=np.nan, e=0, k=0, k_rep=0)); continue
            r = R.evaluate(tgt, s, te, thr[j], rg); r.update(abort=False, cov=r["k"] / r["n"], risk=(r["e"] / r["k"]) if r["k"] else np.nan); res.append(r)
    return f_res, c_res
for tgt, dev in [("AMBER(d) LLaVA", "POPE-3000 LLaVA"), ("POPE-3000 LLaVA", "AMBER(d) LLaVA"), ("POPE-adv LLaVA", "AMBER(d) LLaVA"),
                 ("POPE-adv Qwen2-VL", "AMBER(d) LLaVA"), ("POPE-adv LLaVA+VCD", "AMBER(d) LLaVA")]:
    for a in (0.10, 0.15):
        q = R.external_gate_matched(S[dev], a, S[tgt].n // 3)
        f, c = transfer(S[dev], S[tgt], a, q)
        sf, sc = R.summarise(f, a), R.summarise(c, a); gm, gsd = R.paired_gain(f, c)
        out[f"C|{tgt}|{dev}|{a}"] = dict(filter=sf, ccrc=dict(sc, gain=gm, gain_sd=gsd), q=q)
    print("C", tgt, f"+{time.time()-t0:.0f}s", flush=True)
json.dump(out, open("results_review2.json", "w"), indent=1)
