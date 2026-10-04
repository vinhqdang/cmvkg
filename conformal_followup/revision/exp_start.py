"""Start-point ablation (R1.5): e0 in {0,1,2,3} (tolerated errors at the first hypothesis) and the
placement multiplier c in {1.0, 1.5}, for filtering and for CCRC (fit-fold gate). Writes results_start.json."""
import json, time, ccrc_rev as R
S = R.load_settings(); M = R.Mode; out = {}; t0 = time.time()
for name in R.MAIN:
    for c in (1.0, 1.5):
        for e0 in (0, 1, 2, 3):
            run = R.Runner(S[name], M(e0=e0, c=c))
            for a in (0.05, 0.10, 0.15):
                f = run.run(a, gate=None); x = run.run(a, gate="fitsel")
                sf, sx = R.summarise(f, a), R.summarise(x, a); gm, gs = R.paired_gain(f, x)
                out[f"{name}|c={c}|e0={e0}|{a}"] = dict(ks=R.k_start(a, R.DELTA, e0), filter=sf, ccrc=sx, gain=gm, gain_sd=gs)
    print(f"{name} +{time.time()-t0:.0f}s", flush=True)
json.dump(out, open("results_start.json", "w"), indent=1)
