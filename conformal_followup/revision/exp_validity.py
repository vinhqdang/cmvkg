"""Known-population validity audit of the fixed-sequence procedure (R1.1, R1.3, R1.6).

Real benchmarks cannot audit the guarantee: the population risk of the selected policy is unknown, and
held-out folds are noisy. Here the population is a generative model, so the risk of the SELECTED policy
(a threshold VALUE on the score and a gate VALUE on the evidence margin) is computed to Monte-Carlo
precision on a 400k-item reference sample, and the guarantee Pr[Risk > alpha] <= delta is audited exactly.

Population. Image clusters of K=6 items. Item j of cluster c: u ~ N(0,1); the model is right with
probability sigmoid(b0 + b1 u + z_c), z_c ~ N(0, sigma^2) a shared image effect (sigma=0: i.i.d. items;
sigma>0: items within an image are positively correlated, the case reviewer 1 worries about). The score is
the best item-level predictor s = sigmoid(b0 + b1 u). Channel 2 has margin v ~ U(0,1) and is right with
probability 1 - 0.5 exp(-4 v), independently of the model.

Procedures. `value`: thresholds and gate are values read off an independent FIT sample (Theorem 1's object);
`rank`: thresholds and gate are empirical quantiles of the calibration sample itself (the previous
implementation). `single`: one randomly chosen item per image is used for calibration (exact i.i.d. unit).
Writes results_validity.json."""
import json, sys, time, itertools, numpy as np
from scipy.stats import beta
import ccrc_rev as R
sig = lambda x: 1 / (1 + np.exp(-x))
K = 6; B1 = 1.5; NREF = 400_000; NFIT = 300; GRID = 40
pr = lambda v: 1 - 0.5 * np.exp(-4 * v)

def gen(n_items, sigma, rng, per_cluster=K, b0=None):
    nc = int(np.ceil(n_items / per_cluster)); N = nc * per_cluster
    u = rng.normal(size=N); z = np.repeat(rng.normal(size=nc) * sigma, per_cluster)
    ok = (rng.random(N) < sig(b0 + B1 * u + z)).astype(int)
    v = rng.random(N); okr = (rng.random(N) < pr(v)).astype(int)
    return dict(s=sig(b0 + B1 * u), ok=ok, v=v, okr=okr, cl=np.repeat(np.arange(nc), per_cluster))

def calibrate_b0(mu, rng):
    lo, hi = -3.0, 6.0
    for _ in range(60):
        mid = (lo + hi) / 2
        u = rng.normal(size=400_000); ok = sig(mid + B1 * u).mean()   # sigma-marginal approx; fixed below
        lo, hi = (mid, hi) if 1 - ok > mu else (lo, mid)
    return (lo + hi) / 2

def pop_risk(ref, thr, gv):
    acc = ref["s"] >= thr
    rep = (~acc) & (ref["v"] >= gv) if gv is not None else np.zeros_like(acc)
    k = acc.sum() + rep.sum()
    e = (1 - ref["ok"][acc]).sum() + (1 - ref["okr"][rep]).sum()
    return e / k if k else 0.0

def trial(ref, rng, alpha, delta, n_cal, sigma, proc, arm, calib, b0, e0=1, c=1.5, q=0.25):
    fit = gen(NFIT, 0.0, rng, per_cluster=1, b0=b0) if False else gen(NFIT, sigma, rng, b0=b0)
    cal = gen(n_cal, sigma, rng, b0=b0)
    if calib == "single":                      # one random item per image -> n_cal images
        cal = gen(n_cal * K, sigma, rng, b0=b0)
        pick = np.array([rng.choice(np.flatnonzero(cal["cl"] == c_)) for c_ in np.unique(cal["cl"])])[:n_cal]
        cal = {k_: v_[pick] for k_, v_ in cal.items()}
    ks = R.k_start(alpha, delta, e0)
    lams = np.linspace(min(0.9, max(0.05, c * ks / len(cal["s"]))), 1.0, GRID)
    ref_s = fit["s"] if proc == "value" else cal["s"]
    thr = np.quantile(ref_s, 1 - lams)
    gv = None
    if arm == "repair": gv = np.quantile(fit["v"] if proc == "value" else cal["v"], 1 - q)
    j = R.fst(cal["s"], cal["ok"], thr, alpha, delta,
              rep_cal=None if gv is None else (cal["v"] >= gv), okr_cal=cal["okr"])
    if j is None: return None
    return pop_risk(ref, thr[j], gv)

def run_cell(mu, alpha, n_cal, sigma, proc, arm, calib, trials, seed, b0):
    rng = np.random.default_rng(seed)
    ref = gen(NREF, sigma, np.random.default_rng(999), b0=b0)
    rs = [trial(ref, rng, alpha, R.DELTA, n_cal, sigma, proc, arm, calib, b0) for _ in range(trials)]
    ok = [r for r in rs if r is not None]
    return dict(abort=1 - len(ok) / trials, exc=float(np.mean(np.array(ok) > alpha + 1e-12)) if ok else float("nan"),
                exc_incl_abort=float(np.sum(np.array(ok) > alpha + 1e-12) / trials),
                mean_risk=float(np.mean(ok)) if ok else float("nan"))

if __name__ == "__main__":
    trials = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    out = {}; t0 = time.time(); rng0 = np.random.default_rng(0)
    cfgs = []
    for mu, alpha, n_cal, sigma in itertools.product((0.15, 0.25), (0.10, 0.15), (150, 450), (0.0, 1.0, 2.0)):
        for proc, arm in itertools.product(("value", "rank"), ("filter", "repair")):
            cfgs.append((mu, alpha, n_cal, sigma, proc, arm, "full"))
        if sigma > 0:
            cfgs.append((mu, alpha, n_cal, sigma, "value", "filter", "single"))
            cfgs.append((mu, alpha, n_cal, sigma, "value", "repair", "single"))
    b0s = {}
    for i, cfg in enumerate(cfgs):
        mu = cfg[0]
        if mu not in b0s: b0s[mu] = calibrate_b0(mu, rng0)
        out["|".join(map(str, cfg))] = run_cell(*cfg, trials, 1000 + i, b0s[mu])
        if i % 8 == 0: print(f"{i}/{len(cfgs)} +{time.time()-t0:.0f}s", flush=True)
    json.dump(out, open("results_validity.json", "w"), indent=1)
