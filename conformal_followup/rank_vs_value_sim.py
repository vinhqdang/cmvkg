"""Does the RANK-indexed fixed sequence (empirical-quantile thresholds on the same fold whose
errors are counted) actually control Pr[population risk > alpha] at delta?

Ground truth is known, so population risk of the selected policy is computed exactly by
numerical integration rather than estimated. Four score shapes x four calibration sizes.

Protocol matches ccrc_v3.calibrate: lam grid of 40 points from lam0 = min(.9, max(.05,
1.5*k_min/n_cal)) to 1.0, ascending, Clopper-Pearson upper bound, stop at first non-rejection,
return last passing lam. Filtering only (no repair arm) -- the rank-selection effect is what is
under test.
"""
import numpy as np
from scipy.stats import beta

ALPHA, DELTA, REPS = 0.10, 0.10, 4000
KMIN = int(np.ceil(np.log(DELTA) / np.log(1 - ALPHA)))          # 22


def cp_upper(e, k, d=DELTA):
    if k == 0:
        return 1.0
    return 1.0 if e == k else float(beta.ppf(1 - d, e + 1, k - e))


# ---- four score shapes: p(u) = P(Y=1 | score quantile u), u ~ U(0,1) --------------
# The score itself is monotone in u, so only p(u) matters for a rank-indexed rule; the
# "miscalibrated monotone transform" and "heavy tailed" shapes differ from the others in how
# fast p rises near the top, which is what the upper-tail selection sees.
SHAPES = {
    "well-separated":  lambda u: 0.55 + 0.44 / (1 + np.exp(-12 * (u - 0.45))),
    "weakly-separated": lambda u: 0.72 + 0.20 * u,
    "heavy-tailed":    lambda u: 0.60 + 0.39 * u ** 6,
    "miscalibrated":   lambda u: np.clip(0.50 + 0.49 * np.sqrt(u), 0, 0.995),
}
NCALS = [100, 250, 500, 1500]
GRID = 40

rng_global = np.random.default_rng(7)
print(f"k_min = {KMIN}, alpha = {ALPHA}, delta = {DELTA}, reps = {REPS}\n")
print(f"{'shape':18s}" + "".join(f"{('n=' + str(n)):>17s}" for n in NCALS))
allv = []
for name, p in SHAPES.items():
    # population risk of threshold at score-quantile level u0 (accept u >= u0):
    #   R(u0) = 1 - mean_{u in [u0,1]} p(u)
    ug = np.linspace(0, 1, 200001)
    pg = p(ug)
    cum = np.concatenate([[0.0], np.cumsum((pg[1:] + pg[:-1]) / 2 * np.diff(ug))])
    tot = cum[-1]

    def pop_risk(u0):
        i = np.searchsorted(ug, u0)
        mass = 1.0 - u0
        if mass <= 1e-12:
            return 0.0
        return 1.0 - (tot - cum[i]) / mass

    row = f"{name:18s}"
    for ncal in NCALS:
        lam0 = min(0.9, max(0.05, 1.5 * KMIN / ncal))
        lams = np.linspace(lam0, 1.0, GRID)
        rng = np.random.default_rng(abs(hash(name)) % 10000 + ncal)
        exceed = 0
        certified = 0
        for _ in range(REPS):
            u = rng.random(ncal)
            y = (rng.random(ncal) < p(u)).astype(int)
            order = np.argsort(-u)          # descending score
            ysort = y[order]
            best = None
            for lam in lams:
                k = int(np.ceil(lam * ncal))
                if k < KMIN:
                    break
                e = int((1 - ysort[:k]).sum())
                if cp_upper(e, k) <= ALPHA:
                    best = k
                else:
                    break
            if best is None:
                continue                     # abort: emits nothing, no risk
            certified += 1
            # deployed threshold is the empirical score at rank `best`; its population
            # acceptance level is that item's own quantile
            u0 = np.sort(u)[::-1][best - 1]
            if pop_risk(u0) > ALPHA:
                exceed += 1
        frac_uncond = exceed / REPS
        frac_cond = exceed / max(certified, 1)
        abort = 1 - certified / REPS
        allv.append((frac_uncond, frac_cond, abort))
        row += f"{frac_uncond*100:5.1f}/{frac_cond*100:5.1f}/{abort*100:4.0f}"
    print(row)
u=[a for a,b,c in allv]; c=[b for a,b,c in allv]; ab=[x for _,_,x in allv]
print("\ncolumns are  UNCONDITIONAL exceedance / conditional-on-certifying / abort rate  (%)")
print(f"unconditional range over the 16 cells: {min(u)*100:.1f}% -- {max(u)*100:.1f}%   (delta = {DELTA*100:.0f}%)")
print(f"conditional  range: {min(c)*100:.1f}% -- {max(c)*100:.1f}%")
print(f"abort        range: {min(ab)*100:.0f}% -- {max(ab)*100:.0f}%")
