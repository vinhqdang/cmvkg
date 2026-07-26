"""
Why the port was necessary: the leaky rule measured against the canonical one.

Eight scripts selected the threshold by an exhaustive max-over-thresholds search
over every distinct calibration score:

    for t in np.unique(score_cal):
        m = score_cal >= t; k = m.sum(); e = (1 - ok_cal[m]).sum()
        if cp_upper(e, k, delta) <= alpha and k > best: best, tau = k, t

i.e. "of all O(n_cal) thresholds whose Clopper-Pearson bound passes, keep the one
that emits the most". There is no prefix rule, no k_min start, and no correction
for having looked at every threshold. The selection event is a union over the
whole grid, so the per-threshold delta does not transfer to the selected one.

The canonical rule (ccrc_v3.py, missing_numbers.py, canonical_fst.py) fixes an
ascending lambda grid starting where the bound can certify at all, tests in that
order at full delta, stops at the first non-rejection and returns the last
passing index.

This script runs both on identical paired splits and reports the audit statistic
that matters -- Pr[realised risk > alpha] against delta -- plus the coverage each
rule buys. Both arms see the same folds and the same fitted score; the ONLY
difference is the threshold-selection rule.

Run:  python3 protocol_audit.py
"""
import numpy as np, warnings; warnings.filterwarnings("ignore")
from scipy.stats import beta
from sklearn.linear_model import LogisticRegression
import canonical_fst as F
import ccrc_v3 as C

ALPHAS = (0.10, 0.15)
Q_MAIN = 0.10          # the canonical CCRC repair gate


def leaky(score_cal, ok_cal, alpha, delta=F.DELTA, rep_mask_cal=None, ok_rep_cal=None):
    """The rule that was in the eight scripts, vectorised exactly.

    Original inner loop:
        for t in np.unique(score_cal):
            m = score_cal >= t; k = m.sum(); e = (1 - ok_cal[m]).sum()
            if cp_upper(e, k, delta) <= alpha and k > best: best, tau = k, t
    Because k is nondecreasing as t decreases, "largest k among the passing
    thresholds" is just the smallest passing t. Returns tau or None.
    """
    u = np.unique(score_cal)                       # ascending
    o = np.argsort(-score_cal)
    sd = score_cal[o]                              # descending
    err = (1 - ok_cal)[o].astype(float)
    if rep_mask_cal is None:
        cum_k = np.arange(1, len(sd) + 1, dtype=float)
        cum_e = np.cumsum(err)
    else:
        # emitted = accepted (score >= t) + repaired (score < t AND gate open);
        # repaired mass therefore *shrinks* as t falls, exactly as in the originals.
        rm = rep_mask_cal[o].astype(float)
        err_r = (1 - ok_rep_cal)[o].astype(float)
        tot_r, tot_re = rm.sum(), (rm * err_r).sum()
        cum_k = np.arange(1, len(sd) + 1, dtype=float) + (tot_r - np.cumsum(rm))
        cum_e = np.cumsum(err) + (tot_re - np.cumsum(rm * err_r))
    # k/e at each unique threshold: index of its last occurrence in the desc order
    idx = len(sd) - np.searchsorted(sd[::-1], u, side="left") - 1
    k = cum_k[idx]; e = cum_e[idx]
    ub = np.ones_like(k)
    ok_ = (k > 0) & (e < k)
    ub[ok_] = beta.ppf(1 - delta, e[ok_] + 1, k[ok_] - e[ok_])
    passing = np.flatnonzero((ub <= alpha) & (k > 0))
    if len(passing) == 0: return None
    return float(u[passing[np.argmax(k[passing])]])


def _emit(s, mg, ii, thr, tm, ok_o, ok_r):
    """Emitted mask/counts at (thr, tm). tm=None -> filtering only."""
    acc = s[ii] >= thr
    rep = np.zeros_like(acc) if tm is None else ((~acc) & (mg[ii] >= tm))
    k = int(acc.sum() + rep.sum())
    e = int((1 - ok_o[ii][acc]).sum() + (1 - ok_r[ii][rep]).sum())
    return k, e


def run(alpha, q, reps=F.REPS):
    arm = "filtering" if q is None else f"CCRC (q={q:g})"
    print(f"\n===== alpha={alpha:.2f}, delta={F.DELTA:.2f}, {reps} paired splits, "
          f"arm = {arm} =====")
    print(f"{'setting':22s}{'n':>5s}"
          f"{'cov leaky':>11s}{'cov FST':>9s}{'delta':>8s}"
          f"{'exc(te) leaky':>15s}{'exc(te) FST':>13s}"
          f"{'exc(all) leaky':>16s}{'exc(all) FST':>14s}"
          f"{'abort L':>9s}{'abort FST':>11s}")
    rows = []
    ALL = None
    for name, (X, ok_o, ok_r, mg) in C.settings():
        n = len(ok_o); ALL = np.arange(n)
        covL, covF = [], []
        rteL, rteF, ralL, ralF = [], [], [], []
        lamF, tauL = [], []
        scores = F.fitted_scores(X, ok_o, reps)
        for (tr, cal, te), s in zip(F.SPLITS(n, reps), scores):
            if s is None: continue
            m_cal = mg[cal]
            tm = None if q is None else float(np.quantile(m_cal, 1 - q))
            rep_mask_cal = None if q is None else (m_cal >= tm)

            # --- leaky: max-over-thresholds, no prefix rule, no k_min
            tau = leaky(s[cal], ok_o[cal], alpha,
                        rep_mask_cal=rep_mask_cal, ok_rep_cal=ok_r[cal])
            tauL.append(tau)
            if tau is None:
                covL.append(0.0)
            else:
                k, e = _emit(s, mg, te, tau, tm, ok_o, ok_r)
                covL.append(k / len(te))
                if k: rteL.append(e / k)
                ka, ea = _emit(s, mg, ALL, tau, tm, ok_o, ok_r)
                if ka: ralL.append(ea / ka)

            # --- canonical ascending fixed sequence
            lam = F.fst(s[cal], ok_o[cal], alpha, F.DELTA, F.cp_upper,
                        rep_mask_cal=rep_mask_cal, ok_rep_cal=ok_r[cal])
            lamF.append(lam)
            if lam is None:
                covF.append(0.0)
            else:
                thr = float(np.quantile(s[cal], 1 - lam))
                k, e = _emit(s, mg, te, thr, tm, ok_o, ok_r)
                covF.append(k / len(te))
                if k: rteF.append(e / k)
                ka, ea = _emit(s, mg, ALL, thr, tm, ok_o, ok_r)
                if ka: ralF.append(ea / ka)

        cL = np.array(covL) * 100; cF = np.array(covF) * 100
        r = dict(name=name, n=n, covL=cL.mean(), covF=cF.mean(),
                 xteL=F.exceedance(rteL, alpha) * 100, xteF=F.exceedance(rteF, alpha) * 100,
                 xalL=F.exceedance(ralL, alpha) * 100, xalF=F.exceedance(ralF, alpha) * 100,
                 abL=F.abort_rate(tauL) * 100, abF=F.abort_rate(lamF) * 100,
                 g=F.gain_stats(cL, cF, f"a={alpha:.2f} {arm:14s} {name}"))
        rows.append(r)
        print(f"{name:22s}{n:5d}{r['covL']:10.1f}%{r['covF']:8.1f}%"
              f"{r['covF']-r['covL']:+8.1f}"
              f"{r['xteL']:14.1f}%{r['xteF']:12.1f}%"
              f"{r['xalL']:15.1f}%{r['xalF']:13.1f}%{r['abL']:8.1f}%{r['abF']:10.1f}%")
    xa = [r['xalL'] for r in rows]; xb = [r['xalF'] for r in rows]
    ya = [r['xteL'] for r in rows]; yb = [r['xteF'] for r in rows]
    print(f"\n  exc(all), the population-risk proxy, vs delta={F.DELTA*100:.0f}%:"
          f"  leaky {min(xa):.1f}-{max(xa):.1f}%   fixed sequence {min(xb):.1f}-{max(xb):.1f}%")
    print(f"  exc(te), the noisy unbiased version:"
          f"      leaky {min(ya):.1f}-{max(ya):.1f}%   fixed sequence {min(yb):.1f}-{max(yb):.1f}%")
    return rows


if __name__ == "__main__":
    print(__doc__)
    all_rows = {}
    for q in (None, Q_MAIN):
        for a in ALPHAS:
            all_rows[(a, q)] = run(a, q)
    print("\n  exc(te)  = held-out test fold (unbiased, noisy: n_te = n/3)")
    print("  exc(all) = selected policy re-scored on all n items (low-noise proxy for the")
    print("             POPULATION risk of the selected policy; this is the column that")
    print("             decides whether the guarantee holds)")
    print("  Aborts are excluded from the exceedance denominators (no emitted set => no")
    print("  risk) but enter the coverage means as 0%. The leaky rule aborts far less often")
    print("  because it may pick any threshold at all, including ones with k < k_min.")

    print("\n\n=== paired coverage cost of using the valid rule (FST - leaky), in pp ===")
    print(F.Gain.header(46))
    for key in all_rows:
        for r in all_rows[key]:
            print(r['g'].line(46))
    print("\nThe leaky rule's extra coverage is not a free lunch: it is bought by selecting")
    print("over O(n_cal) thresholds on the calibration fold without paying for the")
    print("selection. It shows up as inflated exceedance above -- consistently higher than")
    print("the fixed sequence on both audit columns, on every cell where either is nonzero.")
