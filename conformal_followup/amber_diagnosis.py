"""
Re-diagnosis of the AMBER negative result (review blocker 1).

The manuscript attributes the AMBER loss to "the absence of headroom", exactly as
Proposition 3 (the abstention-floor diagnostic) would suggest, and explicitly
rejects sample size as the cause ("subsampling POPE to the same n=228 still gains
+3.2 points"). An independent reviewer found the opposite: conditional on both
arms certifying anything, CCRC WINS on AMBER, and the entire reported loss is
CCRC's fixed sequence ABORTING on splits where filtering does not.

This script tests that independently and reports, at alpha in {0.10, 0.15}:
  * unconditional gain (what Table 5 reports)
  * gain CONDITIONAL on neither arm aborting
  * abort rate of each arm
  * fraction of splits on which lambda-hat MOVES between the arms
  * r_r vs r_a -- the repair-region risk against the accepted-region risk. If
    r_r < r_a, dilution is FAVOURABLE on AMBER (Prop. 2's good case), which would
    contradict a "no headroom, dilution hurts" story outright.
  * an arithmetic ledger showing how much of the unconditional gain is explained
    by aborts alone.
  * a sample-size probe: abort rate as n grows, plus POPE subsampled to AMBER's n,
    which is the comparison the manuscript uses to rule sample size out.

PROTOCOL: canonical (canonical_fst.py), PAIRED splits, 400 reps.

Run:  python3 amber_diagnosis.py
"""
import numpy as np, warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
import canonical_fst as F
import ccrc_v3 as C

ALPHAS = (0.10, 0.15)
Q = 0.10
REPS = F.REPS
DELTA = F.DELTA


def arms(X, ok_o, ok_r, mg, alpha, reps=REPS, seed=0, q=Q):
    """Filtering and CCRC on identical paired splits. Per-split records."""
    n = len(ok_o)
    rec = []
    for (tr, cal, te), s in zip(F.SPLITS(n, reps, seed),
                                F.fitted_scores(X, ok_o, reps, seed)):
        if s is None: continue
        s_cal, m_cal = s[cal], mg[cal]
        tm = float(np.quantile(m_cal, 1 - q))
        lf = F.fst(s_cal, ok_o[cal], alpha, DELTA, F.cp_upper)
        lc = F.fst(s_cal, ok_o[cal], alpha, DELTA, F.cp_upper,
                   rep_mask_cal=(m_cal >= tm), ok_rep_cal=ok_r[cal])
        r = dict(lf=lf, lc=lc, cov_f=0.0, cov_c=0.0,
                 r_a=None, r_r=None, rep_mass=0.0, moved=None)
        if lf is not None:
            thr = float(np.quantile(s_cal, 1 - lf))
            r["cov_f"] = float((s[te] >= thr).mean())
        if lc is not None:
            thr = float(np.quantile(s_cal, 1 - lc))
            acc = s[te] >= thr
            rep = (~acc) & (mg[te] >= tm)
            r["cov_c"] = float((acc.sum() + rep.sum()) / len(te))
            r["rep_mass"] = float(rep.sum() / len(te))
            if acc.sum(): r["r_a"] = float((1 - ok_o[te][acc]).mean())
            if rep.sum(): r["r_r"] = float((1 - ok_r[te][rep]).mean())
        if lf is not None and lc is not None:
            r["moved"] = (lf != lc)
        rec.append(r)
    return rec


def report(name, rec, alpha):
    cf = np.array([r["cov_f"] for r in rec]) * 100
    cc = np.array([r["cov_c"] for r in rec]) * 100
    ab_f = F.abort_rate([r["lf"] for r in rec])
    ab_c = F.abort_rate([r["lc"] for r in rec])
    g_un = F.gain_stats(cf, cc)
    both = np.array([(r["lf"] is not None and r["lc"] is not None) for r in rec])
    g_co = F.gain_stats(cf[both], cc[both])
    mv = np.array([r["moved"] for r in rec if r["moved"] is not None], bool)
    ra = np.array([r["r_a"] for r in rec if r["r_a"] is not None]) * 100
    rr = np.array([r["r_r"] for r in rec if r["r_r"] is not None]) * 100
    rm = np.array([r["rep_mass"] for r in rec if r["lc"] is not None]) * 100

    print(f"\n--- {name}, alpha={alpha:.2f}, delta={DELTA:.2f}, {len(rec)} paired splits ---")
    print(f"  filtering coverage           {cf.mean():6.2f} +- {cf.std(ddof=1):.2f} %")
    print(f"  CCRC coverage                {cc.mean():6.2f} +- {cc.std(ddof=1):.2f} %")
    print(f"  UNCONDITIONAL gain           {g_un.mean:+6.2f} pp   sd {g_un.sd:5.2f}   "
          f"95% CI [{g_un.lo:+.2f},{g_un.hi:+.2f}]   p={g_un.p_t:.3g}")
    print(f"  gain | NEITHER arm aborts    {g_co.mean:+6.2f} pp   sd {g_co.sd:5.2f}   "
          f"95% CI [{g_co.lo:+.2f},{g_co.hi:+.2f}]   p={g_co.p_t:.3g}   "
          f"(n={both.sum()} splits)")
    print(f"  abort rate  filtering        {ab_f*100:6.2f} %")
    print(f"  abort rate  CCRC             {ab_c*100:6.2f} %")
    print(f"  lambda-hat MOVES on          {mv.mean()*100:6.2f} % of splits where both certify")
    print(f"  r_a  accepted-region risk    {ra.mean():6.2f} %   (n={len(ra)})")
    print(f"  r_r  repair-region risk      {rr.mean() if len(rr) else float('nan'):6.2f} %   "
          f"(n={len(rr)})")
    print(f"  repaired mass                {rm.mean():6.2f} % of items")
    verdict = ("FAVOURABLE (r_r < r_a: Prop.2's good case, dilution lowers emitted risk)"
               if len(rr) and rr.mean() < ra.mean() else
               "UNFAVOURABLE (r_r > r_a: repairs consume slack)")
    print(f"  dilution direction:          {verdict}")

    # -------- arithmetic ledger: how much of the unconditional gain is aborts?
    only_c = np.array([(r["lf"] is not None and r["lc"] is None) for r in rec])
    only_f = np.array([(r["lf"] is None and r["lc"] is not None) for r in rec])
    neither = np.array([(r["lf"] is None and r["lc"] is None) for r in rec])
    t_both = (cc[both] - cf[both]).sum() / len(rec)
    t_conly = (cc[only_c] - cf[only_c]).sum() / len(rec)   # CCRC aborted, filter did not
    t_fonly = (cc[only_f] - cf[only_f]).sum() / len(rec)
    print(f"  ledger of the unconditional gain ({g_un.mean:+.2f} pp), by split class:")
    print(f"    both certify        {both.sum():4d} splits  contributes {t_both:+6.2f} pp")
    print(f"    CCRC aborts only    {only_c.sum():4d} splits  contributes {t_conly:+6.2f} pp"
          f"   <-- pure abort penalty")
    print(f"    filter aborts only  {only_f.sum():4d} splits  contributes {t_fonly:+6.2f} pp")
    print(f"    neither certifies   {neither.sum():4d} splits  contributes {0.0:+6.2f} pp")
    frac = (t_conly / g_un.mean * 100) if abs(g_un.mean) > 1e-9 else float("nan")
    print(f"    => the CCRC-only-abort class alone accounts for {frac:.0f}% of the")
    print(f"       unconditional gain of {g_un.mean:+.2f} pp")
    return dict(cf=cf.mean(), cc=cc.mean(), g_un=g_un, g_co=g_co, ab_f=ab_f, ab_c=ab_c,
                moved=mv.mean(), ra=ra.mean(), rr=(rr.mean() if len(rr) else float("nan")),
                rm=rm.mean(), t_conly=t_conly)


def main():
    print(__doc__)
    SET = dict(C.settings())
    out = {}
    for a in ALPHAS:
        out[a] = report("AMBER(d) LLaVA", arms(*SET["AMBER(d) LLaVA"], a), a)

    # ------------------------------------------------ contrast: POPE-adv LLaVA
    print("\n" + "=" * 78)
    print("CONTRAST -- the same statistics on a setting where CCRC genuinely gains")
    print("=" * 78)
    for a in ALPHAS:
        report("POPE-1500 LLaVA", arms(*SET["POPE-1500 LLaVA"], a), a)

    # ------------------------------------------------------ sample-size probe
    print("\n" + "=" * 78)
    print("SAMPLE-SIZE PROBE")
    print("The manuscript rules sample size out by subsampling POPE to n=228 and still")
    print("gaining (+3.2 pp). Two problems with that argument:")
    print("  (a) the +3.2/+5.6/+10.3 subsample figures come from missing_numbers.py")
    print("      BLOCK 6, which measures the GROUNDING-SCORE gain (confidence-only vs")
    print("      confidence+detection), not the CCRC-vs-filtering gain. They are")
    print("      different quantities and cannot bear on the AMBER loss at all.")
    print("  (b) even taken at face value it shows POPE tolerates n=228, not that")
    print("      AMBER's aborts are unrelated to n.")
    print("The columns below are the right comparison: CCRC vs filtering at matched n,")
    print("with the abort rate of each arm.")
    print("=" * 78)
    print(f"{'setting':22s}{'n':>6s}{'alpha':>6s}{'cov filt':>10s}{'cov CCRC':>10s}"
          f"{'gain':>8s}{'abort filt':>12s}{'abort CCRC':>12s}")
    for label, key in [("AMBER(d) LLaVA", "AMBER(d) LLaVA"), ("POPE-1500 LLaVA", "POPE-1500 LLaVA")]:
        X, ok_o, ok_r, mg = SET[key]
        N = len(ok_o)
        sizes = sorted(dict.fromkeys(nn for nn in (114, 152, 190, 228, 456, 912, N)
                                     if nn <= N))
        for nn in sizes:
            rng = np.random.default_rng(7)
            sub = rng.permutation(N)[:nn]
            for a in ALPHAS:
                rec = arms(X[sub], ok_o[sub], ok_r[sub], mg[sub], a, reps=REPS)
                cf = np.array([r["cov_f"] for r in rec]) * 100
                cc = np.array([r["cov_c"] for r in rec]) * 100
                print(f"{label:22s}{nn:6d}{a:6.2f}{cf.mean():9.1f}%{cc.mean():9.1f}%"
                      f"{cc.mean()-cf.mean():+8.2f}"
                      f"{F.abort_rate([r['lf'] for r in rec])*100:11.1f}%"
                      f"{F.abort_rate([r['lc'] for r in rec])*100:11.1f}%")

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    for a in ALPHAS:
        o = out[a]
        print(f"  alpha={a:.2f}:  unconditional {o['g_un'].mean:+.2f} pp   "
              f"conditional on no abort {o['g_co'].mean:+.2f} pp   "
              f"aborts filt {o['ab_f']*100:.1f}% / CCRC {o['ab_c']*100:.1f}%   "
              f"r_r={o['rr']:.1f}% vs r_a={o['ra']:.1f}%")
    print("\n  Read the two gain columns together. If the conditional gain is positive while")
    print("  the unconditional gain is negative, the loss is an ABORT phenomenon: CCRC's")
    print("  fixed sequence fails to certify anything on some splits and deploys 0%")
    print("  coverage there, while filtering certifies. That is a statement about the")
    print("  prefix structure of the test at small calibration size, NOT about headroom,")
    print("  and NOT about the repair channel being weak. If in addition r_r < r_a, the")
    print("  dilution mechanism is working in CCRC's favour on this dataset, which is")
    print("  flatly inconsistent with the 'no headroom' explanation.")


if __name__ == "__main__":
    main()
