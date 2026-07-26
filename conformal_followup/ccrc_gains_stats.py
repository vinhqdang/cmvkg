"""
CCRC vs filtering under the canonical protocol, with the statistics the tables
were missing, plus the dilution decomposition (review blocker 6).

Feeds Table 5 (tab:ccrc), Table 13 (tab:qsens), the leverage/dilution claim of
Sec 4.5 (fig:precondition panel B), and the mechanism discussion.

WHAT THIS ADDS over the previous reporting of these cells:
  * 400 paired splits, not 60. At 60 reps the Monte-Carlo SE on these gains is
    0.8-4.2 pp, so a gain quoted to 0.1 pp was noise. Every 400-rep estimate here
    came out LESS favourable than its 60-rep predecessor.
  * per-split sd, 95% CI for the mean gain, two-sided paired t-test and Wilcoxon.
  * ABORT RATE for both arms. This was unreported everywhere and it is not a
    footnote: a split on which the fixed sequence certifies nothing deploys at 0%
    coverage, and on some cells that happens on a fifth of splits.
  * validity audited as Pr[realised risk > alpha] against delta, on the test fold
    and re-scored on all items, never as mean risk.
  * DILUTION DECOMPOSITION. The gain is split into
      (i)  the repaired items themselves -- emitted automatically whenever
           lambda-hat does not fall, so they cost nothing and prove nothing about
           dilution;
      (ii) the change from lambda-hat actually MOVING -- the only part
           attributable to dilution.
    Plus the fraction of splits on which lambda-hat differs at all between the two
    arms. If lambda-hat is identical on most splits, the "leverage multiplier"
    story is mostly term (i), i.e. arithmetic, not leverage.

PROTOCOL: canonical (canonical_fst.py) -- ccrc_v3's lam_grid, missing_numbers'
bound-specific k_min, ascending fixed sequence stopping at the first
non-rejection, single test at full delta. The repair gate is ccrc_v3's decoupled
gate: fixed evidence-margin percentile q, independent of lambda.

Run:  python3 ccrc_gains_stats.py
"""
import os, numpy as np, warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
import canonical_fst as F
import ccrc_v3 as C

ALPHAS = (0.10, 0.15)
QS = (None, 0.10, 0.25, 0.50)
Q_MAIN = 0.10
REPS = F.REPS
DELTA = F.DELTA


def one_setting(name, X, ok_o, ok_r, mg, reps=REPS):
    """Runs every (alpha, q) cell on identical paired splits. Returns nested dict."""
    n = len(ok_o)
    res = {a: {q: dict(cov=[], rte=[], ral=[], lam=[], rep_mass=[],
                       r_a=[], r_r=[]) for q in QS} for a in ALPHAS}
    dec = {a: dict(rep_term=[], dil_term=[], d_acc=[], d_rep=[], moved=[],
                   cov_f=[], cov_c=[], both=[]) for a in ALPHAS}
    for (tr, cal, te), s in zip(F.SPLITS(n, reps), F.fitted_scores(X, ok_o, reps)):
        if s is None: continue
        s_cal, m_cal = s[cal], mg[cal]
        for a in ALPHAS:
            lam_by_q = {}
            for q in QS:
                rep_mask_cal = None if q is None else (m_cal >= np.quantile(m_cal, 1 - q))
                lam = F.fst(s_cal, ok_o[cal], a, DELTA, F.cp_upper,
                            rep_mask_cal=rep_mask_cal, ok_rep_cal=ok_r[cal])
                lam_by_q[q] = lam
                R = res[a][q]
                R["lam"].append(lam)
                if lam is None:
                    R["cov"].append(0.0); R["rte"].append(None); R["ral"].append(None)
                    R["rep_mass"].append(0.0); R["r_a"].append(None); R["r_r"].append(None)
                    continue
                thr = np.quantile(s_cal, 1 - lam)
                tm = None if q is None else np.quantile(m_cal, 1 - q)
                for ii, key in ((te, "rte"), (np.arange(n), "ral")):
                    acc = s[ii] >= thr
                    rep = np.zeros_like(acc) if q is None else ((~acc) & (mg[ii] >= tm))
                    k = int(acc.sum() + rep.sum())
                    e = int((1 - ok_o[ii][acc]).sum() + (1 - ok_r[ii][rep]).sum())
                    R[key].append(e / k if k else None)
                acc = s[te] >= thr
                rep = np.zeros_like(acc) if q is None else ((~acc) & (mg[te] >= tm))
                R["cov"].append((int(acc.sum()) + int(rep.sum())) / len(te))
                R["rep_mass"].append(int(rep.sum()) / len(te))
                # region risks at the SELECTED policy (test fold)
                R["r_a"].append(float((1 - ok_o[te][acc]).mean()) if acc.sum() else None)
                R["r_r"].append(float((1 - ok_r[te][rep]).mean()) if rep.sum() else None)

            # ---------- dilution decomposition, filtering vs CCRC(q=Q_MAIN)
            lf, lc = lam_by_q[None], lam_by_q[Q_MAIN]
            D = dec[a]
            D["moved"].append(None if (lf is None or lc is None) else (lf != lc))
            if lf is None or lc is None:
                D["both"].append(False)
                for k_ in ("rep_term", "dil_term", "d_acc", "d_rep", "cov_f", "cov_c"):
                    D[k_].append(None)
                continue
            D["both"].append(True)
            tm = np.quantile(m_cal, 1 - Q_MAIN)
            thr_f, thr_c = np.quantile(s_cal, 1 - lf), np.quantile(s_cal, 1 - lc)
            acc_f = s[te] >= thr_f
            rep_f = (~acc_f) & (mg[te] >= tm)          # repairs available AT lambda_filt
            acc_c = s[te] >= thr_c
            rep_c = (~acc_c) & (mg[te] >= tm)
            nt = len(te)
            cov_f = acc_f.sum() / nt
            cov_c = (acc_c.sum() + rep_c.sum()) / nt
            D["cov_f"].append(cov_f * 100); D["cov_c"].append(cov_c * 100)
            # (i) the repaired items themselves, at the threshold filtering chose
            D["rep_term"].append(rep_f.sum() / nt * 100)
            # (ii) everything attributable to lambda-hat moving
            D["dil_term"].append((cov_c - cov_f - rep_f.sum() / nt) * 100)
            D["d_acc"].append((acc_c.sum() - acc_f.sum()) / nt * 100)
            D["d_rep"].append((rep_c.sum() - rep_f.sum()) / nt * 100)
    return res, dec


def main():
    print(__doc__)
    SET = C.settings()
    ALL = {}
    for name, (X, ok_o, ok_r, mg) in SET:
        ALL[name] = (one_setting(name, X, ok_o, ok_r, mg), len(ok_o), 1 - ok_o.mean())

    # =================================================== Table 5, canonical
    for a in ALPHAS:
        print("\n" + "=" * 118)
        print(f"TABLE 5 (tab:ccrc) CANONICAL -- alpha={a:.2f}, delta={DELTA:.2f}, "
              f"{REPS} paired three-way splits, q={Q_MAIN}")
        print("=" * 118)
        print(f"{'setting':22s}{'n':>5s}{'mu':>7s}"
              f"{'Filter':>16s}{'ab%':>6s}{'CCRC':>16s}{'ab%':>6s}"
              f"{'gain':>8s}{'sd':>7s}{'95% CI':>18s}{'p':>9s}")
        for name, ((res, dec), n, mu) in ALL.items():
            f_ = res[a][None]; c_ = res[a][Q_MAIN]
            cf = np.array(f_["cov"]) * 100; cc = np.array(c_["cov"]) * 100
            g = F.gain_stats(cf, cc)
            print(f"{name:22s}{n:5d}{mu*100:6.1f}%"
                  f"{cf.mean():9.1f}+-{cf.std(ddof=1):4.1f}%{F.abort_rate(f_['lam'])*100:5.1f}%"
                  f"{cc.mean():9.1f}+-{cc.std(ddof=1):4.1f}%{F.abort_rate(c_['lam'])*100:5.1f}%"
                  f"{g.mean:+8.2f}{g.sd:7.2f}"
                  f"  [{g.lo:+6.2f},{g.hi:+6.2f}]{g.p_t:9.2g}")
        print("  ab% = abort rate: fraction of splits on which the fixed sequence certifies")
        print("        nothing, deploying 0% coverage. Included in the coverage mean.")
        print("  p is a TWO-SIDED paired t-test over splits, conditional on this dataset.")

    # ============================================== validity audit (task 4)
    print("\n" + "=" * 118)
    print(f"VALIDITY AUDIT -- Pr[realised risk > alpha] against delta={DELTA:.2f}")
    print("The guarantee is Pr[Risk <= alpha] >= 1-delta. Mean risk is NOT the guarantee")
    print("and is not reported. Aborted splits emit nothing and are excluded from the")
    print("denominators; the abort rate is given beside so the reader can reconstruct.")
    print("=" * 118)
    print(f"{'setting':22s}{'alpha':>6s}{'arm':>8s}{'exc(te)':>9s}{'exc(all)':>10s}"
          f"{'abort':>7s}{'verdict':>10s}")
    for name, ((res, dec), n, mu) in ALL.items():
        for a in ALPHAS:
            for q in QS:
                R = res[a][q]
                xa = F.exceedance(R["ral"], a)
                arm = "filter" if q is None else f"q={q:g}"
                print(f"{name:22s}{a:6.2f}{arm:>8s}"
                      f"{F.exceedance(R['rte'],a)*100:8.1f}%{xa*100:9.1f}%"
                      f"{F.abort_rate(R['lam'])*100:6.1f}%"
                      f"{('VIOLATED' if xa > DELTA + 1e-12 else 'OK'):>10s}")
    print("  exc(te)  = held-out test fold (unbiased, noisy: n_te = n/3)")
    print("  exc(all) = policy re-scored on all n items (low-noise proxy for population risk)")

    # ============================================ dilution decomposition (5)
    for a in ALPHAS:
        print("\n" + "=" * 118)
        print(f"DILUTION DECOMPOSITION (blocker 6) -- alpha={a:.2f}, q={Q_MAIN}, "
              f"{REPS} paired splits")
        print("Splits where BOTH arms certify. gain = (i) repaired items + (ii) lambda-hat moved")
        print("=" * 118)
        print(f"{'setting':22s}{'splits':>7s}{'lam moves':>10s}"
              f"{'gain':>9s}{'(i) repairs':>12s}{'(ii) dilution':>14s}"
              f"{'  95% CI (ii)':>18s}{'p(ii)':>8s}{'d_acc':>7s}{'d_rep':>7s}")
        for name, ((res, dec), n, mu) in ALL.items():
            D = dec[a]
            keep = [i for i, b in enumerate(D["both"]) if b]
            if not keep:
                print(f"{name:22s}{0:7d}   both arms never certify on the same split")
                continue
            g = lambda k: np.array([D[k][i] for i in keep], float)
            gain = g("cov_c") - g("cov_f")
            rep_t, dil_t = g("rep_term"), g("dil_term")
            moved = np.array([D["moved"][i] for i in keep], bool)
            gd = F.Gain(dil_t)
            print(f"{name:22s}{len(keep):7d}{moved.mean()*100:9.1f}%"
                  f"{gain.mean():+9.2f}{rep_t.mean():+12.2f}{dil_t.mean():+14.2f}"
                  f"  [{gd.lo:+6.2f},{gd.hi:+6.2f}]{gd.p_t:8.2g}"
                  f"{g('d_acc').mean():+7.2f}{g('d_rep').mean():+7.2f}")
        print("  lam moves = fraction of splits on which lambda-hat DIFFERS between filtering")
        print("              and CCRC. Where this is low, term (ii) is structurally near zero")
        print("              and the gain is just the repaired items being emitted.")
        print("  (i)  repaired items emitted at filtering's own threshold -- free, automatic,")
        print("       and NOT evidence of dilution.")
        print("  (ii) net effect of lambda-hat moving = d_acc + d_rep. d_acc is the extra")
        print("       ACCEPTED mass (the dilution payoff); d_rep is negative because a more")
        print("       permissive lambda absorbs items that were previously repairs.")
        print("  Leverage multiplier = gain / repaired mass. Its honest version is")
        print("  1 + (ii)/(i): anything above 1x comes from term (ii) alone.")
        for name, ((res, dec), n, mu) in ALL.items():
            D = dec[a]; keep = [i for i, b in enumerate(D["both"]) if b]
            if not keep: continue
            gv = lambda k: np.array([D[k][i] for i in keep], float)
            rt, dt = gv("rep_term").mean(), gv("dil_term").mean()
            tot = (gv("cov_c") - gv("cov_f")).mean()
            mult = tot / rt if rt > 1e-9 else float("nan")
            print(f"    {name:22s} repaired mass {rt:5.2f}%   gain {tot:+5.2f} pp   "
                  f"multiplier {mult:4.2f}x   of which dilution {dt:+5.2f} pp")

    # ==================================================== q sensitivity (13)
    print("\n" + "=" * 118)
    print(f"TABLE 13 (tab:qsens) CANONICAL -- gain over filtering by repair gate q, "
          f"{REPS} paired splits")
    print("=" * 118)
    print(f"{'setting':22s}{'alpha':>6s}" + "".join(
        f"{f'q={q:g}':>26s}" for q in QS if q is not None))
    cells = {q: [] for q in QS if q is not None}
    for name, ((res, dec), n, mu) in ALL.items():
        for a in ALPHAS:
            row = f"{name:22s}{a:6.2f}"
            base = np.array(res[a][None]["cov"]) * 100
            for q in QS:
                if q is None: continue
                g = F.gain_stats(base, np.array(res[a][q]["cov"]) * 100)
                cells[q].append(g.mean)
                row += f"{g.mean:+8.2f} [{g.lo:+5.2f},{g.hi:+5.2f}]".rjust(26)
            print(row)
    print(f"\n{'q':>6s}{'cells positive':>16s}{'range':>22s}{'worst cell':>12s}")
    for q, v in cells.items():
        v = np.array(v)
        print(f"{q:6g}{f'{(v>0).sum()}/{len(v)}':>16s}"
              f"{f'{v.min():+.2f} to {v.max():+.2f}':>22s}{v.min():+12.2f}")
    print("\nNote: the eight-cell count in the manuscript excludes AMBER; all nine cells are")
    print("shown above, with AMBER included, because excluding the loss makes the")
    print("sensitivity table read more favourably than the data supports.")


if __name__ == "__main__":
    main()
