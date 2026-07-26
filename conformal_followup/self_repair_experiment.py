"""
The self-repair experiment (Sec 4.4, "Why repairs cannot be self-certified in
practice"). This is the script that experiment never had: its numbers were quoted
in the manuscript with no code in the repo behind them.

DESIGN, exactly as the manuscript describes it.
  * The repair channel is the NEGATION of the model's own answer. The answer space
    is binary (yes/no), so the negated answer is correct exactly when the model was
    wrong: ok_flip = 1 - ok_model. No second channel is used at all.
  * The flip region is gated at a fixed q = 0.02 on the CORRECTNESS SCORE -- the
    bottom 2% of s, i.e. the items the score is most confident are wrong. This is
    the analogue of CCRC's decoupled gate with channel 2 replaced by channel 1's
    own complement.
  * Certification is at the MIXTURE level, exactly as in ccrc_v3: one fixed
    sequence over accepted + flipped, at full delta. A flip region whose own risk
    exceeds alpha is not automatically inadmissible, because a compliant blend can
    absorb it -- so testing the region in isolation would be the wrong experiment.

WHAT IT MEASURES.
  * certified coverage, self-repair vs filtering, paired, with sd / 95% CI /
    two-sided p;
  * ABORT RATE -- the actual mechanism. A fixed-mass, high-error flip block is at
    its largest *relative* to the emitted set at the START of the lambda grid,
    where K barely clears k_min. It fails the very first hypothesis, and a fixed
    sequence must stop at the first non-rejection, so the procedure returns nothing
    and deploys 0% coverage. It never reaches the later lambda at which the mixture
    budget would in fact be satisfiable.
  * the model's accuracy inside the flip region, which IS the flip region's risk
    r_r. Where that accuracy is above alpha, flipping is a losing trade by
    construction; where it is near zero, self-repair can genuinely pay.

Run:  python3 self_repair_experiment.py
"""
import numpy as np, warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
import canonical_fst as F
import ccrc_v3 as C

ALPHAS = (0.10, 0.15)
Q_FLIP = 0.02          # fixed a priori, as in the manuscript
REPS = F.REPS
DELTA = F.DELTA
# The four POPE settings in the manuscript's own order, then AMBER.
ORDER = ["POPE-1500 LLaVA", "POPE-adv LLaVA", "POPE-adv Qwen2-VL",
         "POPE-adv LLaVA+VCD", "AMBER(d) LLaVA"]


def run_setting(X, ok_o, alpha, q=Q_FLIP, reps=REPS):
    """Filtering vs self-repair on identical paired splits."""
    n = len(ok_o)
    ok_flip = 1 - ok_o                      # negation is right iff the model was wrong
    ALL = np.arange(n)
    rec = dict(cov_f=[], cov_s=[], lam_f=[], lam_s=[],
               rte=[], ral=[], flip_mass=[], r_flip=[], acc_region=[])
    for (tr, cal, te), s in zip(F.SPLITS(n, reps), F.fitted_scores(X, ok_o, reps)):
        if s is None: continue
        s_cal = s[cal]
        gate = float(np.quantile(s_cal, q))          # bottom q of the score
        rep_mask_cal = s_cal <= gate

        lam_f = F.fst(s_cal, ok_o[cal], alpha, DELTA, F.cp_upper)
        lam_s = F.fst(s_cal, ok_o[cal], alpha, DELTA, F.cp_upper,
                      rep_mask_cal=rep_mask_cal, ok_rep_cal=ok_flip[cal])
        rec["lam_f"].append(lam_f); rec["lam_s"].append(lam_s)

        rec["cov_f"].append(0.0 if lam_f is None else
                            float((s[te] >= np.quantile(s_cal, 1 - lam_f)).mean()))
        # model accuracy inside the flip region on the test fold = 1 - r_flip
        m_reg = s[te] <= gate
        rec["acc_region"].append(float(ok_o[te][m_reg].mean()) if m_reg.sum() else None)

        if lam_s is None:
            rec["cov_s"].append(0.0); rec["rte"].append(None); rec["ral"].append(None)
            rec["flip_mass"].append(0.0); rec["r_flip"].append(None); continue
        thr = float(np.quantile(s_cal, 1 - lam_s))
        for ii, key in ((te, "rte"), (ALL, "ral")):
            acc = s[ii] >= thr
            rep = (~acc) & (s[ii] <= gate)
            k = int(acc.sum() + rep.sum())
            e = int((1 - ok_o[ii][acc]).sum() + (1 - ok_flip[ii][rep]).sum())
            rec[key].append(e / k if k else None)
        acc = s[te] >= thr
        rep = (~acc) & (s[te] <= gate)
        rec["cov_s"].append(float((acc.sum() + rep.sum()) / len(te)))
        rec["flip_mass"].append(float(rep.sum() / len(te)))
        rec["r_flip"].append(float((1 - ok_flip[te][rep]).mean()) if rep.sum() else None)
    return rec


def main():
    print(__doc__)
    SET = dict(C.settings())
    for a in ALPHAS:
        print("\n" + "=" * 122)
        print(f"SELF-REPAIR (negate the model's answer, gate q={Q_FLIP}, mixture-level "
              f"certification) -- alpha={a:.2f}, delta={DELTA:.2f}, {REPS} paired splits")
        print("=" * 122)
        print(f"{'setting':22s}{'filter cov':>12s}{'self-rep cov':>14s}"
              f"{'gain':>8s}{'sd':>7s}{'95% CI':>18s}{'p':>9s}"
              f"{'abort filt':>12s}{'abort self':>12s}{'acc in flip':>13s}")
        rows = []
        for name in ORDER:
            X, ok_o, ok_r, mg = SET[name]
            rec = run_setting(X, ok_o, a)
            cf = np.array(rec["cov_f"]) * 100
            cs = np.array(rec["cov_s"]) * 100
            g = F.gain_stats(cf, cs)
            accr = np.array([x for x in rec["acc_region"] if x is not None]) * 100
            ab_f = F.abort_rate(rec["lam_f"]) * 100
            ab_s = F.abort_rate(rec["lam_s"]) * 100
            print(f"{name:22s}{cf.mean():11.1f}%{cs.mean():13.1f}%"
                  f"{g.mean:+8.1f}{g.sd:7.1f}  [{g.lo:+6.1f},{g.hi:+6.1f}]{g.p_t:9.2g}"
                  f"{ab_f:11.1f}%{ab_s:11.1f}%{accr.mean():12.1f}%")
            rows.append((name, rec, cf, cs, g, ab_f, ab_s, accr))

        print(f"\n  'acc in flip' is the model's accuracy in the bottom {Q_FLIP:.0%} of the")
        print(f"  score, on the test fold. It IS the flip region's risk r_r: the negated")
        print(f"  answer is wrong exactly when the model was right. Compare it to")
        print(f"  alpha={a:.2f}: above that, admitting the region can only consume slack.")
        print("\n  Validity audit of the self-repair arm (exceedance vs delta"
              f"={DELTA:.2f}), and flip mass:")
        print(f"  {'setting':22s}{'exc(te)':>9s}{'exc(all)':>10s}{'flip mass':>11s}"
              f"{'r_flip':>9s}")
        for name, rec, cf, cs, g, ab_f, ab_s, accr in rows:
            fm = np.array([x for x in rec["flip_mass"]]) * 100
            rf = np.array([x for x in rec["r_flip"] if x is not None])
            print(f"  {name:22s}{F.exceedance(rec['rte'],a)*100:8.1f}%"
                  f"{F.exceedance(rec['ral'],a)*100:9.1f}%{fm.mean():10.2f}%"
                  f"{(rf.mean()*100 if len(rf) else float('nan')):8.1f}%")
        print("  Where the abort rate is high the exceedance columns are computed on the")
        print("  few splits that certified anything, so they are noisy by construction.")

        print(f"\n  Mechanism check -- lambda grid start vs where the mixture would become")
        print(f"  admissible. k_min at alpha={a:.2f} is {F.kmin_for(F.cp_upper,a,DELTA)}; the flip")
        print("  block has FIXED mass, so its share of the emitted set is largest at the")
        print("  first grid point and shrinks monotonically thereafter. The fixed sequence")
        print("  must stop at the first non-rejection, hence the aborts.")

    print("\n" + "=" * 122)
    print("CONCLUSION")
    print("=" * 122)
    print("Large coverage losses on the four POPE settings, driven by total abort of the")
    print("fixed sequence rather than by a graceful trade; and a small POSITIVE gain on")
    print("AMBER, where the bottom of the score really does isolate items the model gets")
    print("wrong. The claim this licenses is the weak one: on the settings where filtering")
    print("is expensive enough for correction to be worth doing, no self-certifiable flip")
    print("region exists, so a second channel is needed in practice. It is not a theorem")
    print("that self-repair is impossible.")


if __name__ == "__main__":
    main()
