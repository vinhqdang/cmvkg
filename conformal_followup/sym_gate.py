"""
sym_gate.py -- Task 1: does a POLARITY-SYMMETRIC repair gate help?

BACKGROUND (the defect being tested)
------------------------------------
ccrc_v3.build() forms the channel-2 evidence margin as

    m = |o - THR|,     THR = 0.15   (OWLv2 operating point)

and the repair gate admits the top-q of m (q = 0.10 fixed a priori).  Because the
detector confidence o lives in [0,1] and THR = 0.15, the two polarities have
*structurally unequal* dynamic range:

    "object IS present"  (o >= THR):  m in [0, 0.85]
    "object is ABSENT"   (o <  THR):  m in [0, 0.15]

Measured q=0.10 gate values are 0.383 / 0.357 / 0.366 / 0.366 / 0.209, i.e. ABOVE the
entire negative-side range in 4 of 5 settings.  So 100% of admitted repairs are
"detector says present", CCRC can only repair MISSED objects, and it can never repair
the false-positive existence hallucination POPE-adversarial is built to elicit.

WHAT THIS SCRIPT ADDS
---------------------
Two polarity-symmetric constructions, both with the gate quantile q held FIXED and
pre-specified at exactly the value the current method uses (q = 0.10).  Only the
acceptance threshold lambda is calibrated, as in ccrc_v3.

  sym-2gate  (main proposal: two pre-specified gates)
      mp = (o - THR) / (1 - THR)        in [0,1] for present claims
      mn = (THR - o) / THR              in [0,1] for absent  claims
      t_pos = Q_{1-q}( mp | cal, o >= THR )
      t_neg = Q_{1-q}( mn | cal, o <  THR )
      REPAIR  if  (o >= THR and mp >= t_pos)  or  (o < THR and mn >= t_neg)

  sym-norm  (single gate on a properly normalised two-sided margin)
      m_sym = mp if o >= THR else mn
      t     = Q_{1-q}( m_sym | cal )
      REPAIR  if  m_sym >= t

VALIDITY / NESTING
------------------
In both constructions the repair mask M is a function of the calibration fold and of
o ONLY -- it does not depend on lambda.  Therefore the emitted set

    E_lam = A_lam  u  (M \\ A_lam)  =  A_lam u M

is a union of a lambda-nested increasing family A_lam with a lambda-CONSTANT set, hence
E_lam is itself nested increasing in lambda.  Exactly as for the one-directional gate, a
SINGLE ascending fixed-sequence test over lambda at FULL delta is valid; no union bound
and no delta split is introduced.  (This is the same argument as ccrc_v3 docstring; the
symmetric gate changes *which* items are in M, not the lambda-geometry of the family.)

Note also that sym-2gate does not enlarge the repair budget: it admits the top-q of each
polarity, so total repair mass is still ~q of items -- the change is a REDISTRIBUTION of
a fixed budget from positive-only to both polarities, not an expansion.

PROTOCOL
--------
Identical to ccrc_v3: 3-way disjoint split (fit score / calibrate lambda / test),
ascending-lambda fixed-sequence test stopping at the first non-rejection,
Clopper-Pearson upper bounds, delta = 0.10, logistic-regression correctness score on the
same 4 features as ccrc_v3.build().  All arms share the SAME splits and the SAME fitted
score, so every arm-to-arm comparison is exactly paired.

Run:  python3 sym_gate.py
      python3 sym_gate.py --reps 400
"""
import argparse, json, os, sys, numpy as np, warnings
warnings.filterwarnings("ignore")
from scipy import stats
from sklearn.linear_model import LogisticRegression
import ccrc_v3 as C                      # read-only reuse of the canonical machinery

_here = os.path.dirname(os.path.abspath(__file__))
J = lambda f: json.load(open(os.path.join(_here, f)))
DELTA = C.DELTA          # 0.10
THR = C.THR              # 0.15
Q_FIXED = 0.10           # pre-specified repair-gate quantile, NOT tuned (same as v3)
ALPHAS = (0.10, 0.15)
#   filter     : no repair channel at all (Prop.3-bound baseline)
#   onedir     : the CURRENT CCRC v3 gate, top-q of |o - THR| pooled over polarities
#   pos-2gate  : top-q of the NORMALISED POSITIVE margin only  (isolates re-scaling)
#   neg-2gate  : top-q of the NORMALISED NEGATIVE margin only  (the new capability alone)
#   sym-2gate  : union of the two above -- the polarity-symmetric proposal
#   sym-norm   : single gate on the two-sided normalised margin
ARMS = ("filter", "onedir", "pos-2gate", "neg-2gate", "sym-2gate", "sym-norm")


# --------------------------------------------------------------------- data
def build_full(p_yes, answer, gold, owl):
    """Same as ccrc_v3.build() but ALSO returns the raw detector confidence o,
    which the symmetric gate needs in order to know the polarity of each claim."""
    p = np.asarray(p_yes, float); a = np.asarray(answer, int)
    g = np.asarray(gold, int);    o = np.asarray(owl, float)
    keep = o >= 0
    p, a, g, o = p[keep], a[keep], g[keep], o[keep]
    conf = np.abs(p - .5) * 2
    dn = (o - o.min()) / (o.max() - o.min() + 1e-9)          # ccrc_v3.mm
    X = np.column_stack([p, conf, dn, np.where(a == 1, dn, 1 - dn)])
    ok_o = (a == g).astype(int)                              # original answer correct
    ok_r = ((o >= THR).astype(int) == g).astype(int)          # detector answer correct
    return X, ok_o, ok_r, o, a, g


def load_settings():
    S = []
    r = J("raw_scores.json"); ow = J("owlv2_scores.json")
    S.append(("POPE-1500 LLaVA", build_full(r["p_yes"], r["answer"], r["gold"],
                                            ow["ground_det"])))
    for f, lab, pk in [("exp7_pope.json", "POPE-adv LLaVA", "p_yes"),
                       ("exp8_qwen_pope.json", "POPE-adv Qwen2-VL", "p_yes"),
                       ("exp9_vcd_pope.json", "POPE-adv LLaVA+VCD", "p_vcd"),
                       ("exp12_amber_all.json", "AMBER(d) LLaVA", "p_yes")]:
        d = J(f)
        S.append((lab, build_full(d[pk], d["answer"], d["gold"], d["owl"])))
    return S


# ------------------------------------------------------- repair masks (lam-free)
def margins(o):
    """Returns (one-directional margin, positive-side normalised, negative-side
    normalised, polarity mask)."""
    pos = o >= THR
    m1 = np.abs(o - THR)
    mp = (o - THR) / (1.0 - THR)          # meaningful only where pos
    mn = (THR - o) / THR                  # meaningful only where ~pos
    return m1, mp, mn, pos


def repair_mask(arm, o, cal, q=Q_FIXED):
    """Pre-specified, lambda-INDEPENDENT repair region, thresholds fitted on `cal`."""
    m1, mp, mn, pos = margins(o)
    if arm == "filter":
        return np.zeros(len(o), bool)
    if arm == "onedir":                            # current CCRC v3
        return m1 >= np.quantile(m1[cal], 1 - q)
    if arm in ("sym-2gate", "pos-2gate", "neg-2gate"):   # pre-specified per-polarity gates
        cp, cn = cal[pos[cal]], cal[~pos[cal]]
        M = np.zeros(len(o), bool)
        if arm in ("sym-2gate", "pos-2gate") and len(cp) >= 3:
            M |= pos & (mp >= np.quantile(mp[cp], 1 - q))
        if arm in ("sym-2gate", "neg-2gate") and len(cn) >= 3:
            M |= (~pos) & (mn >= np.quantile(mn[cn], 1 - q))
        return M
    if arm == "sym-norm":                          # one gate on normalised margin
        ms = np.where(pos, mp, mn)
        return ms >= np.quantile(ms[cal], 1 - q)
    raise ValueError(arm)


# ------------------------------------------------------------------- FST
def fst(s, cal, M, ok_o, ok_r, alpha):
    """Ascending-lambda fixed sequence, CP at full delta; stop at first non-rejection.
    Mirrors ccrc_v3.calibrate() exactly, with M as the fixed repair region."""
    sc = s[cal]
    best = None
    for lam in C.lam_grid(len(cal), alpha, DELTA):
        acc = sc >= np.quantile(sc, 1 - lam)
        rep = (~acc) & M[cal]
        k = int(acc.sum() + rep.sum())
        if k == 0:
            continue
        e = int((1 - ok_o[cal][acc]).sum() + (1 - ok_r[cal][rep]).sum())
        if C.cp_upper(e, k, DELTA) <= alpha:
            best = lam
        else:
            break
    return best


def apply_policy(lam, s, cal, idx, M):
    thr = np.quantile(s[cal], 1 - lam)
    acc = s[idx] >= thr
    rep = (~acc) & M[idx]
    return acc, rep


# ------------------------------------------------------------------- driver
def run_setting(name, X, ok_o, ok_r, o, ans, gold, reps, seed=0):
    n = len(ok_o)
    _, _, _, pos = margins(o)
    allidx = np.arange(n)
    rng = np.random.default_rng(seed)
    # A repaired item is a corrected FALSE-POSITIVE existence hallucination iff the VLM
    # said "yes", the gold is "no", and the substituted detector answer is "no".
    fp_fix = (ans == 1) & (gold == 0) & (~pos)
    # ... and a corrected MISSED object (false negative) iff the reverse.
    fn_fix = (ans == 0) & (gold == 1) & pos
    res = {a: {arm: {k: [] for k in
                     ("cov", "risk", "riskall", "rmass", "rpos", "rneg", "abort",
                      "fpfix", "fnfix", "prec_pos", "prec_neg")}
               for arm in ARMS} for a in ALPHAS}
    done = 0
    while done < reps:
        idxp = rng.permutation(n); t = n // 3
        tr, cal, te = idxp[:t], idxp[t:2 * t], idxp[2 * t:]
        if len(np.unique(ok_o[tr])) < 2:
            continue
        s = LogisticRegression(max_iter=1000).fit(X[tr], ok_o[tr]).predict_proba(X)[:, 1]
        masks = {arm: repair_mask(arm, o, cal) for arm in ARMS}
        for a in ALPHAS:
            for arm in ARMS:
                M = masks[arm]
                lam = fst(s, cal, M, ok_o, ok_r, a)
                R = res[a][arm]
                if lam is None:
                    for k in ("cov", "rmass", "rpos", "rneg", "fpfix", "fnfix"):
                        R[k].append(0.0)
                    R["abort"].append(1.0)
                    R["risk"].append(0.0); R["riskall"].append(0.0)
                    continue
                R["abort"].append(0.0)
                acc, rep = apply_policy(lam, s, cal, te, M)
                k = int(acc.sum() + rep.sum())
                R["cov"].append(k / len(te))
                R["rmass"].append(int(rep.sum()) / len(te))
                rp_m, rn_m = rep & pos[te], rep & ~pos[te]
                R["rpos"].append(int(rp_m.sum()) / len(te))
                R["rneg"].append(int(rn_m.sum()) / len(te))
                R["fpfix"].append(int((rep & fp_fix[te]).sum()) / len(te))
                R["fnfix"].append(int((rep & fn_fix[te]).sum()) / len(te))
                if rp_m.sum(): R["prec_pos"].append(float(ok_r[te][rp_m].mean()))
                if rn_m.sum(): R["prec_neg"].append(float(ok_r[te][rn_m].mean()))
                if k:
                    e = int((1 - ok_o[te][acc]).sum() + (1 - ok_r[te][rep]).sum())
                    R["risk"].append(e / k)
                else:
                    R["risk"].append(0.0)
                aa, rr = apply_policy(lam, s, cal, allidx, M)
                ka = int(aa.sum() + rr.sum())
                if ka:
                    ea = int((1 - ok_o[aa]).sum() + (1 - ok_r[rr]).sum())
                    R["riskall"].append(ea / ka)
                else:
                    R["riskall"].append(0.0)
        done += 1
    return res


def paired(a, b):
    """mean gain (pp), 95% CI on the paired-split mean, two-sided paired t p-value,
    Wilcoxon signed-rank p-value.  Splits, not fresh datasets -- see caveat in report."""
    d = (np.asarray(b) - np.asarray(a)) * 100
    m = d.mean(); nn = len(d)
    se = d.std(ddof=1) / np.sqrt(nn) if nn > 1 else 0.0
    tcrit = stats.t.ppf(0.975, nn - 1) if nn > 1 else 0.0
    lo, hi = m - tcrit * se, m + tcrit * se
    sd = d.std(ddof=1) if nn > 1 else 0.0
    if np.allclose(d, 0):
        return m, lo, hi, 1.0, 1.0, sd
    pt = float(stats.ttest_rel(np.asarray(b), np.asarray(a)).pvalue)
    try:
        pw = float(stats.wilcoxon(d, zero_method="wilcox").pvalue)
    except Exception:
        pw = float("nan")
    return m, lo, hi, pt, pw, sd


def diagnose():
    """Split-free structural diagnostic: WHERE do the false-positive existence
    hallucinations live on the negative side of the detector, and can ANY negative gate
    reach them without destroying repair precision?  This is what decides whether a
    symmetric gate can do what the manuscript hopes, independently of any calibration."""
    print("=" * 108)
    print("STRUCTURAL DIAGNOSTIC (no splits): can a negative-side gate reach the")
    print("false-positive existence hallucinations at all?")
    print("=" * 108)
    for name, (X, ok_o, ok_r, o, ans, gold) in load_settings():
        n = len(o)
        m1, mp, mn, pos = margins(o)
        fp = (ans == 1) & (gold == 0)         # VLM hallucinated an absent object
        neg = ~pos
        det_no_right = neg & (gold == 0)
        print(f"\n### {name}  n={n}")
        print(f"  false-positive existence hallucinations: {int(fp.sum())} "
              f"({fp.mean()*100:.1f}% of items) -- the errors POPE-adv is built to elicit")
        print(f"  of those, detector DISAGREES (o < THR, so a repair would flip the "
              f"answer): {int((fp & neg).sum())} ({(fp&neg).sum()/max(fp.sum(),1)*100:.0f}%)")
        print(f"  {'neg-margin decile (mn)':24s}{'n':>6s}{'o range':>16s}"
              f"{'det acc':>9s}{'#FP-halluc reachable':>22s}{'cum. reachable':>16s}")
        mnn = mn[neg]; idxn = np.where(neg)[0]
        order = idxn[np.argsort(-mnn)]        # most confident 'absent' first
        cum = 0
        for d in range(10):
            lo_i, hi_i = int(d * len(order) / 10), int((d + 1) * len(order) / 10)
            sel = order[lo_i:hi_i]
            if len(sel) == 0: continue
            acc = ok_r[sel].mean() * 100
            nfp = int(fp[sel].sum()); cum += nfp
            print(f"  {'top ' + str((d+1)*10) + '% .. ':24s}{len(sel):6d}"
                  f"{f'[{o[sel].min():.3f},{o[sel].max():.3f}]':>16s}{acc:8.1f}%"
                  f"{nfp:22d}{cum:16d}")
        print(f"  => the q=0.10 negative gate covers the top row only: "
              f"{int(fp[order[:max(1,len(order)//10)]].sum())} FP hallucinations reachable "
              f"out of {int(fp.sum())}.")
    print()
    verify_nesting()


def verify_nesting(reps=30):
    """Empirical check of the validity precondition: for every arm the emitted set
    E_lam = ACCEPT_lam u REPAIR_lam must be NESTED INCREASING in lambda, otherwise a
    single ascending fixed-sequence test at full delta is not valid for that family."""
    print("=" * 108)
    print("NESTING CHECK -- E_lam must be monotone increasing in lambda for a single FST")
    print("at full delta to be valid.  Verified numerically on random splits.")
    print("=" * 108)
    grid_n = 40
    for name, (X, ok_o, ok_r, o, ans, gold) in load_settings():
        n = len(ok_o); rng = np.random.default_rng(1)
        bad = {arm: 0 for arm in ARMS}
        for _ in range(reps):
            idxp = rng.permutation(n); t = n // 3
            tr, cal = idxp[:t], idxp[t:2 * t]
            if len(np.unique(ok_o[tr])) < 2: continue
            s = LogisticRegression(max_iter=1000).fit(X[tr], ok_o[tr]).predict_proba(X)[:, 1]
            for arm in ARMS:
                M = repair_mask(arm, o, cal)
                prev = None
                for lam in np.linspace(0.05, 1.0, grid_n):
                    acc = s >= np.quantile(s[cal], 1 - lam)
                    E = acc | (M & ~acc)      # = acc | M
                    if prev is not None and not np.all(prev <= E):
                        bad[arm] += 1
                    prev = E
        print(f"  {name:22s}" + "".join(
            f"{arm}:{'OK' if bad[arm] == 0 else 'VIOLATED(' + str(bad[arm]) + ')':<10s} "
            for arm in ARMS))
    print("  All arms nested => single FST at full delta=0.10 is valid for each; no")
    print("  union bound and no delta split is introduced by the symmetric construction.")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=500)
    ap.add_argument("--json", type=str, default="")
    ap.add_argument("--diag-only", action="store_true")
    args = ap.parse_args()
    reps = args.reps
    diagnose()
    if args.diag_only:
        return

    print("=" * 108)
    print(f"TASK 1 -- polarity-symmetric repair gate.  q FIXED = {Q_FIXED} (pre-specified, "
          f"not tuned).  delta={DELTA}, {reps} paired splits")
    print("=" * 108)

    out = {}
    for name, (X, ok_o, ok_r, o, ans, gold) in load_settings():
        n = len(ok_o)
        m1, mp, mn, pos = margins(o)
        nfp = int(((ans == 1) & (gold == 0)).sum()); nfn = int(((ans == 0) & (gold == 1)).sum())
        print(f"\n### {name}   n={n}   mu={(1-ok_o.mean())*100:.1f}%   "
              f"det-positive={pos.mean()*100:.1f}%")
        print(f"    VLM errors: {nfp} false-POSITIVE existence hallucinations "
              f"({nfp/n*100:.1f}%), {nfn} missed objects ({nfn/n*100:.1f}%)")
        print(f"    one-directional q=0.10 gate on |o-THR| = "
              f"{np.quantile(m1, 1-Q_FIXED):.3f}   (max attainable by a NEGATIVE = {THR:.2f})")
        res = run_setting(name, X, ok_o, ok_r, o, ans, gold, reps)
        out[name] = {}
        for a in ALPHAS:
            print(f"\n  alpha={a:.2f}")
            print(f"    {'arm':11s}{'cov%':>8s}{'gain':>8s}{'95% CI':>18s}{'sd':>7s}"
                  f"{'p(t)':>10s}{'p(W)':>10s}{'risk%':>8s}{'exc_te':>8s}{'exc_all':>9s}"
                  f"{'abort':>7s}{'rep%':>7s}{'rep+':>7s}{'rep-':>7s}")
            base = np.array(res[a]["filter"]["cov"])
            row = {}
            for arm in ARMS:
                R = res[a][arm]
                cov = np.mean(R["cov"]) * 100
                rk = np.mean(R["risk"]) * 100
                xt = float((np.array(R["risk"]) > a + 1e-12).mean())
                xa = float((np.array(R["riskall"]) > a + 1e-12).mean())
                ab = float(np.mean(R["abort"]))
                rm, rp, rn = (np.mean(R["rmass"]) * 100, np.mean(R["rpos"]) * 100,
                              np.mean(R["rneg"]) * 100)
                g, lo, hi, pt, pw, sd = paired(base, R["cov"])
                gs = "--" if arm == "filter" else f"{g:+.2f}"
                cis = "--" if arm == "filter" else f"[{lo:+.2f},{hi:+.2f}]"
                sds = "--" if arm == "filter" else f"{sd:.2f}"
                ps = "--" if arm == "filter" else f"{pt:.2e}"
                pws = "--" if arm == "filter" else f"{pw:.2e}"
                flag = "!" if xa > DELTA + 1e-12 else " "
                print(f"    {arm:11s}{cov:7.2f}%{gs:>8s}{cis:>18s}{sds:>7s}{ps:>10s}{pws:>10s}"
                      f"{rk:7.2f}%{xt:8.3f}{xa:8.3f}{flag}{ab:7.3f}{rm:6.2f}%"
                      f"{rp:6.2f}%{rn:6.2f}%")
                row[arm] = dict(cov=cov, gain=None if arm == "filter" else g,
                                ci=None if arm == "filter" else [lo, hi],
                                sd=None if arm == "filter" else sd,
                                p_t=None if arm == "filter" else pt,
                                p_w=None if arm == "filter" else pw,
                                risk=rk, exc_te=xt, exc_all=xa, abort=ab,
                                rep_mass=rm, rep_pos=rp, rep_neg=rn)
            # contrasts against the one-directional gate, paired on identical splits
            for arm in ("pos-2gate", "neg-2gate", "sym-2gate", "sym-norm"):
                g, lo, hi, pt, pw, sd = paired(res[a]["onedir"]["cov"], res[a][arm]["cov"])
                print(f"      -> {arm:10s} vs one-directional CCRC: {g:+6.2f} pp "
                      f"[{lo:+.2f},{hi:+.2f}]  p(t)={pt:.3g}  p(W)={pw:.3g}")
                row[arm]["vs_onedir"] = dict(gain=g, ci=[lo, hi], p_t=pt, p_w=pw)
            # decomposition: what do the NEGATIVE-polarity repairs themselves buy?
            g, lo, hi, pt, pw, sd = paired(res[a]["pos-2gate"]["cov"],
                                           res[a]["sym-2gate"]["cov"])
            print(f"      => attributable to NEWLY-ADMITTED NEGATIVE repairs "
                  f"(sym-2gate - pos-2gate): {g:+.2f} pp [{lo:+.2f},{hi:+.2f}] "
                  f"p(t)={pt:.3g}  p(W)={pw:.3g}")
            row["neg_attribution"] = dict(gain=g, ci=[lo, hi], p_t=pt, p_w=pw)
            g2, lo2, hi2, pt2, pw2, _ = paired(res[a]["onedir"]["cov"],
                                               res[a]["pos-2gate"]["cov"])
            print(f"      => attributable to RE-SCALING the positive gate "
                  f"(pos-2gate - onedir):    {g2:+.2f} pp [{lo2:+.2f},{hi2:+.2f}] "
                  f"p(t)={pt2:.3g}  p(W)={pw2:.3g}")
            row["rescale_attribution"] = dict(gain=g2, ci=[lo2, hi2], p_t=pt2, p_w=pw2)
            # does the gate actually CORRECT existence hallucinations?
            print(f"      {'arm':11s}{'FP-halluc fixed':>17s}{'missed-obj fixed':>18s}"
                  f"{'repair prec +':>15s}{'repair prec -':>15s}")
            for arm in ARMS[1:]:
                R = res[a][arm]
                pp = np.mean(R["prec_pos"]) * 100 if R["prec_pos"] else float("nan")
                pn = np.mean(R["prec_neg"]) * 100 if R["prec_neg"] else float("nan")
                print(f"      {arm:11s}{np.mean(R['fpfix'])*100:16.3f}%"
                      f"{np.mean(R['fnfix'])*100:17.3f}%"
                      f"{pp:14.1f}%{pn:14.1f}%")
                row[arm]["fp_fixed"] = float(np.mean(R["fpfix"]) * 100)
                row[arm]["fn_fixed"] = float(np.mean(R["fnfix"]) * 100)
                row[arm]["prec_pos"] = None if np.isnan(pp) else pp
                row[arm]["prec_neg"] = None if np.isnan(pn) else pn
            out[name][f"alpha={a:.2f}"] = row
    if args.json:
        json.dump(out, open(args.json, "w"), indent=1)
    print("\nNOTES")
    print("  rep%/rep+/rep- : repaired mass as a fraction of the test fold, split by the")
    print("                   polarity of the detector's substituted answer ('present' /")
    print("                   'absent').  rep+ + rep- = rep%.")
    print("  exc_te / exc_all : validity audit = FRACTION of splits whose realised risk")
    print("                   exceeds alpha, on the held-out test fold and with the")
    print("                   selected policy re-scored on all n items.  Must be <= delta.")
    print("  abort          : fraction of splits certifying nothing (FST returns no lambda).")
    print("  CIs/p-values are over SPLITS of a fixed dataset (conditional on the data);")
    print("  they do not account for dataset-sampling variability.")


if __name__ == "__main__":
    main()
