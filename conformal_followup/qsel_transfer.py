"""
qsel_transfer.py -- Task 3: pricing the selection of the repair-gate quantile q.

THE PROBLEM
-----------
ALGORITHM.md sec.4 and sec.6 assert that q "is fixed a priori, not tuned per dataset",
and then present a sensitivity table over q in {0.10, 0.25, 0.50} on the SAME eight
evaluation cells (4 POPE-family settings x 2 alpha) that produce the headline gains,
concluding q=0.10 is "positive in all 8 cells" while 0.25 and 0.50 are "mixed".  Choosing
the value that wins on the evaluation cells and then reporting those cells' gains as
headline results is selection on the test set, however it is described.  The exposure has
to be priced.

WHAT THIS SCRIPT COMPUTES
-------------------------
  (1) The full q-sensitivity grid, so the selection surface is visible.
  (2) ORACLE-per-dataset q: an upper bound on what tuning could buy (and therefore an
      upper bound on how much the headline could be inflated by selection).
  (3) LEAVE-ONE-DATASET-OUT (LODO) transfer: for each evaluation dataset, q is chosen to
      maximise the mean gain on the OTHER datasets only, then applied to the held-out one.
      This is an honest estimate of the gain a practitioner would obtain, because the
      selection never sees the dataset it is scored on.  Reported both pooled over alpha
      and per-alpha (leave-one-CELL-out is also reported).
  (4) The MULTIPLICITY-CORRECTED alternative: run the fixed-sequence test for all three q
      candidates at delta/3 each, so all three certificates hold simultaneously by a union
      bound, and then legitimately SELECT the best-certified q on the CALIBRATION fold.
      This makes data-driven q formally valid; the price is a tighter Clopper-Pearson
      bound, which raises the minimum certifiable emitted-set size
      k_min = ceil(ln delta / ln(1-alpha)) and costs coverage.  That cost is measured.

PROTOCOL is identical to ccrc_v3 in every other respect (3-way disjoint split, ascending
fixed sequence stopping at the first non-rejection, Clopper-Pearson, logistic score on
the same features).  All arms share splits, so contrasts are exactly paired.

Run:  python3 qsel_transfer.py
      python3 qsel_transfer.py --reps 400
"""
import argparse, json, os, numpy as np, warnings
warnings.filterwarnings("ignore")
from scipy import stats
from sklearn.linear_model import LogisticRegression
import ccrc_v3 as C
from sym_gate import load_settings, repair_mask, fst, apply_policy, paired

DELTA = C.DELTA
QS = (0.10, 0.25, 0.50)
ALPHAS = (0.10, 0.15)


def kmin(alpha, delta):
    return int(np.ceil(np.log(delta) / np.log(1 - alpha)))


def run_setting(X, ok_o, ok_r, o, reps, seed=0):
    """Returns cov[alpha][key] over splits, where key is
       'filter'              -- no repair
       ('q', q)              -- one-directional gate at q, single FST at FULL delta
       ('q3', q)             -- same but FST at delta/3 (for the union-bound family)
       'sel3'                -- delta/3 family + calibration-fold selection of q
    Also returns the FST-certified calibration coverage used by 'sel3'."""
    n = len(ok_o); allidx = np.arange(n)
    rng = np.random.default_rng(seed)
    keys = ["filter"] + [("q", q) for q in QS] + [("q3", q) for q in QS] + ["sel3"]
    res = {a: {k: {"cov": [], "risk": [], "riskall": [], "abort": [], "qpick": []}
               for k in keys} for a in ALPHAS}
    done = 0
    while done < reps:
        idxp = rng.permutation(n); t = n // 3
        tr, cal, te = idxp[:t], idxp[t:2 * t], idxp[2 * t:]
        if len(np.unique(ok_o[tr])) < 2:
            continue
        s = LogisticRegression(max_iter=1000).fit(X[tr], ok_o[tr]).predict_proba(X)[:, 1]
        M = {q: repair_mask("onedir", o, cal, q=q) for q in QS}
        Mz = np.zeros(n, bool)

        def record(a, key, lam, mask, qpick=np.nan):
            R = res[a][key]
            R["qpick"].append(qpick)
            if lam is None:
                R["cov"].append(0.0); R["risk"].append(0.0)
                R["riskall"].append(0.0); R["abort"].append(1.0); return
            R["abort"].append(0.0)
            acc, rep = apply_policy(lam, s, cal, te, mask)
            k = int(acc.sum() + rep.sum())
            R["cov"].append(k / len(te))
            e = int((1 - ok_o[te][acc]).sum() + (1 - ok_r[te][rep]).sum()) if k else 0
            R["risk"].append(e / k if k else 0.0)
            aa, rr = apply_policy(lam, s, cal, allidx, mask)
            ka = int(aa.sum() + rr.sum())
            ea = int((1 - ok_o[aa]).sum() + (1 - ok_r[rr]).sum()) if ka else 0
            R["riskall"].append(ea / ka if ka else 0.0)

        for a in ALPHAS:
            record(a, "filter", fst(s, cal, Mz, ok_o, ok_r, a), Mz)
            for q in QS:
                record(a, ("q", q), fst(s, cal, M[q], ok_o, ok_r, a), M[q])
            # ---- union-bound family: every candidate certified at delta/3
            d3 = DELTA / len(QS)
            lam3, cov3 = {}, {}
            for q in QS:
                lam = fst_delta(s, cal, M[q], ok_o, ok_r, a, d3)
                lam3[q] = lam
                record(a, ("q3", q), lam, M[q])
                if lam is None:
                    cov3[q] = -1.0
                else:
                    acc, rep = apply_policy(lam, s, cal, cal, M[q])
                    cov3[q] = (int(acc.sum()) + int(rep.sum())) / len(cal)
            # selection on the CALIBRATION fold only (legitimate: all 3 certified at d/3)
            qb = max(QS, key=lambda q: cov3[q])
            record(a, "sel3", lam3[qb], M[qb], qpick=qb if cov3[qb] > 0 else np.nan)
        done += 1
    return res


def fst_delta(s, cal, M, ok_o, ok_r, alpha, delta):
    """fst() at an arbitrary delta.  The lambda grid start is recomputed from k_min(delta)
    exactly as missing_numbers.fst does, so a tighter delta correctly starts the sequence
    later and can abort it sooner."""
    sc = s[cal]
    km = kmin(alpha, delta)
    lam0 = min(0.95, max(1.5 * km / max(len(cal), 1), 0.05))
    best = None
    for lam in np.linspace(lam0, 1.0, 40):
        acc = sc >= np.quantile(sc, 1 - lam)
        rep = (~acc) & M[cal]
        k = int(acc.sum() + rep.sum())
        if k == 0:
            continue
        if k < km:
            break
        e = int((1 - ok_o[cal][acc]).sum() + (1 - ok_r[cal][rep]).sum())
        if C.cp_upper(e, k, delta) <= alpha:
            best = lam
        else:
            break
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=500)
    args = ap.parse_args()

    print("=" * 104)
    print(f"TASK 3 -- pricing the selection of the repair-gate quantile q.  "
          f"delta={DELTA}, {args.reps} paired splits")
    print("=" * 104)
    print(f"\n  k_min = ceil(ln delta / ln(1-alpha)), the smallest emitted-set size "
          f"Clopper-Pearson can certify:")
    for a in ALPHAS:
        print(f"    alpha={a:.2f}:  delta={DELTA:.3f} -> k_min={kmin(a, DELTA):3d}    "
              f"delta/3={DELTA/3:.4f} -> k_min={kmin(a, DELTA/3):3d}   "
              f"(+{kmin(a, DELTA/3)-kmin(a, DELTA)} required clean samples)")

    S = load_settings()
    names = [nm for nm, _ in S]
    R = {}
    for nm, (X, ok_o, ok_r, o, _ans, _gold) in S:
        R[nm] = run_setting(X, ok_o, ok_r, o, args.reps)

    # ---------------------------------------------------------- (1) grid
    print("\n" + "-" * 104)
    print("(1)  THE SELECTION SURFACE -- gain over filtering (pp), by q.  This is the table")
    print("     the manuscript reports; note that the SAME cells select q and score it.")
    print("-" * 104)
    gains = {}                    # gains[(name, alpha)][q] = mean paired gain pp
    for a in ALPHAS:
        print(f"\n  alpha={a:.2f}")
        print(f"    {'setting':20s}{'filter':>9s}" +
              "".join(f"{'q='+format(q,'g'):>22s}" for q in QS))
        for nm in names:
            base = R[nm][a]["filter"]["cov"]
            row = f"    {nm:20s}{np.mean(base)*100:8.2f}%"
            gains[(nm, a)] = {}
            for q in QS:
                g, lo, hi, pt, pw, sd = paired(base, R[nm][a][("q", q)]["cov"])
                gains[(nm, a)][q] = g
                row += f"{g:+8.2f} [{lo:+.1f},{hi:+.1f}]"
            print(row)
    print("\n    Cell-wise argmax of q (the quantity the manuscript's claim depends on):")
    for a in ALPHAS:
        picks = {nm: max(QS, key=lambda q: gains[(nm, a)][q]) for nm in names}
        print(f"      alpha={a:.2f}: " + "  ".join(f"{nm.split()[0]}:{picks[nm]:g}"
                                                  for nm in names))
    npos = {q: sum(1 for nm in names for a in ALPHAS if gains[(nm, a)][q] > 0) for q in QS}
    tot = len(names) * len(ALPHAS)
    print(f"    cells with a positive gain (out of {tot}): " +
          "  ".join(f"q={q:g}: {npos[q]}" for q in QS))

    # ------------------------------------------- (2)(3) oracle vs LODO transfer
    print("\n" + "-" * 104)
    print("(2)/(3)  ORACLE-per-dataset q  vs  LEAVE-ONE-DATASET-OUT transfer of q")
    print("-" * 104)
    print("     fixed  : q=0.10 as the manuscript reports it")
    print("     oracle : q chosen per (dataset, alpha) to maximise that cell's own gain")
    print("              -- an UPPER BOUND on what selection could buy")
    print("     LODO-A : q chosen to maximise mean gain over the OTHER datasets, pooled")
    print("              over both alpha, then applied to the held-out dataset")
    print("     LODO-C : q chosen to maximise mean gain over the other datasets at the")
    print("              SAME alpha (leave-one-cell-out within an alpha)")
    print()
    print(f"    {'setting':20s}{'alpha':>6s}{'fixed(q=.10)':>14s}{'oracle':>9s}{'q*':>5s}"
          f"{'LODO-A':>9s}{'q*':>5s}{'LODO-C':>9s}{'q*':>5s}{'LODOA-oracle':>14s}"
          f"{'LODOA-fixed':>13s}")
    summ = {k: [] for k in ("fixed", "oracle", "lodoA", "lodoC")}
    lodo_detail = []
    for nm in names:
        for a in ALPHAS:
            others = [x for x in names if x != nm]
            qA = max(QS, key=lambda q: np.mean([gains[(x, b)][q]
                                                for x in others for b in ALPHAS]))
            qC = max(QS, key=lambda q: np.mean([gains[(x, a)][q] for x in others]))
            gf, go = gains[(nm, a)][0.10], max(gains[(nm, a)].values())
            qo = max(QS, key=lambda q: gains[(nm, a)][q])
            gA, gC = gains[(nm, a)][qA], gains[(nm, a)][qC]
            print(f"    {nm:20s}{a:6.2f}{gf:+13.2f}{go:+9.2f}{qo:5g}{gA:+9.2f}{qA:5g}"
                  f"{gC:+9.2f}{qC:5g}{gA-go:+14.2f}{gA-gf:+13.2f}")
            summ["fixed"].append(gf); summ["oracle"].append(go)
            summ["lodoA"].append(gA); summ["lodoC"].append(gC)
            lodo_detail.append((nm, a, gf, go, qo, gA, qA, gC, qC))
    print("    " + "-" * 96)
    print(f"    {'MEAN over 10 cells':20s}{'':6s}{np.mean(summ['fixed']):+13.2f}"
          f"{np.mean(summ['oracle']):+9.2f}{'':5s}{np.mean(summ['lodoA']):+9.2f}{'':5s}"
          f"{np.mean(summ['lodoC']):+9.2f}")
    pope = [i for i, (nm, a, *_) in enumerate(lodo_detail) if not nm.startswith("AMBER")]
    print(f"    {'MEAN over the 8 POPE cells':26s}{np.mean([summ['fixed'][i] for i in pope]):+7.2f}"
          f"{np.mean([summ['oracle'][i] for i in pope]):+9.2f}{'':5s}"
          f"{np.mean([summ['lodoA'][i] for i in pope]):+9.2f}{'':5s}"
          f"{np.mean([summ['lodoC'][i] for i in pope]):+9.2f}")
    print(f"\n    selection exposure = oracle - LODO-A (mean over 10 cells) = "
          f"{np.mean(summ['oracle']) - np.mean(summ['lodoA']):+.2f} pp")
    print(f"    q=0.10 vs LODO-A   (mean over 10 cells) = "
          f"{np.mean(summ['fixed']) - np.mean(summ['lodoA']):+.2f} pp")
    qA_all = set(d[6] for d in lodo_detail)
    verdict = ("   <-- q TRANSFERS: every held-out fold selects the same value"
               if len(qA_all) == 1 else "   <-- q does NOT transfer uniquely")
    print(f"    q selected by LODO-A across all folds: {sorted(qA_all)}{verdict}")

    # ------------------------------------------------- (4) multiplicity version
    print("\n" + "-" * 104)
    print("(4)  MULTIPLICITY-CORRECTED q (union bound: each candidate certified at delta/3,")
    print("     then q selected on the CALIBRATION fold).  This makes data-driven q valid.")
    print("-" * 104)
    print(f"    {'setting':20s}{'alpha':>6s}{'filter':>9s}{'q=.10@d':>9s}{'q=.10@d/3':>11s}"
          f"{'sel@d/3':>9s}{'gain(sel)':>11s}{'cost vs q=.10@d':>17s}{'abort':>8s}"
          f"{'exc_all':>9s}")
    costs = []
    for nm in names:
        for a in ALPHAS:
            b = R[nm][a]["filter"]["cov"]
            cf = np.mean(b) * 100
            c1 = np.mean(R[nm][a][("q", 0.10)]["cov"]) * 100
            c3 = np.mean(R[nm][a][("q3", 0.10)]["cov"]) * 100
            cs = np.mean(R[nm][a]["sel3"]["cov"]) * 100
            gs, lo, hi, pt, pw, sd = paired(b, R[nm][a]["sel3"]["cov"])
            ab = np.mean(R[nm][a]["sel3"]["abort"])
            xa = float((np.array(R[nm][a]["sel3"]["riskall"]) > a + 1e-12).mean())
            print(f"    {nm:20s}{a:6.2f}{cf:8.2f}%{c1:8.2f}%{c3:10.2f}%{cs:8.2f}%"
                  f"{gs:+10.2f}{cs-c1:+16.2f}{ab:8.3f}{xa:9.3f}"
                  f"{'!' if xa > DELTA + 1e-12 else ''}")
            costs.append(cs - c1)
    print("    " + "-" * 96)
    print(f"    MEAN coverage cost of the delta/3 union-bound version vs the reported "
          f"q=0.10 at full delta: {np.mean(costs):+.2f} pp")
    print("\n    q selected on the calibration fold by the delta/3 procedure (mode per cell):")
    for nm in names:
        for a in ALPHAS:
            qp = np.array(R[nm][a]["sel3"]["qpick"], float)
            qp = qp[~np.isnan(qp)]
            if len(qp) == 0:
                print(f"      {nm:20s} alpha={a:.2f}: (all splits abort)"); continue
            frac = {q: float((qp == q).mean()) for q in QS}
            print(f"      {nm:20s} alpha={a:.2f}: " +
                  "  ".join(f"q={q:g}:{frac[q]*100:4.0f}%" for q in QS))
    print("\n  NOTE on the calibration-fold selection rule: it maximises CERTIFIED")
    print("  calibration coverage, which is why it usually picks the LOOSEST q -- a looser")
    print("  gate adds more repaired mass and, on the calibration fold, larger certified")
    print("  coverage.  It does not know that the loose gate transfers badly.  That is the")
    print("  honest reason a fixed strict q is preferable to a validly-selected one here.")


if __name__ == "__main__":
    main()
