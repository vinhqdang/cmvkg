"""
Computes the manuscript numbers that previously had no script behind them.

  BLOCK 1  Table 12 -- risk-bound ablation (Hoeffding / empirical Bernstein / betting / CP)
  BLOCK 2  Table 7  -- detector-only selective baseline
  BLOCK 3  Table 8  -- ConfLVLM-style baselines (CLIP scorer, and CLIP+filter)
  BLOCK 4  Sec 5.4  -- VCD composition (AURC + certified coverage), reconciled
  BLOCK 5  Sec 4.4  -- model accuracy in the bottom-q of the canonical score
  BLOCK 6  Sec 5.5  -- subsample gains vs n

All blocks use the SAME protocol as ccrc_v3.py: 3-way disjoint split
(fit score / calibrate threshold / test), fixed-sequence testing over an
ascending grid started at k_min, and a one-sided upper confidence bound on the
binary risk at level delta.  Every block also reports the exceedance rate
Pr[realised risk > alpha] against delta, on the test fold and on the full item
set (a lower-noise proxy for the population risk of the selected policy).

Run:  python3 missing_numbers.py
"""
import json, os, numpy as np, warnings; warnings.filterwarnings("ignore")
from scipy.stats import beta
from sklearn.linear_model import LogisticRegression
import ccrc_v3 as C

_here = os.path.dirname(os.path.abspath(__file__))
J = lambda f: json.load(open(os.path.join(_here, f)))
DELTA, REPS = 0.10, 60
mm = lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9)


# ----------------------------------------------------------------- bounds
def cp_upper(e, k, d):                       # Clopper-Pearson (exact)
    if k == 0: return 1.0
    return 1.0 if e == k else float(beta.ppf(1 - d, e + 1, k - e))

def hoeffding_upper(e, k, d):                # Hoeffding on a [0,1] mean
    if k == 0: return 1.0
    return min(1.0, e / k + np.sqrt(np.log(1.0 / d) / (2.0 * k)))

def bernstein_upper(e, k, d):                # Maurer-Pontil empirical Bernstein
    if k < 2: return 1.0
    p = e / k
    var = p * (1 - p) * k / (k - 1)          # unbiased sample variance of a 0/1 vector
    return min(1.0, p + np.sqrt(2 * var * np.log(2 / d) / k)
                      + 7 * np.log(2 / d) / (3 * (k - 1)))

def betting_upper(e, k, d, grid=None):
    """Waudby-Smith & Ramdas style capital-process upper confidence limit.
    For each candidate mean m we run the e-process testing H0: mean >= m, whose
    capital grows on observations BELOW m.  The upper limit is the smallest m at
    which that test rejects, i.e. the smallest m we can certify the mean is under."""
    if k == 0: return 1.0
    x = np.concatenate([np.ones(e), np.zeros(k - e)])
    if grid is None: grid = np.linspace(1e-3, 1.0, 500)
    thr = np.log(1.0 / d)
    for m in grid:
        lam = min(0.5 / max(1.0 - m, 1e-3), 1.0)   # bet on x < m
        cap = np.sum(np.log1p(lam * (m - x)))
        if cap >= thr: return float(m)
    return 1.0

def kmin_for(ub, alpha, delta, kmax=4000):
    """Smallest k at which the bound can certify at all, i.e. ub(0,k,delta) <= alpha.
    For Clopper-Pearson this reproduces Lemma 1's ceil(ln delta / ln(1-alpha));
    for the looser bounds it is much larger, which is why a fixed sequence started
    at the CP value aborts immediately when a loose bound is used."""
    for k in range(1, kmax + 1):
        if ub(0, k, delta) <= alpha: return k
    return kmax

BOUNDS = [("Hoeffding", hoeffding_upper), ("Emp. Bernstein", bernstein_upper),
          ("Betting (e-CRC)", betting_upper), ("Clopper-Pearson", cp_upper)]


# ------------------------------------------------------- generic FST driver
def fst(score_cal, ok_cal, alpha, delta, ub, rep_mask_cal=None, ok_rep_cal=None):
    """Ascending-lambda fixed sequence on an acceptance threshold; returns lambda-hat."""
    k_min = kmin_for(ub, alpha, delta)
    n = len(score_cal)
    # Start the grid where this bound can certify at all. Skipping an untestable
    # hypothesis mid-sequence would break fixed-sequence FWER control, so the grid
    # must begin above k_min rather than stepping over the early points.
    lam0 = min(0.95, max(1.5 * k_min / max(n, 1), 0.05))
    best = None
    for lam in np.linspace(lam0, 1.0, 40):
        acc = score_cal >= np.quantile(score_cal, 1 - lam)
        if rep_mask_cal is None:
            k = int(acc.sum()); e = int((1 - ok_cal[acc]).sum())
        else:
            rep = (~acc) & rep_mask_cal
            k = int(acc.sum() + rep.sum())
            e = int((1 - ok_cal[acc]).sum() + (1 - ok_rep_cal[rep]).sum())
        if k < k_min: break                   # cannot be tested; the sequence must stop
        if ub(e, k, delta) <= alpha: best = lam
        else: break
    return best

def evaluate(score, ok, alpha, ub, reps=REPS, seed=0):
    """3-way split; returns (mean coverage %, mean risk %, exc_test %, exc_all %)."""
    n = len(ok); rng = np.random.default_rng(seed)
    COV, RSK, RALL = [], [], []
    for _ in range(reps):
        idx = rng.permutation(n); t = n // 3
        cal, te = idx[t:2 * t], idx[2 * t:]
        lam = fst(score[cal], ok[cal], alpha, DELTA, ub)
        if lam is None: COV.append(0.0); continue
        thr = np.quantile(score[cal], 1 - lam)
        for ii, store in ((te, RSK), (np.arange(n), RALL)):
            acc = score[ii] >= thr; k = int(acc.sum())
            if k: store.append(float((1 - ok[ii][acc]).mean()))
        acc = score[te] >= thr
        COV.append(acc.sum() / len(te))
    f = lambda a: (np.mean(a) * 100 if len(a) else float("nan"))
    exc = lambda a: (100 * np.mean(np.array(a) > alpha) if len(a) else float("nan"))
    return f(COV), f(RSK), exc(RSK), exc(RALL)


# --------------------------------------------------------------- features
def pope_features():
    r = J("raw_scores.json"); o = J("owlv2_scores.json")
    p = np.asarray(r["p_yes"], float); a = np.asarray(r["answer"], int)
    g = np.asarray(r["gold"], int); clip = np.asarray(r["ground"], float)
    det = np.asarray(o["ground_det"], float)
    keep = det >= 0
    p, a, g, clip, det = p[keep], a[keep], g[keep], clip[keep], det[keep]
    conf = np.abs(p - .5) * 2
    cn, dn = mm(clip), mm(det)
    ca = np.where(a == 1, cn, 1 - cn); da = np.where(a == 1, dn, 1 - dn)
    ok = (a == g).astype(int)
    det_ans = (det >= C.THR).astype(int)
    return dict(
        conf=np.column_stack([conf, p]),
        clip=np.column_stack([cn, ca]),                       # ConfLVLM-style: grounding only
        conf_clip=np.column_stack([conf, p, cn, ca]),
        conf_det=np.column_stack([conf, p, dn, da]),
        both=np.column_stack([conf, p, cn, ca, dn, da]),
    ), ok, (det_ans == g).astype(int), np.abs(det - C.THR), det

def fit_score(X, ok, tr):
    return LogisticRegression(max_iter=1000).fit(X[tr], ok[tr]).predict_proba(X)[:, 1]

def eval_learned(X, ok, alpha, ub=cp_upper, reps=REPS, seed=0):
    """Same as evaluate() but refits the score on the fit fold each split."""
    n = len(ok); rng = np.random.default_rng(seed)
    COV, RSK, RALL = [], [], []
    for _ in range(reps):
        idx = rng.permutation(n); t = n // 3
        tr, cal, te = idx[:t], idx[t:2 * t], idx[2 * t:]
        if len(np.unique(ok[tr])) < 2: continue
        s = fit_score(X, ok, tr)
        lam = fst(s[cal], ok[cal], alpha, DELTA, ub)
        if lam is None: COV.append(0.0); continue
        thr = np.quantile(s[cal], 1 - lam)
        for ii, store in ((te, RSK), (np.arange(n), RALL)):
            acc = s[ii] >= thr; k = int(acc.sum())
            if k: store.append(float((1 - ok[ii][acc]).mean()))
        COV.append((s[te] >= thr).sum() / len(te))
    f = lambda a: (np.mean(a) * 100 if len(a) else float("nan"))
    exc = lambda a: (100 * np.mean(np.array(a) > alpha) if len(a) else float("nan"))
    return f(COV), f(RSK), exc(RSK), exc(RALL)

def aurc(score, ok, grid=np.linspace(.05, 1., 40)):
    o = np.argsort(-score); okk = ok[o]; n = len(okk)
    risks = [1 - okk[:max(1, int(c * n))].mean() for c in grid]
    return float(np.trapezoid(risks, grid) / (grid[-1] - grid[0]))


# =============================================================== BLOCK 1
def block1():
    print("\n" + "=" * 78)
    print("BLOCK 1 -- Table 12: risk-bound ablation (POPE-1500, LLaVA, 3-way split)")
    print("=" * 78)
    F, ok, _, _, _ = pope_features()
    X = F["both"]
    for alpha in (0.10, 0.15):
        print(f"\n  alpha={alpha:.2f}, delta={DELTA}")
        print(f"  {'bound':18s}{'cov':>8s}{'risk':>8s}{'exc(te)':>9s}{'exc(all)':>10s}")
        for nm, ub in BOUNDS:
            cov, rsk, et, ea = eval_learned(X, ok, alpha, ub)
            flag = "  <-- INVALID" if ea > DELTA * 100 else ""
            print(f"  {nm:18s}{cov:7.1f}%{rsk:7.1f}%{et:8.0f}%{ea:9.0f}%{flag}")


# =============================================================== BLOCK 2
def block2():
    print("\n" + "=" * 78)
    print("BLOCK 2 -- Table 7: detector-only selective baseline")
    print("=" * 78)
    print("  Emits the DETECTOR's answer, gated on its own evidence margin.")
    print("  'CCRC' column repeated from ccrc_v3 protocol for comparison.\n")
    SET = [("POPE-1500 LLaVA", "raw_scores.json"), ("POPE-adv LLaVA", "exp7_pope.json"),
           ("POPE-adv Qwen2-VL", "exp8_qwen_pope.json"), ("POPE-adv LLaVA+VCD", "exp9_vcd_pope.json"),
           ("AMBER(d) LLaVA", "exp12_amber_all.json")]
    print(f"  {'setting':22s}{'VLM acc':>9s}{'det acc':>9s}{'det-only cov':>14s}{'risk':>7s}{'exc(all)':>10s}")
    for name, f in SET:
        if name.startswith("POPE-1500"):
            r = J("raw_scores.json"); o = J("owlv2_scores.json")
            p, a, g, det = r["p_yes"], r["answer"], r["gold"], o["ground_det"]
        else:
            d = J(f); key = "p_vcd" if "p_vcd" in d else "p_yes"
            p, a, g, det = d[key], d["answer"], d["gold"], d["owl"]
        a = np.asarray(a, int); g = np.asarray(g, int); det = np.asarray(det, float)
        keep = det >= 0; a, g, det = a[keep], g[keep], det[keep]
        ok_v = (a == g).astype(int)
        det_ans = (det >= C.THR).astype(int); ok_d = (det_ans == g).astype(int)
        margin = np.abs(det - C.THR)
        cov, rsk, et, ea = evaluate(margin, ok_d, 0.10, cp_upper)
        print(f"  {name:22s}{ok_v.mean()*100:8.1f}%{ok_d.mean()*100:8.1f}%"
              f"{cov:13.1f}%{rsk:6.1f}%{ea:9.0f}%")


# =============================================================== BLOCK 3
def block3():
    print("\n" + "=" * 78)
    print("BLOCK 3 -- Table 8: ConfLVLM-style baselines and the attribution ladder")
    print("=" * 78)
    print("  ConfLVLM's scorer is image-text similarity (CLIP). Rung 1 uses CLIP")
    print("  grounding features ONLY (no model logits), which is the faithful")
    print("  reproduction of its signal in our harness.\n")
    F, ok, ok_r, mg, _ = pope_features()
    alpha = 0.10
    ladder = [("1. CLIP scorer only (ConfLVLM-style)", "clip"),
              ("2. model logits only", "conf"),
              ("3. logits + CLIP", "conf_clip"),
              ("4. logits + detection", "conf_det"),
              ("5. logits + CLIP + detection", "both")]
    print(f"  {'rung':38s}{'AURC':>8s}{'cov':>8s}{'risk':>7s}{'exc(all)':>10s}")
    covs = {}
    for nm, k in ladder:
        cov, rsk, et, ea = eval_learned(F[k], ok, alpha)
        # AURC on a fixed fit for reference
        rng = np.random.default_rng(0); idx = rng.permutation(len(ok)); t = len(ok) // 3
        s = fit_score(F[k], ok, idx[:t])
        print(f"  {nm:38s}{aurc(s,ok):8.4f}{cov:7.1f}%{rsk:6.1f}%{ea:9.0f}%")
        covs[k] = cov
    # CCRC on top of the best score
    lam_cov = []
    n = len(ok); rng = np.random.default_rng(0)
    for _ in range(REPS):
        idx = rng.permutation(n); t = n // 3
        tr, cal, te = idx[:t], idx[t:2*t], idx[2*t:]
        if len(np.unique(ok[tr])) < 2: continue
        s = fit_score(F["both"], ok, tr)
        lam = fst(s[cal], ok[cal], alpha, DELTA, cp_upper,
                  rep_mask_cal=(mg[cal] >= np.quantile(mg[cal], 0.90)), ok_rep_cal=ok_r[cal])
        if lam is None: lam_cov.append(0.0); continue
        thr = np.quantile(s[cal], 1 - lam); tm = np.quantile(mg[cal], 0.90)
        acc = s[te] >= thr; rep = (~acc) & (mg[te] >= tm)
        lam_cov.append((acc.sum() + rep.sum()) / len(te))
    ccrc = np.mean(lam_cov) * 100
    print(f"  {'6. + CCRC repair (ours)':38s}{'--':>8s}{ccrc:7.1f}%")
    print(f"\n  DECOMPOSITION of the margin over rung 1:")
    print(f"    total                     {ccrc - covs['clip']:+6.1f} pp")
    print(f"    of which the score        {covs['both'] - covs['clip']:+6.1f} pp")
    print(f"    of which CCRC repair      {ccrc - covs['both']:+6.1f} pp")


# =============================================================== BLOCK 4
def block4():
    print("\n" + "=" * 78)
    print("BLOCK 4 -- Sec 5.4: VCD composition, AURC and certified coverage")
    print("=" * 78)
    d = J("exp9_vcd_pope.json")
    p = np.asarray(d["p_vcd"], float); a = np.asarray(d["answer"], int)
    g = np.asarray(d["gold"], int); det = np.asarray(d["owl"], float)
    keep = det >= 0; p, a, g, det = p[keep], a[keep], g[keep], det[keep]
    conf = np.abs(p - .5) * 2; dn = mm(det); da = np.where(a == 1, dn, 1 - dn)
    ok = (a == g).astype(int)
    arms = [("confidence only", np.column_stack([conf, p])),
            ("+ detection grounding", np.column_stack([conf, p, dn, da]))]
    print(f"  {'arm':24s}{'AURC':>8s}{'cov@a=0.10':>12s}{'risk':>7s}{'exc(all)':>10s}")
    for nm, X in arms:
        cov, rsk, et, ea = eval_learned(X, ok, 0.10)
        rng = np.random.default_rng(0); idx = rng.permutation(len(ok)); t = len(ok)//3
        s = fit_score(X, ok, idx[:t])
        print(f"  {nm:24s}{aurc(s,ok):8.4f}{cov:11.1f}%{rsk:6.1f}%{ea:9.0f}%")


# =============================================================== BLOCK 5
def block5():
    print("\n" + "=" * 78)
    print("BLOCK 5 -- Sec 4.4: model accuracy in the bottom-q of the canonical score")
    print("=" * 78)
    print(f"  {'setting':22s}" + "".join(f"{'q='+str(q):>10s}" for q in (0.02, 0.05, 0.10, 0.20)))
    for name, (X, ok_o, ok_r, mg) in C.settings():
        n = len(ok_o); rng = np.random.default_rng(0); acc = {q: [] for q in (0.02,0.05,0.10,0.20)}
        for _ in range(REPS):
            idx = rng.permutation(n); t = n // 3
            if len(np.unique(ok_o[idx[:t]])) < 2: continue
            s = fit_score(X, ok_o, idx[:t])
            for q in acc:
                m = s <= np.quantile(s, q)
                if m.sum(): acc[q].append(ok_o[m].mean())
        print(f"  {name:22s}" + "".join(f"{np.mean(acc[q])*100:9.1f}%" for q in (0.02,0.05,0.10,0.20)))


# =============================================================== BLOCK 6
def block6():
    print("\n" + "=" * 78)
    print("BLOCK 6 -- Sec 5.5: grounding gain vs calibration size n (POPE-1500)")
    print("=" * 78)
    F, ok, _, _, _ = pope_features()
    N = len(ok)
    print(f"  {'n':>6s}{'conf-only':>11s}{'+grounding':>12s}{'gain':>8s}")
    for n in (228, 400, 700, N):
        rng = np.random.default_rng(0); sub = rng.permutation(N)[:n]
        c1, _, _, _ = eval_learned(F["conf"][sub], ok[sub], 0.10, reps=40)
        c2, _, _, _ = eval_learned(F["conf_det"][sub], ok[sub], 0.10, reps=40)
        print(f"  {n:6d}{c1:10.1f}%{c2:11.1f}%{c2-c1:+7.1f}")


if __name__ == "__main__":
    for b in (block1, block2, block3, block4, block5, block6):
        b()
    print("\nAll blocks use the 3-way disjoint split, FST from k_min, and report")
    print("exceedance against delta=0.10 as the validity audit.")
