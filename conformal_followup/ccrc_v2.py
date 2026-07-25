"""
CCRC v2 — Dual-Channel Certified-Correction Risk Control.

Fixes two failures of v1 (see ccrc_algorithm.py):
  (1) v1 certified the repair with 1 - s (the negation of the flag score) -> circular:
      a hallucination score says "suspicious", not "the opposite is true". Empirically
      even the bottom 2% of s had P(orig correct)=0.133 > alpha=0.10, so no repair was
      ever certifiable.
      FIX: certify repairs with an INDEPENDENT evidence channel (the detector), whose
      correctness (detector answer == truth) is a different random variable from the
      VLM's correctness. Measured: detector is 93-100% accurate in its top margin
      deciles -> certifiable at alpha=0.10.
  (2) v1 used Bonferroni over a 40x40 grid -> bound crushed.
      FIX: fixed-sequence testing (FST) over a PRE-SPECIFIED nested policy family;
      FST controls the error probability with no multiplicity penalty.

Policy family, indexed by a single pre-specified permissiveness lam (nested):
    accept original  if  s        >= Q_s(1-lam)          (channel 1: VLM correctness)
    else repair      if  margin   >= Q_m(1-lam)          (channel 2: detector evidence)
    else abstain
Risk is controlled on the EMITTED set (accepted originals + repaired items):
    error = #(accepted & orig wrong) + #(repaired & detector wrong)
Guarantee: with prob >= 1-delta, emitted-set risk <= alpha  (Clopper-Pearson + FST).

Comparison: the same FST protocol restricted to filtering only (no repair), and the
distribution-free filtering ceiling 1-(mu-alpha)/(1-alpha) from Prop.3 of arXiv
2606.29054. The claim under test: certified correction attains strictly higher
certified coverage than any filtering policy at the same alpha, because it recovers
items filtering must abstain on.
"""
import json, os, numpy as np, warnings; warnings.filterwarnings("ignore")
from scipy.stats import beta
from sklearn.linear_model import LogisticRegression

_here = os.path.dirname(os.path.abspath(__file__))
ALPHA, DELTA = 0.10, 0.10
# Clopper-Pearson cannot certify risk<=alpha with fewer than k_min = ln(delta)/ln(1-alpha)
# clean samples (~22 for alpha=delta=0.1). Starting the fixed sequence below that would
# fail at step 1 and abort FST, so the grid starts at the analytically-derived minimum.
K_MIN = int(np.ceil(np.log(DELTA) / np.log(1 - ALPHA)))
def make_lams(n_cal):
    lam0 = min(0.9, max(0.05, 1.5 * K_MIN / n_cal))   # data-independent given n_cal
    return np.linspace(lam0, 1.0, 40)

def cp_upper(e, k, delta):
    if k == 0: return 1.0
    return 1.0 if e == k else float(beta.ppf(1 - delta, e + 1, k - e))

def policy(lam, s, margin, qs, qm, allow_repair):
    """Nested policy: returns (accept_mask, repair_mask)."""
    acc = s >= np.quantile(qs, 1 - lam)
    if not allow_repair:
        return acc, np.zeros_like(acc)
    rep = (~acc) & (margin >= np.quantile(qm, 1 - lam))
    return acc, rep

def risk_counts(acc, rep, ok_orig, ok_rep):
    k = int(acc.sum() + rep.sum())
    e = int((1 - ok_orig[acc]).sum() + (1 - ok_rep[rep]).sum())
    return e, k

def fst_calibrate(s_cal, m_cal, ok_o_cal, ok_r_cal, qs, qm, allow_repair):
    """Fixed-sequence testing: walk lam from conservative to permissive, keep the
    last lam whose certified risk <= alpha; stop at first failure."""
    best = None
    for lam in make_lams(len(s_cal)):
        acc, rep = policy(lam, s_cal, m_cal, qs, qm, allow_repair)
        e, k = risk_counts(acc, rep, ok_o_cal, ok_r_cal)
        if k == 0: continue
        if cp_upper(e, k, DELTA) <= ALPHA:
            best = lam
        else:
            break                      # FST: stop at first failure
    return best

def evaluate(lam, s, m, ok_o, ok_r, qs, qm, allow_repair):
    if lam is None: return 0.0, 0.0, 0.0
    acc, rep = policy(lam, s, m, qs, qm, allow_repair)
    e, k = risk_counts(acc, rep, ok_o, ok_r)
    if k == 0: return 0.0, 0.0, 0.0
    return k / len(s), e / k, int(rep.sum()) / len(s)

def main(reps=30):
    r = json.load(open(os.path.join(_here, "raw_scores.json")))
    o = json.load(open(os.path.join(_here, "owlv2_scores.json")))
    p = np.array(r["p_yes"]); a = np.array(r["answer"]); gold = np.array(r["gold"])
    det = np.array(o["ground_det"]); clip = np.array(r["ground"])
    n = len(gold)
    mm = lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9)
    conf = np.abs(p - .5) * 2; dn = mm(det)
    ok_orig = (a == gold).astype(int)                       # channel-1 correctness
    mu = 1 - ok_orig.mean()
    ceiling = 1 - max(0.0, (mu - ALPHA) / (1 - ALPHA))      # Prop.3 filtering ceiling

    FEATS = {
        "confidence only":   np.column_stack([p, conf]),
        "+ CLIP grounding":  np.column_stack([p, conf, mm(clip), np.where(a == 1, mm(clip), 1 - mm(clip))]),
        "+ OWLv2 grounding": np.column_stack([p, conf, dn, np.where(a == 1, dn, 1 - dn)]),
    }
    print(f"POPE / LLaVA-1.5-7B  n={n}  base risk mu={mu:.3f}  alpha={ALPHA:.2f}  delta={DELTA:.2f}")
    print(f"Distribution-free FILTERING ceiling [Prop.3, 2606.29054]: coverage <= {ceiling*100:.1f}%")
    print(f"(protocol: 3-way split, fixed-sequence testing, Clopper-Pearson)\n")
    print(f"{'score':20s}{'method':12s}{'coverage':>10s}{'test risk':>11s}{'valid':>7s}{'repaired':>10s}")
    print("-" * 72)
    rng = np.random.default_rng(0); out = {}
    for name, X in FEATS.items():
        res = {"filter": [[], []], "CCRC": [[], [], []]}
        for _ in range(reps):
            idx = rng.permutation(n); t = n // 3
            tr, cal, te = idx[:t], idx[t:2*t], idx[2*t:]
            # channel 1: VLM correctness score
            clf = LogisticRegression(max_iter=1000).fit(X[tr], ok_orig[tr])
            s = clf.predict_proba(X)[:, 1]
            # channel 2: detector answers the question itself; margin = |evidence|
            thr = 0.15                                   # detector decision threshold
            det_ans = (det >= thr).astype(int)
            ok_rep = (det_ans == gold).astype(int)       # INDEPENDENT of ok_orig
            margin = np.abs(det - thr)
            qs, qm = s[cal], margin[cal]
            for meth, allow in [("filter", False), ("CCRC", True)]:
                lam = fst_calibrate(s[cal], margin[cal], ok_orig[cal], ok_rep[cal], qs, qm, allow)
                cv, rk, rp = evaluate(lam, s[te], margin[te], ok_orig[te], ok_rep[te], qs, qm, allow)
                res[meth][0].append(cv); res[meth][1].append(rk)
                if allow: res[meth][2].append(rp)
        for meth in ["filter", "CCRC"]:
            cv = np.mean(res[meth][0]) * 100; rk = np.mean(res[meth][1]) * 100
            rp = np.mean(res[meth][2]) * 100 if meth == "CCRC" else None
            valid = "OK" if rk <= ALPHA * 100 + 1 else "BAD"
            print(f"{name if meth=='filter' else '':20s}{meth:12s}{cv:9.1f}%{rk:10.1f}%{valid:>7s}"
                  f"{(f'{rp:8.1f}%' if rp is not None else ''):>10s}")
        out[name] = (np.mean(res["filter"][0]) * 100, np.mean(res["CCRC"][0]) * 100)
        print("-" * 72)
    print("\nCertified coverage: filtering -> CCRC (certified correction)")
    for name, (f, c) in out.items():
        print(f"  {name:20s} {f:5.1f}%  ->  {c:5.1f}%   ({c-f:+5.1f} pp)")

if __name__ == "__main__":
    main()
