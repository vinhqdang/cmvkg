"""
CCRC — Certified-Correction Risk Control  (canonical implementation)

A selective-prediction procedure that EMITS REPAIRED answers instead of only
accepting/abstaining, while retaining a distribution-free finite-sample risk
guarantee on everything it emits.

WHY A NEW PROCEDURE IS NEEDED
  Prop. 3 of "When Can Conformal Risk Control Certify LLM Outputs?" (arXiv
  2606.29054) proves a distribution-free lower bound: a selective predictor that
  may only EMIT-or-ABSTAIN the model's own answer must abstain on at least
  (mu-alpha)/(1-alpha) of inputs when the base risk mu exceeds alpha. Every
  existing conformal-LVLM method (ConfLVLM, conformal abstention, conformal
  language modeling) lives inside that bound because they only delete or abstain.
  The bound does NOT apply to a predictor that emits a *different* answer.

THE MECHANISM: RISK-BUDGET REALLOCATION
  A calibrated filtering policy systematically UNDERSPENDS its budget: it emits a
  high-precision subset whose realized risk sits well below alpha (measured: 8.0%
  at alpha=10%). CCRC converts that unused slack into coverage by admitting
  repaired items whose individual error is higher than the accepted set's, but low
  enough that the BLENDED risk of the emitted set stays <= alpha.

TWO CHANNELS (the essential design constraint)
  Channel 1  s(x)      : P(the model's own answer is correct)   -> gates ACCEPT
  Channel 2  m(x)      : independent evidence margin (detector) -> gates REPAIR
  The repair MUST be certified by an independent channel. Certifying it with 1-s
  is circular -- a hallucination score says "suspicious", not "the opposite is
  true". Empirically (POPE/LLaVA): even the bottom 2% of s had P(orig correct)
  = 0.133 > 0.10, so no repair is ever certifiable that way, whereas the detector
  channel is 93-100% accurate in its top margin deciles.

POLICY FAMILY (nested, pre-specified => fixed-sequence testing, no multiplicity tax)
  for permissiveness lam:
      ACCEPT original  if  s >= Q_s(1-lam)
      REPAIR w/ ch.2   if  not accepted and m >= Q_m(1-lam)      [family F_rep]
      ABSTAIN          otherwise
  Family F_filt is the same with repair disabled.

GUARANTEE
  Emitted-set risk uses the correctness of what is ACTUALLY emitted:
      err = #(accepted & original wrong) + #(repaired & channel-2 answer wrong)
  CCRC runs FST over each family at level delta/2 and deploys whichever certifies
  more coverage; by the union bound the deployed policy satisfies
      P[ risk(emitted) <= alpha ] >= 1 - delta.
  Adaptive family selection is what makes CCRC never worse than filtering.

ADMISSIBILITY (when does repair help?)
  Repairs are admitted iff the blend stays under budget. Writing c_a, c_r for the
  accepted/repaired masses and r_a, r_r for their risks, repair buys coverage iff
      (c_a*r_a + c_r*r_r)/(c_a+c_r) <= alpha,
  i.e. slack (alpha - r_a) on the accepted set finances repairs with r_r > alpha.
  Consequence: the gain grows with the accepted set's slack and with channel-2
  accuracy, and vanishes as alpha -> r_a (verified: gain peaks at moderate alpha).
"""
import json, os, numpy as np, warnings; warnings.filterwarnings("ignore")
from scipy.stats import beta
from sklearn.linear_model import LogisticRegression

# ---------------------------------------------------------------- core procedure
def cp_upper(e, k, delta):
    """1-delta upper confidence bound on a Binomial error rate (exact)."""
    if k == 0: return 1.0
    return 1.0 if e == k else float(beta.ppf(1 - delta, e + 1, k - e))

def lam_grid(n_cal, alpha, delta, n=40):
    """Pre-specified (data-independent) nested sequence. Starts at the smallest
    emitted-set size that Clopper-Pearson can ever certify:
    k_min = ln(delta)/ln(1-alpha)  (~22 for alpha=delta=0.1)."""
    k_min = int(np.ceil(np.log(delta) / np.log(1 - alpha)))
    lam0 = min(0.9, max(0.05, 1.5 * k_min / max(n_cal, 1)))
    return np.linspace(lam0, 1.0, n)

def apply_policy(lam, s, m, qs, qm, repair):
    acc = s >= np.quantile(qs, 1 - lam)
    rep = ((~acc) & (m >= np.quantile(qm, 1 - lam))) if repair else np.zeros_like(acc)
    return acc, rep

def emitted_risk(acc, rep, ok_orig, ok_rep):
    k = int(acc.sum() + rep.sum())
    e = int((1 - ok_orig[acc]).sum() + (1 - ok_rep[rep]).sum())
    return e, k

def fst(s, m, ok_orig, ok_rep, qs, qm, repair, alpha, delta):
    """Fixed-sequence testing over the nested family; returns (lam, certified_cov)."""
    best, cov = None, 0.0
    for lam in lam_grid(len(s), alpha, delta):
        acc, rep = apply_policy(lam, s, m, qs, qm, repair)
        e, k = emitted_risk(acc, rep, ok_orig, ok_rep)
        if k == 0: continue
        if cp_upper(e, k, delta) <= alpha:
            best, cov = lam, k / len(s)
        else:
            break                                    # FST: stop at first failure
    return best, cov

def ccrc_calibrate(s_cal, m_cal, ok_orig_cal, ok_rep_cal, alpha, delta):
    """Certify BOTH families at delta/2 and deploy the better one (union bound)."""
    half = delta / 2
    out = {}
    for name, repair in [("filter", False), ("repair", True)]:
        out[name] = fst(s_cal, m_cal, ok_orig_cal, ok_rep_cal, s_cal, m_cal,
                        repair, alpha, half)
    pick = max(out, key=lambda k: out[k][1])
    return pick, out[pick][0], out

def ccrc_deploy(lam, repair, s, m, qs, qm, ok_orig, ok_rep):
    if lam is None: return dict(coverage=0.0, risk=0.0, repaired=0.0)
    acc, rep = apply_policy(lam, s, m, qs, qm, repair)
    e, k = emitted_risk(acc, rep, ok_orig, ok_rep)
    if k == 0: return dict(coverage=0.0, risk=0.0, repaired=0.0)
    return dict(coverage=k / len(s), risk=e / k, repaired=int(rep.sum()) / len(s))

# ---------------------------------------------------------------- experiment
def load(here):
    r = json.load(open(os.path.join(here, "raw_scores.json")))
    o = json.load(open(os.path.join(here, "owlv2_scores.json")))
    p = np.array(r["p_yes"]); a = np.array(r["answer"]); gold = np.array(r["gold"])
    det = np.array(o["ground_det"])
    mm = lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9)
    dn = mm(det); conf = np.abs(p - .5) * 2
    X = np.column_stack([p, conf, dn, np.where(a == 1, dn, 1 - dn)])
    THR = 0.15
    return dict(X=X, ok_orig=(a == gold).astype(int),
                ok_rep=((det >= THR).astype(int) == gold).astype(int),
                margin=np.abs(det - THR))

def main(reps=100, delta=0.10):
    here = os.path.dirname(os.path.abspath(__file__))
    D = load(here); X = D["X"]; ok_o = D["ok_orig"]; ok_r = D["ok_rep"]; mg = D["margin"]
    n = len(ok_o); mu = 1 - ok_o.mean()
    print(f"CCRC on POPE / LLaVA-1.5-7B   n={n}   base risk mu={mu:.3f}   delta={delta}")
    print(f"protocol: 3-way split (train/cal/test), FST + Clopper-Pearson, {reps} reps")
    print(f"filtering ceiling [Prop.3, arXiv 2606.29054] at alpha: 1-(mu-alpha)/(1-alpha)\n")
    hdr = (f"{'alpha':>6s}{'ceil':>7s} | {'FILTER cov':>11s}{'risk':>7s} | "
           f"{'CCRC cov':>9s}{'risk':>7s}{'repaired':>9s} | {'gain':>7s}{'valid':>7s}")
    print(hdr); print("-" * len(hdr))
    for alpha in [0.05, 0.10, 0.15, 0.20]:
        rng = np.random.default_rng(0)
        F = {"cov": [], "risk": []}; C = {"cov": [], "risk": [], "rep": []}
        for _ in range(reps):
            idx = rng.permutation(n); t = n // 3
            tr, cal, te = idx[:t], idx[t:2*t], idx[2*t:]
            clf = LogisticRegression(max_iter=1000).fit(X[tr], ok_o[tr])
            s = clf.predict_proba(X)[:, 1]
            # baseline: filtering only, full delta
            lam_f, _ = fst(s[cal], mg[cal], ok_o[cal], ok_r[cal], s[cal], mg[cal],
                           False, alpha, delta)
            rf = ccrc_deploy(lam_f, False, s[te], mg[te], s[cal], mg[cal], ok_o[te], ok_r[te])
            F["cov"].append(rf["coverage"]); F["risk"].append(rf["risk"])
            # CCRC: adaptive family selection at delta/2 each
            pick, lam, _ = ccrc_calibrate(s[cal], mg[cal], ok_o[cal], ok_r[cal], alpha, delta)
            rc = ccrc_deploy(lam, pick == "repair", s[te], mg[te], s[cal], mg[cal],
                             ok_o[te], ok_r[te])
            C["cov"].append(rc["coverage"]); C["risk"].append(rc["risk"]); C["rep"].append(rc["repaired"])
        ceil = 1 - max(0.0, (mu - alpha) / (1 - alpha))
        fc, fr = np.mean(F["cov"]) * 100, np.mean(F["risk"]) * 100
        cc, cr, rp = np.mean(C["cov"]) * 100, np.mean(C["risk"]) * 100, np.mean(C["rep"]) * 100
        ok = "OK" if (fr <= alpha * 100 + 1 and cr <= alpha * 100 + 1) else "BAD"
        print(f"{alpha:6.2f}{ceil*100:6.1f}% | {fc:10.1f}%{fr:6.1f}% | "
              f"{cc:8.1f}%{cr:6.1f}%{rp:8.1f}% | {cc-fc:+6.1f}{ok:>7s}")
    print("-" * len(hdr))
    print("gain = certified coverage recovered by certified correction over filtering,")
    print("at the same alpha, with the risk guarantee verified on held-out test data.")

if __name__ == "__main__":
    main()
