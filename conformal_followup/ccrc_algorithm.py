"""
CCRC — Certified-Correction Risk Control.

NEW ALGORITHM. Motivation: Prop. 3 of "When Can Conformal Risk Control Certify LLM
Outputs?" (arXiv 2606.29054) proves a distribution-free IMPOSSIBILITY for
filtering-based selective prediction: if the base risk is mu > alpha, ANY such
predictor must abstain on at least (mu-alpha)/(1-alpha) of inputs. That bound
assumes the predictor may only EMIT or ABSTAIN the model's original answer.

Key idea: escape the bound by emitting a *modified* answer. A calibrated
correctness score s identifies not only reliably-CORRECT items (high s) but also
reliably-INCORRECT items (low s). Filtering throws the latter away; CCRC REPAIRS
them with the evidence-implied answer and certifies the repaired output.

Policy (two-sided, jointly calibrated):
    s >= tau_hi   -> EMIT original answer          (certified correct)
    s <= tau_lo   -> EMIT repaired answer          (certified correction)
    otherwise     -> ABSTAIN
Risk is controlled on the EMITTED set (accepted originals + repairs) at level
alpha with confidence 1-delta, using Clopper-Pearson with Bonferroni correction
over the 2-D threshold grid (Learn-Then-Test style multiplicity control).

Because the policy is calibrated jointly, calibration is ON-POLICY by
construction for this one-shot (discriminative) setting -- no self-induced shift.

This script tests, on real POPE/LLaVA data:
  (a) does CCRC keep the risk guarantee on held-out test data?
  (b) does it exceed the filtering-only coverage AND the theoretical filtering
      lower bound on abstention?
  (c) is the gain larger with the structured grounding score (better separation
      => more mass in the reliably-wrong region)?
"""
import json, os, numpy as np, warnings; warnings.filterwarnings("ignore")
from scipy.stats import beta
from sklearn.linear_model import LogisticRegression

_here = os.path.dirname(os.path.abspath(__file__))
ALPHA, DELTA = 0.10, 0.10

def cp_upper(e, k, delta):
    """1-delta upper confidence bound on a Binomial error rate."""
    if k == 0: return 1.0
    return 1.0 if e == k else float(beta.ppf(1 - delta, e + 1, k - e))

# ---------------- policies ----------------
def calibrate_filter(s, correct, alpha, delta):
    """Filtering-only: one-sided threshold, maximize coverage s.t. certified risk<=alpha."""
    grid = np.unique(s); d = delta / max(len(grid), 1)      # Bonferroni over grid
    best = (None, -1)
    for t in grid:
        m = s >= t; k = int(m.sum())
        if k == 0: continue
        if cp_upper(int((1 - correct[m]).sum()), k, d) <= alpha and k > best[1]:
            best = (float(t), k)
    return best[0]

def calibrate_ccrc(s, correct, alpha, delta, n_grid=40):
    """CCRC: two-sided (tau_lo, tau_hi). Repaired items are correct iff the original
    was incorrect (binary task), so error among repaired = correct[repaired]."""
    qs = np.quantile(s, np.linspace(0, 1, n_grid))
    lo_grid = np.unique(qs); hi_grid = np.unique(qs)
    d = delta / max(len(lo_grid) * len(hi_grid), 1)          # Bonferroni over 2-D grid
    best = (None, None, -1)
    for th in hi_grid:
        acc = s >= th
        e_acc = int((1 - correct[acc]).sum()); k_acc = int(acc.sum())
        for tl in lo_grid:
            if tl >= th: continue
            rep = s <= tl
            # a repaired item is WRONG exactly when the original answer was RIGHT
            e_rep = int(correct[rep].sum()); k_rep = int(rep.sum())
            k = k_acc + k_rep
            if k == 0: continue
            if cp_upper(e_acc + e_rep, k, d) <= alpha and k > best[2]:
                best = (float(tl), float(th), k)
    return best[0], best[1]

def eval_filter(s, correct, tau):
    if tau is None: return 0.0, 0.0
    m = s >= tau; k = int(m.sum())
    if k == 0: return 0.0, 0.0
    return k / len(s), float((1 - correct[m]).mean())

def eval_ccrc(s, correct, tau_lo, tau_hi):
    if tau_hi is None: return 0.0, 0.0, 0.0
    acc = s >= tau_hi; rep = s <= tau_lo
    k = int(acc.sum() + rep.sum())
    if k == 0: return 0.0, 0.0, 0.0
    err = int((1 - correct[acc]).sum()) + int(correct[rep].sum())
    return k / len(s), err / k, int(rep.sum()) / len(s)

# ---------------- data ----------------
def load_pope():
    r = json.load(open(os.path.join(_here, "raw_scores.json")))
    o = json.load(open(os.path.join(_here, "owlv2_scores.json")))
    p = np.array(r["p_yes"]); a = np.array(r["answer"]); c = np.array(r["correct"])
    det = np.array(o["ground_det"]); clip = np.array(r["ground"])
    mm = lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9)
    conf = np.abs(p - .5) * 2; dn = mm(det); cn = mm(clip)
    feats = {
        "confidence only":      np.column_stack([p, conf]),
        "+ CLIP grounding":     np.column_stack([p, conf, cn, np.where(a == 1, cn, 1 - cn)]),
        "+ OWLv2 grounding":    np.column_stack([p, conf, dn, np.where(a == 1, dn, 1 - dn)]),
    }
    return feats, c

def run(reps=30):
    feats, correct = load_pope(); n = len(correct)
    mu = 1 - correct.mean()
    bound_abstain = max(0.0, (mu - ALPHA) / (1 - ALPHA))
    print(f"POPE / LLaVA-1.5-7B   n={n}   base risk mu={mu:.3f}   alpha={ALPHA:.2f}")
    print(f"Filtering impossibility [Prop.3, 2606.29054]: must abstain >= {bound_abstain*100:.1f}% "
          f"=> max coverage {100*(1-bound_abstain):.1f}%\n")
    print(f"{'score':20s}{'method':10s}{'coverage':>10s}{'test risk':>11s}{'valid':>7s}{'repaired':>10s}")
    print("-" * 70)
    rng = np.random.default_rng(0); summary = {}
    for name, X in feats.items():
        R = {"filter": {"cov": [], "risk": []}, "ccrc": {"cov": [], "risk": [], "rep": []}}
        for _ in range(reps):
            idx = rng.permutation(n); t = n // 3
            tr, cal, te = idx[:t], idx[t:2*t], idx[2*t:]
            clf = LogisticRegression(max_iter=1000).fit(X[tr], correct[tr])
            s = clf.predict_proba(X)[:, 1]
            tf = calibrate_filter(s[cal], correct[cal], ALPHA, DELTA)
            cv, rk = eval_filter(s[te], correct[te], tf)
            R["filter"]["cov"].append(cv); R["filter"]["risk"].append(rk)
            tl, th = calibrate_ccrc(s[cal], correct[cal], ALPHA, DELTA)
            cv2, rk2, rp = eval_ccrc(s[te], correct[te], tl, th)
            R["ccrc"]["cov"].append(cv2); R["ccrc"]["risk"].append(rk2); R["ccrc"]["rep"].append(rp)
        for meth in ["filter", "ccrc"]:
            cov = np.mean(R[meth]["cov"]) * 100; rk = np.mean(R[meth]["risk"]) * 100
            rep = np.mean(R[meth]["rep"]) * 100 if meth == "ccrc" else None
            valid = "OK" if rk <= ALPHA * 100 + 1 else "BAD"
            print(f"{name if meth=='filter' else '':20s}{meth:10s}{cov:9.1f}%{rk:10.1f}%{valid:>7s}"
                  f"{(f'{rep:8.1f}%' if rep is not None else ''):>10s}")
        summary[name] = (np.mean(R["filter"]["cov"]) * 100, np.mean(R["ccrc"]["cov"]) * 100)
        print("-" * 70)
    print("\nCoverage gain from certified correction (CCRC - filtering):")
    for name, (f, c) in summary.items():
        flag = "  <-- EXCEEDS filtering bound" if c > (1 - bound_abstain) * 100 else ""
        print(f"  {name:20s} {f:5.1f}% -> {c:5.1f}%   ({c-f:+5.1f} pp){flag}")

if __name__ == "__main__":
    run()
