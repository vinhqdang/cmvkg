"""
BCEA vs CCRC vs COMPOSED — analysis (CPU-only).
Requires exp13_bcea.json from colab_exp13_bcea.py.

Arms, all under the identical FST + Clopper-Pearson protocol and 3-way split:
  1. FILTER      : ch.1 score (confidence+grounding features), filter only      [baseline]
  2. BCEA        : filter on the post-acquisition score s_acq (arXiv 2606.16667)
                   -> rescues by RE-SCORING the model's original answer
  3. CCRC        : ch.1 score + repair via the independent detector channel
                   -> rescues by REPLACING the answer
  4. COMPOSED    : s_acq as ch.1 + repair via ch.2  -> both mechanisms together

The interesting question is 4 vs max(2,3): the two mechanisms are different (re-read vs
replace), so if they are complementary the composition should dominate both.
"""
import json, os, sys, numpy as np, warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
import ccrc_v3 as V

_here = os.path.dirname(os.path.abspath(__file__))
F = os.path.join(_here, "exp13_bcea.json")
if not os.path.exists(F):
    sys.exit("exp13_bcea.json not found — run colab_exp13_bcea.py on a GPU first.")
d = json.load(open(F))
mm, THR, DELTA = V.mm, V.THR, V.DELTA

p = np.array(d["p_yes"]); a = np.array(d["answer"]); g = np.array(d["gold"])
owl = np.array(d["owl"]); s_acq = np.array(d["s_acq"]); l_full = np.array(d["l_full"])
keep = owl >= 0                                     # need ch.2 for repair arms
p, a, g, owl, s_acq, l_full = (v[keep] for v in (p, a, g, owl, s_acq, l_full))
n = len(g); conf = np.abs(p - .5) * 2; dn = mm(owl)
ok_orig = (a == g).astype(int)
ok_rep = ((owl >= THR).astype(int) == g).astype(int)
margin = np.abs(owl - THR)
mu = 1 - ok_orig.mean()

# FAIR DESIGN: hold the BASE score fixed (VLM confidence only) and add ONE mechanism at
# a time, so each arm isolates a mechanism rather than inheriting the other's features.
# (An earlier version gave the filter arm our grounding features while BCEA got only its
#  own acquisition score -- that understates BCEA and is not a valid comparison.)
BASE  = np.column_stack([p, conf])                                     # VLM confidence
X_acq = np.column_stack([p, conf, s_acq, l_full])                       # + BCEA acquisition
X_grd = np.column_stack([p, conf, dn, np.where(a == 1, dn, 1 - dn)])    # + our grounding score
X_all = np.column_stack([p, conf, dn, np.where(a == 1, dn, 1 - dn), s_acq, l_full])

ARMS = [("BASE filter (confidence only)",        BASE,  None),
        ("+ BCEA acquisition (re-score)",        X_acq, None),
        ("+ our grounding score (filter)",       X_grd, None),
        ("+ CCRC repair (ours, full)",           X_grd, 0.10),
        ("COMPOSED (BCEA + grounding + repair)", X_all, 0.10)]

print(f"POPE / LLaVA-1.5-7B   n={n} (grounded)   base risk mu={mu:.3f}   delta={DELTA}")
print(f"BCEA s_acq range {s_acq.min():.2f}..{s_acq.max():.2f}\n")
for alpha in [0.10, 0.15]:
    print(f"===== alpha={alpha:.2f} (60 reps, 3-way split, FST+CP) =====")
    print(f"{'arm':40s}{'coverage':>10s}{'risk':>8s}{'repaired':>10s}{'valid':>7s}")
    print("-" * 75)
    res = {}
    for name, X, q in ARMS:
        rng = np.random.default_rng(0); C, R, P = [], [], []
        for _ in range(60):
            idx = rng.permutation(n); t = n // 3
            tr, cal, te = idx[:t], idx[t:2*t], idx[2*t:]
            if len(np.unique(ok_orig[tr])) < 2: continue
            s = LogisticRegression(max_iter=1000).fit(X[tr], ok_orig[tr]).predict_proba(X)[:, 1]
            lam = V.calibrate(s[cal], margin[cal], ok_orig[cal], ok_rep[cal], alpha, DELTA, q)
            cv, rk, rp = V.deploy(lam, s[te], margin[te], s[cal], margin[cal],
                                  ok_orig[te], ok_rep[te], q)
            C.append(cv); R.append(rk); P.append(rp)
        cov, rsk, rep = np.mean(C) * 100, np.mean(R) * 100, np.mean(P) * 100
        res[name] = cov
        ok = "OK" if rsk <= alpha * 100 + 1 else "BAD"
        print(f"{name:40s}{cov:9.1f}%{rsk:7.1f}%{rep:9.1f}%{ok:>7s}")
    print("-" * 75)
    base = res["BASE filter (confidence only)"]
    b = res["+ BCEA acquisition (re-score)"]; g = res["+ our grounding score (filter)"]
    c = res["+ CCRC repair (ours, full)"]; comp = res["COMPOSED (BCEA + grounding + repair)"]
    print(f"mechanism deltas vs BASE {base:.1f}%:  BCEA acquisition {b-base:+.1f}pp | "
          f"grounding score {g-base:+.1f}pp | grounding+repair {c-base:+.1f}pp")
    print(f"COMPOSED {comp:.1f}% ({comp-base:+.1f}pp)  ->  "
          f"{'complementary: beats every single-mechanism arm' if comp>max(b,g,c) else 'no additive benefit over the best single mechanism'}\n")
