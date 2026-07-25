"""
CCRC replication across backbones/decoders (CPU-only, reuses extracted signals).

Runs the canonical CCRC procedure (ccrc.py) on every setting where both channels
are available: channel 1 = VLM correctness score, channel 2 = OWLv2 detector
answering the question itself.

Settings: POPE-1500 (LLaVA), POPE-adv (LLaVA), POPE-adv (Qwen2-VL), POPE-adv (VCD).
Reports certified coverage for filtering vs CCRC at each alpha, with realized test
risk so validity is auditable.
"""
import json, os, numpy as np, warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from ccrc import fst, ccrc_calibrate, ccrc_deploy

_here = os.path.dirname(os.path.abspath(__file__))
J = lambda f: json.load(open(os.path.join(_here, f)))
mm = lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9)
DELTA = 0.10
THR = 0.15                      # detector decision threshold (channel 2)

def build(p_yes, answer, gold, owl):
    """Return (X, ok_orig, ok_rep, margin) for CCRC."""
    p = np.asarray(p_yes, float); a = np.asarray(answer, int)
    g = np.asarray(gold, int);    o = np.asarray(owl, float)
    keep = o >= 0                                   # need channel 2 to exist
    p, a, g, o = p[keep], a[keep], g[keep], o[keep]
    conf = np.abs(p - .5) * 2; dn = mm(o)
    X = np.column_stack([p, conf, dn, np.where(a == 1, dn, 1 - dn)])
    ok_orig = (a == g).astype(int)
    ok_rep = ((o >= THR).astype(int) == g).astype(int)
    return X, ok_orig, ok_rep, np.abs(o - THR)

def settings():
    S = []
    r = J("raw_scores.json"); o = J("owlv2_scores.json")
    S.append(("POPE-1500  LLaVA-1.5", build(r["p_yes"], r["answer"], r["gold"], o["ground_det"])))
    d = J("exp7_pope.json")
    S.append(("POPE-adv   LLaVA-1.5", build(d["p_yes"], d["answer"], d["gold"], d["owl"])))
    d = J("exp8_qwen_pope.json")
    S.append(("POPE-adv   Qwen2-VL-2B", build(d["p_yes"], d["answer"], d["gold"], d["owl"])))
    d = J("exp9_vcd_pope.json")
    S.append(("POPE-adv   LLaVA+VCD", build(d["p_vcd"], d["answer"], d["gold"], d["owl"])))
    for f, lab in [("exp12_amber_all.json", "AMBER(d)   LLaVA-1.5")]:
        if os.path.exists(os.path.join(_here, f)):
            d = J(f); S.append((lab, build(d["p_yes"], d["answer"], d["gold"], d["owl"])))
    return S

def run_setting(X, ok_o, ok_r, mg, alpha, reps=60):
    n = len(ok_o); rng = np.random.default_rng(0)
    F = {"c": [], "r": []}; C = {"c": [], "r": [], "p": []}
    for _ in range(reps):
        idx = rng.permutation(n); t = n // 3
        tr, cal, te = idx[:t], idx[t:2*t], idx[2*t:]
        if len(np.unique(ok_o[tr])) < 2: continue
        clf = LogisticRegression(max_iter=1000).fit(X[tr], ok_o[tr])
        s = clf.predict_proba(X)[:, 1]
        lam_f, _ = fst(s[cal], mg[cal], ok_o[cal], ok_r[cal], s[cal], mg[cal],
                       False, alpha, DELTA)
        rf = ccrc_deploy(lam_f, False, s[te], mg[te], s[cal], mg[cal], ok_o[te], ok_r[te])
        F["c"].append(rf["coverage"]); F["r"].append(rf["risk"])
        pick, lam, _ = ccrc_calibrate(s[cal], mg[cal], ok_o[cal], ok_r[cal], alpha, DELTA)
        rc = ccrc_deploy(lam, pick == "repair", s[te], mg[te], s[cal], mg[cal], ok_o[te], ok_r[te])
        C["c"].append(rc["coverage"]); C["r"].append(rc["risk"]); C["p"].append(rc["repaired"])
    if not F["c"]: return None
    return (np.mean(F["c"])*100, np.mean(F["r"])*100,
            np.mean(C["c"])*100, np.mean(C["r"])*100, np.mean(C["p"])*100)

print(f"CCRC replication (delta={DELTA}, 3-way split, FST+Clopper-Pearson, 60 reps)\n")
for alpha in [0.10, 0.15]:
    print(f"===== alpha = {alpha:.2f} =====")
    print(f"{'setting':24s}{'n':>6s}{'mu':>7s} | {'FILTER':>8s}{'risk':>7s} | "
          f"{'CCRC':>8s}{'risk':>7s}{'repair':>8s} | {'gain':>7s}{'valid':>7s}")
    print("-" * 92)
    for name, (X, ok_o, ok_r, mg) in settings():
        out = run_setting(X, ok_o, ok_r, mg, alpha)
        if out is None:
            print(f"{name:24s}{len(ok_o):6d}   -- too small --"); continue
        fc, fr, cc, cr, rp = out
        mu = (1 - ok_o.mean()) * 100
        ok = "OK" if (fr <= alpha*100+1 and cr <= alpha*100+1) else "BAD"
        print(f"{name:24s}{len(ok_o):6d}{mu:6.1f}% | {fc:7.1f}%{fr:6.1f}% | "
              f"{cc:7.1f}%{cr:6.1f}%{rp:7.1f}% | {cc-fc:+6.1f}{ok:>7s}")
    print()
