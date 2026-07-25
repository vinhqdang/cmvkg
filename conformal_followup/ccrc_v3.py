"""
CCRC v3 — decoupled repair gate.

v2 flaw (found by replication across backbones): the repair gate was tied to the
same permissiveness lam as the acceptance gate. As lam loosened, the repair region
loosened too, admitting low-precision repairs; FST then stopped early. Result:
gains on LLaVA (+4.7/+5.3 pp) but LOSSES on Qwen2-VL (-4.2) and VCD (-2.9), where
repair mass was small (3.1%/6.4%) and the delta/2 selection cost dominated.

v3 fix: gate repairs at a FIXED, pre-specified evidence-quality level q (a constant
percentile of the channel-2 margin), independent of lam:
      ACCEPT   if s >= Q_s(1-lam)
      REPAIR   if not accepted AND m >= Q_m(1-q)        <-- q fixed, NOT lam
      ABSTAIN  otherwise
This keeps repair precision constant while lam sweeps acceptance, and the family
stays nested in lam (as lam grows the accepted set grows and the repair set
shrinks), so a SINGLE fixed-sequence test at full delta suffices -- no union-bound
split, removing the cost that made v2 lose on 2 of 4 settings.

q is a hyperparameter of the procedure, chosen a priori (not tuned per dataset);
we report sensitivity across q in {0.10, 0.25, 0.50}.
"""
import json, os, numpy as np, warnings; warnings.filterwarnings("ignore")
from scipy.stats import beta
from sklearn.linear_model import LogisticRegression

_here = os.path.dirname(os.path.abspath(__file__))
DELTA = 0.10
THR = 0.15
mm = lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9)

def cp_upper(e, k, d):
    if k == 0: return 1.0
    return 1.0 if e == k else float(beta.ppf(1 - d, e + 1, k - e))

def lam_grid(n_cal, alpha, delta, n=40):
    k_min = int(np.ceil(np.log(delta) / np.log(1 - alpha)))
    lam0 = min(0.9, max(0.05, 1.5 * k_min / max(n_cal, 1)))
    return np.linspace(lam0, 1.0, n)

def policy(lam, s, m, s_ref, m_ref, q):
    """q=None -> filtering only. Otherwise repair gate fixed at margin percentile 1-q."""
    acc = s >= np.quantile(s_ref, 1 - lam)
    if q is None: return acc, np.zeros_like(acc)
    rep = (~acc) & (m >= np.quantile(m_ref, 1 - q))
    return acc, rep

def counts(acc, rep, ok_o, ok_r):
    k = int(acc.sum() + rep.sum())
    e = int((1 - ok_o[acc]).sum() + (1 - ok_r[rep]).sum())
    return e, k

def calibrate(s_cal, m_cal, ok_o, ok_r, alpha, delta, q):
    best = None
    for lam in lam_grid(len(s_cal), alpha, delta):
        acc, rep = policy(lam, s_cal, m_cal, s_cal, m_cal, q)
        e, k = counts(acc, rep, ok_o, ok_r)
        if k == 0: continue
        if cp_upper(e, k, delta) <= alpha: best = lam
        else: break
    return best

def deploy(lam, s_te, m_te, s_ref, m_ref, ok_o, ok_r, q):
    if lam is None: return 0.0, 0.0, 0.0
    acc, rep = policy(lam, s_te, m_te, s_ref, m_ref, q)
    e, k = counts(acc, rep, ok_o, ok_r)
    if k == 0: return 0.0, 0.0, 0.0
    return k / len(s_te), e / k, int(rep.sum()) / len(s_te)

# ------------------------------------------------------------------ data
def build(p_yes, answer, gold, owl):
    p = np.asarray(p_yes, float); a = np.asarray(answer, int)
    g = np.asarray(gold, int);    o = np.asarray(owl, float)
    keep = o >= 0
    p, a, g, o = p[keep], a[keep], g[keep], o[keep]
    conf = np.abs(p - .5) * 2; dn = mm(o)
    X = np.column_stack([p, conf, dn, np.where(a == 1, dn, 1 - dn)])
    return X, (a == g).astype(int), ((o >= THR).astype(int) == g).astype(int), np.abs(o - THR)

def settings():
    J = lambda f: json.load(open(os.path.join(_here, f)))
    S = []
    r = J("raw_scores.json"); o = J("owlv2_scores.json")
    S.append(("POPE-1500 LLaVA", build(r["p_yes"], r["answer"], r["gold"], o["ground_det"])))
    d = J("exp7_pope.json");  S.append(("POPE-adv LLaVA", build(d["p_yes"], d["answer"], d["gold"], d["owl"])))
    d = J("exp8_qwen_pope.json"); S.append(("POPE-adv Qwen2-VL", build(d["p_yes"], d["answer"], d["gold"], d["owl"])))
    d = J("exp9_vcd_pope.json");  S.append(("POPE-adv LLaVA+VCD", build(d["p_vcd"], d["answer"], d["gold"], d["owl"])))
    for f, lab in [("exp12_amber_all.json", "AMBER(d) LLaVA")]:
        if os.path.exists(os.path.join(_here, f)):
            d = J(f); S.append((lab, build(d["p_yes"], d["answer"], d["gold"], d["owl"])))
    return S

def run(alpha, reps=60, qs=(None, 0.10, 0.25, 0.50)):
    print(f"===== alpha={alpha:.2f}, delta={DELTA} (3-way split, single FST at full delta, {reps} reps) =====")
    hdr = f"{'setting':20s}{'n':>5s}{'mu':>7s}" + "".join(
        f"{('filter' if q is None else f'q={q:g}'):>15s}" for q in qs)
    print(hdr); print("-" * len(hdr))
    for name, (X, ok_o, ok_r, mg) in settings():
        n = len(ok_o); row = f"{name:20s}{n:5d}{(1-ok_o.mean())*100:6.1f}%"
        base = None
        for q in qs:
            rng = np.random.default_rng(0); C, R = [], []
            for _ in range(reps):
                idx = rng.permutation(n); t = n // 3
                tr, cal, te = idx[:t], idx[t:2*t], idx[2*t:]
                if len(np.unique(ok_o[tr])) < 2: continue
                clf = LogisticRegression(max_iter=1000).fit(X[tr], ok_o[tr])
                s = clf.predict_proba(X)[:, 1]
                lam = calibrate(s[cal], mg[cal], ok_o[cal], ok_r[cal], alpha, DELTA, q)
                cv, rk, _ = deploy(lam, s[te], mg[te], s[cal], mg[cal], ok_o[te], ok_r[te], q)
                C.append(cv); R.append(rk)
            cov = np.mean(C) * 100 if C else 0.0; rsk = np.mean(R) * 100 if R else 0.0
            flag = "!" if rsk > alpha * 100 + 1 else ""
            if q is None: base = cov; row += f"{cov:12.1f}%{flag:>3s}"
            else:         row += f"{cov:8.1f}%{cov-base:+5.1f}{flag:1s}"
        print(row)
    print()

if __name__ == "__main__":
    for a in [0.10, 0.15]:
        run(a)
