"""
Validation of CCRC v2: does the guarantee hold across risk levels, and are the
repairs actually doing the work? Sweeps alpha and reports, for filtering vs CCRC:
certified coverage, realized test risk (validity), repaired fraction, and the
accuracy of the repaired subset. 100 reps for tight error bars. CPU-only.
"""
import json, os, numpy as np, warnings; warnings.filterwarnings("ignore")
from scipy.stats import beta
from sklearn.linear_model import LogisticRegression

_here = os.path.dirname(os.path.abspath(__file__))
DELTA = 0.10
r = json.load(open(os.path.join(_here, "raw_scores.json")))
o = json.load(open(os.path.join(_here, "owlv2_scores.json")))
p = np.array(r["p_yes"]); a = np.array(r["answer"]); gold = np.array(r["gold"])
det = np.array(o["ground_det"])
n = len(gold)
mm = lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9)
conf = np.abs(p - .5) * 2; dn = mm(det)
X = np.column_stack([p, conf, dn, np.where(a == 1, dn, 1 - dn)])
ok_orig = (a == gold).astype(int)
THR = 0.15
det_ans = (det >= THR).astype(int)
ok_rep = (det_ans == gold).astype(int)
margin = np.abs(det - THR)
mu = 1 - ok_orig.mean()

def cp_upper(e, k, d):
    if k == 0: return 1.0
    return 1.0 if e == k else float(beta.ppf(1 - d, e + 1, k - e))

def run(alpha, reps=100):
    kmin = int(np.ceil(np.log(DELTA) / np.log(1 - alpha)))
    rng = np.random.default_rng(0)
    R = {"filter": {"cov": [], "risk": []},
         "CCRC":   {"cov": [], "risk": [], "rep": [], "repacc": []},
         "CCRC*":  {"cov": [], "risk": []}}
    for _ in range(reps):
        idx = rng.permutation(n); t = n // 3
        tr, cal, te = idx[:t], idx[t:2*t], idx[2*t:]
        clf = LogisticRegression(max_iter=1000).fit(X[tr], ok_orig[tr])
        s = clf.predict_proba(X)[:, 1]
        lams = np.linspace(min(0.9, max(0.05, 1.5 * kmin / len(cal))), 1.0, 40)
        cert = {}
        for meth, allow in [("filter", False), ("CCRC", True)]:
            best = None
            for lam in lams:
                acc = s[cal] >= np.quantile(s[cal], 1 - lam)
                rep = (~acc) & (margin[cal] >= np.quantile(margin[cal], 1 - lam)) if allow \
                      else np.zeros_like(acc)
                k = int(acc.sum() + rep.sum())
                if k == 0: continue
                e = int((1 - ok_orig[cal][acc]).sum() + (1 - ok_rep[cal][rep]).sum())
                if cp_upper(e, k, DELTA) <= alpha: best = lam
                else: break
            if best is None:
                R[meth]["cov"].append(0.0); R[meth]["risk"].append(0.0)
                cert[meth]=(None,0.0)
                if allow: R[meth]["rep"].append(0.0); R[meth]["repacc"].append(np.nan)
                continue
            accC = s[cal] >= np.quantile(s[cal], 1 - best)
            repC = (~accC) & (margin[cal] >= np.quantile(margin[cal], 1 - best)) if allow else np.zeros_like(accC)
            cert[meth]=(best, (accC.sum()+repC.sum())/len(cal))
            accT = s[te] >= np.quantile(s[cal], 1 - best)
            repT = (~accT) & (margin[te] >= np.quantile(margin[cal], 1 - best)) if allow \
                   else np.zeros_like(accT)
            k = int(accT.sum() + repT.sum())
            e = int((1 - ok_orig[te][accT]).sum() + (1 - ok_rep[te][repT]).sum())
            R[meth]["cov"].append(k / len(te)); R[meth]["risk"].append(e / max(k, 1))
            if allow:
                R[meth]["rep"].append(int(repT.sum()) / len(te))
                R[meth]["repacc"].append(ok_rep[te][repT].mean() if repT.sum() else np.nan)
        # CCRC*: pick the family with higher CERTIFIED coverage on calibration (delta/2 each)
        pick = max(cert, key=lambda k: cert[k][1]) if cert else "filter"
        allow = (pick == "CCRC"); best = cert.get(pick,(None,0))[0]
        if best is None:
            R["CCRC*"]["cov"].append(0.0); R["CCRC*"]["risk"].append(0.0)
        else:
            accT = s[te] >= np.quantile(s[cal], 1 - best)
            repT = (~accT) & (margin[te] >= np.quantile(margin[cal], 1 - best)) if allow else np.zeros_like(accT)
            k = int(accT.sum()+repT.sum())
            e = int((1-ok_orig[te][accT]).sum() + (1-ok_rep[te][repT]).sum())
            R["CCRC*"]["cov"].append(k/len(te)); R["CCRC*"]["risk"].append(e/max(k,1))
    return R

print(f"POPE/LLaVA-1.5-7B  n={n}  base risk mu={mu:.3f}  delta={DELTA}  (100 reps, 3-way split, FST+CP)")
print(f"{'alpha':>6s} {'filter cov':>11s} {'filt risk':>10s} | {'CCRC cov':>9s} {'CCRC risk':>10s} "
      f"{'repaired':>9s} {'repair acc':>11s} {'gain':>7s} {'valid':>6s}")
print("-" * 92)
for alpha in [0.05, 0.10, 0.15, 0.20]:
    R = run(alpha)
    fc = np.mean(R["filter"]["cov"]) * 100; fr = np.mean(R["filter"]["risk"]) * 100
    cc = np.mean(R["CCRC"]["cov"]) * 100;   cr = np.mean(R["CCRC"]["risk"]) * 100
    rp = np.mean(R["CCRC"]["rep"]) * 100
    ra = np.nanmean(R["CCRC"]["repacc"]) * 100
    sc = np.mean(R["CCRC*"]["cov"])*100; sr = np.mean(R["CCRC*"]["risk"])*100
    valid = "OK" if (fr <= alpha*100+1 and cr <= alpha*100+1 and sr <= alpha*100+1) else "BAD"
    print(f"{alpha:6.2f} {fc:10.1f}% {fr:9.1f}% | {cc:8.1f}% {cr:9.1f}% {rp:8.1f}% "
          f"{ra:10.1f}% {cc-fc:+6.1f} {valid:>6s}   CCRC*={sc:5.1f}% (risk {sr:4.1f}%, {sc-fc:+5.1f})")
print("-" * 92)
print("repair acc = accuracy of the detector-supplied answers on repaired items")
print("(must be >= 1-alpha for repairs to be certifiable; this is the independent channel)")
