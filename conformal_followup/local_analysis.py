"""
Local (CPU, free) analysis of the extracted LLaVA+CLIP POPE signals.

Isolates the contribution of CLIP grounding to conformal SELECTIVE prediction by
comparing LEARNED combiners with vs without grounding features (so the only
difference is grounding, not "learning"). Robust over 20 random cal/test splits.

Scores compared:
  raw_conf         : VLM confidence margin (standard baseline, no learning)
  learned_conf     : logistic P(correct) from [conf, p_yes]
  learned_conf+grd : logistic P(correct) from [conf, p_yes, ground, ground*answer]

Metrics (test):
  AURC            : area under risk-coverage curve (LOWER = better selective score)
  cov@10%         : RCPS coverage at alpha=0.10 (Clopper-Pearson), higher=better
  err@10%         : realized error among accepted (must be <= ~0.10 -> valid)
"""
import json, os, numpy as np
from scipy.stats import beta
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

_here = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(_here, "raw_scores.json")))
p_yes  = np.array(d["p_yes"]); ground = np.array(d["ground"])
answer = np.array(d["answer"]); correct = np.array(d["correct"])
n = len(correct)
conf = np.abs(p_yes - 0.5) * 2
g = (ground - ground.min()) / (ground.max() - ground.min())
g_agree = np.where(answer == 1, g, 1 - g)      # grounding that agrees with answer

FEATS = {
    "learned_conf":     np.column_stack([conf, p_yes]),
    "learned_conf+grd": np.column_stack([conf, p_yes, g, g_agree]),
}
ALPHA, DELTA = 0.10, 0.10

def cp_upper(e, k, delta):
    if k == 0: return 1.0
    return 1.0 if e == k else float(beta.ppf(1 - delta, e + 1, k - e))

def rcps_cov(score_cal, corr_cal, score_test, corr_test, alpha, delta):
    tau, best = None, -1
    for t in np.unique(score_cal):
        m = score_cal >= t; k = int(m.sum())
        if k == 0: continue
        e = int((1 - corr_cal[m]).sum())
        if cp_upper(e, k, delta) <= alpha and k > best:
            best, tau = k, float(t)
    if tau is None: return 0.0, 0.0
    m = score_test >= tau; k = int(m.sum())
    if k == 0: return 0.0, 0.0
    return k / len(score_test), float((1 - corr_test[m]).mean())

def aurc(score, corr):
    order = np.argsort(-score)
    c = corr[order]
    risks = (np.cumsum(1 - c) / np.arange(1, len(c) + 1))
    return float(risks.mean())

rng = np.random.default_rng(0)
agg = {}
SPLITS = 20
for s in range(SPLITS):
    idx = rng.permutation(n); h = n // 2; cal, test = idx[:h], idx[h:]
    # raw conf baseline
    for name, sc in [("raw_conf", conf)]:
        cov, err = rcps_cov(sc[cal], correct[cal], sc[test], correct[test], ALPHA, DELTA)
        agg.setdefault(name, []).append((aurc(sc[test], correct[test]),
                                         cov, err, roc_auc_score(correct[test], sc[test])))
    # learned combiners
    for name, X in FEATS.items():
        clf = LogisticRegression(max_iter=1000).fit(X[cal], correct[cal])
        pc = clf.predict_proba(X)[:, 1]
        cov, err = rcps_cov(pc[cal], correct[cal], pc[test], correct[test], ALPHA, DELTA)
        agg.setdefault(name, []).append((aurc(pc[test], correct[test]),
                                         cov, err, roc_auc_score(correct[test], pc[test])))

print(f"POPE n={n}  VLM acc={correct.mean():.3f}   ({SPLITS} splits, mean±std)")
print(f"{'score':20s} {'AURC↓':>14s} {'cov@10%↑':>14s} {'err@10%':>12s} {'AUROC↑':>12s}")
for name in ["raw_conf", "learned_conf", "learned_conf+grd"]:
    a = np.array(agg[name])
    m, sd = a.mean(0), a.std(0)
    print(f"{name:20s} {m[0]:.4f}±{sd[0]:.3f} {m[1]:.3f}±{sd[1]:.3f}  "
          f"{m[2]:.3f}±{sd[2]:.3f} {m[3]:.4f}±{sd[3]:.3f}")

# paired improvement of grounding (conf+grd vs conf), same splits
gain_cov = np.array(agg["learned_conf+grd"])[:,1] - np.array(agg["learned_conf"])[:,1]
gain_aurc = np.array(agg["learned_conf"])[:,0] - np.array(agg["learned_conf+grd"])[:,0]
print(f"\nGrounding effect (paired, conf+grd vs conf):")
print(f"  Δcoverage@10% = {gain_cov.mean():+.4f} ± {gain_cov.std():.4f}")
print(f"  ΔAURC (improvement) = {gain_aurc.mean():+.4f} ± {gain_aurc.std():.4f}")
