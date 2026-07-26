"""
Master comparison across every dataset/backbone collected. For each row: base
accuracy, and (confidence-only vs +grounding) selective prediction under a proper
three-way split (fit combiner / calibrate lambda / test).

Feeds Table 6 (tab:datasets) of the manuscript.

PROTOCOL (canonical, see canonical_fst.py). Previously an exhaustive
max-over-thresholds search; now the canonical ascending-lambda fixed sequence
(stop at first non-rejection, return last passing index) on PAIRED three-way
disjoint splits, so the two coverage columns of each row are directly comparable
and the gain can be tested pairwise.

REPORTING. Per-split sd, 95% CI and a two-sided paired p-value for every gain;
abort rate per arm (this table contains the settings where aborting dominates --
MME-existence at n=60, HallusionBench at chance accuracy -- and reporting a mean
coverage without the abort rate hides the mechanism entirely); validity audited
as the FRACTION of splits with realised risk above alpha, never as mean risk.
"""
import json, os, numpy as np, warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
import canonical_fst as F

_here = os.path.dirname(os.path.abspath(__file__))
J = lambda f: json.load(open(os.path.join(_here, f)))
mm = lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9)
GRID = np.linspace(.02, 1, 99)
ALPHA, DELTA, REPS = 0.10, F.DELTA, F.REPS


def rc(s, c):
    c = c[np.argsort(-s)]
    return np.interp(GRID, np.arange(1, len(c) + 1) / len(c),
                     np.cumsum(1 - c) / np.arange(1, len(c) + 1))


def evaluate(p_yes, answer, correct, owl, extra=None):
    """Metrics for confidence-only and +grounding on identical paired splits."""
    n = len(correct); conf = np.abs(p_yes - .5) * 2
    feats = {"conf": [p_yes, conf]}
    if extra is not None: feats["conf"].append(extra)      # e.g. self-consistency
    has_g = owl is not None and (np.array(owl) >= 0).any()
    if has_g:
        g = np.array(owl, float); g[g < 0] = np.median(g[g >= 0])
        gn = mm(g); gag = np.where(answer == 1, gn, 1 - gn)
        feats["grnd"] = feats["conf"] + [gn, gag]
    R = {k: dict(aurc=[], cov=[], rte=[], ral=[], lam=[]) for k in feats}
    for tr, cal, te in F.SPLITS(n, REPS):
        if len(np.unique(correct[tr])) < 2: continue      # tiny/degenerate fold
        for k, cols in feats.items():
            X = np.column_stack(cols)
            s = LogisticRegression(max_iter=1000).fit(X[tr], correct[tr]).predict_proba(X)[:, 1]
            A = R[k]
            A["aurc"].append(rc(s[te], correct[te]).mean())
            lam = F.fst(s[cal], correct[cal], ALPHA)
            A["lam"].append(lam)
            if lam is None:
                A["cov"].append(0.0); A["rte"].append(None); A["ral"].append(None); continue
            thr = np.quantile(s[cal], 1 - lam)
            m = s[te] >= thr; kk = int(m.sum())
            A["cov"].append(kk / len(te))
            A["rte"].append(float((1 - correct[te][m]).mean()) if kk else None)
            ma = s >= thr
            A["ral"].append(float((1 - correct[ma]).mean()) if ma.sum() else None)
    if not R["conf"]["aurc"]: return None, has_g           # too small to evaluate
    return R, has_g


ROWS = []
r = J("raw_scores.json"); o = J("owlv2_scores.json")
ROWS.append(("POPE (1500)", "LLaVA-1.5", np.array(r["p_yes"]), np.array(r["answer"]),
             np.array(r["correct"]), o["ground_det"], None))
d = J("exp7_pope.json")
ROWS.append(("POPE-adv (450)", "LLaVA-1.5", np.array(d["p_yes"]), np.array(d["answer"]),
             np.array(d["correct"]), d["owl"], np.array(d["sc_yesfrac"])))
d = J("exp8_qwen_pope.json")
ROWS.append(("POPE-adv (600)", "Qwen2-VL-2B", np.array(d["p_yes"]), np.array(d["answer"]),
             np.array(d["correct"]), d["owl"], None))
d = J("exp9_vcd_pope.json")
ROWS.append(("POPE-adv (600)", "LLaVA+VCD", np.array(d["p_vcd"]), np.array(d["answer"]),
             np.array(d["correct"]), d["owl"], None))
d = J("exp10_mme.json")
ROWS.append(("MME (700)", "LLaVA-1.5", np.array(d["p_yes"]), np.array(d["answer"]),
             np.array(d["correct"]), d["owl"], None))
d = J("exp11_mme_exist.json")
ROWS.append(("MME-exist (60)", "LLaVA-1.5", np.array(d["p_yes"]), np.array(d["answer"]),
             np.array(d["correct"]), d["owl"], None))
d = J("exp11_gqa.json")
ROWS.append(("GQA (700)", "LLaVA-1.5", np.array(d["p_yes"]), np.array(d["answer"]),
             np.array(d["correct"]), d["owl"], None))
d = J("exp11_hallusion.json")
ROWS.append(("HallusionBench (447)", "LLaVA-1.5", np.array(d["p_yes"]), np.array(d["answer"]),
             np.array(d["correct"]), d["owl"], None))

print("=" * 132)
print(f"Full sweep, alpha={ALPHA:.2f}, delta={DELTA:.2f}, {REPS} paired three-way splits"
      f"  (canonical fixed-sequence protocol)")
print("=" * 132)
print(f"{'dataset':22s}{'backbone':13s}{'acc':>6s}{'grnd':>6s}"
      f"{'AURC conf':>11s}{'AURC +g':>10s}"
      f"{'cov conf':>16s}{'abort':>7s}{'cov +g':>16s}{'abort':>7s}"
      f"{'exc(te)':>9s}{'exc(all)':>10s}")
print("-" * 132)
gains = []
for name, bb, p, a, c, owl, extra in ROWS:
    R, has_g = evaluate(p, a, c, owl, extra)
    acc = c.mean()
    if R is None:
        print(f"{name:22s}{bb:13s}{acc*100:5.1f}%{('yes' if has_g else 'no'):>6s}"
              f"{'  n too small for a conformal split':>50s}")
        continue
    cf = R["conf"]; gf = R.get("grnd")
    cc = np.array(cf["cov"]) * 100
    best = gf if gf else cf                                # audit the reported arm
    line = (f"{name:22s}{bb:13s}{acc*100:5.1f}%{('yes' if has_g else 'no'):>6s}"
            f"{np.mean(cf['aurc']):11.4f}"
            f"{(np.mean(gf['aurc']) if gf else float('nan')):10.4f}"
            f"{cc.mean():9.1f}+-{cc.std(ddof=1):4.1f}%{F.abort_rate(cf['lam'])*100:6.1f}%")
    if gf:
        gc = np.array(gf["cov"]) * 100
        line += f"{gc.mean():9.1f}+-{gc.std(ddof=1):4.1f}%{F.abort_rate(gf['lam'])*100:6.1f}%"
        gains.append(F.gain_stats(cc, gc, f"{name} {bb}"))
    else:
        line += f"{'n/a':>16s}{'':>7s}"
    line += (f"{F.exceedance(best['rte'],ALPHA)*100:8.1f}%"
             f"{F.exceedance(best['ral'],ALPHA)*100:9.1f}%")
    print(line)
print("=" * 132)
print("grnd=yes: object-existence questions where channel 2 (OWLv2) applies (POPE/AMBER).")
print("MME/GQA/HallusionBench: grounding n/a (attribute/counting/reasoning questions);")
print("  the procedure reduces to filtering, which remains valid.")
print(f"exc columns audit the grounded arm where it exists, else confidence-only:")
print(f"  exceedance = fraction of splits with realised risk > alpha={ALPHA:.2f}, "
      f"vs delta={DELTA:.2f}; exc(te)=test fold, exc(all)=all n items (low-noise proxy).")
print("  Mean realised risk is NOT the guarantee and is deliberately not reported here.")
print("abort = fraction of splits certifying nothing (coverage 0%, included in the mean).")

print(f"\nPaired grounding gains (identical splits, two-sided tests)")
print(F.Gain.header(34))
for g in gains: print(g.line(34))
