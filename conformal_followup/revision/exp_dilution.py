"""Gain decomposition under P4 (fixed gate q=0.10, all items): (i) repairs available at filtering's own
threshold (emitted automatically) and (ii) the extra accepted mass admitted because the certified threshold
moved (dilution). Conditional on both arms certifying. Writes results_dilution.json."""
import json, numpy as np, ccrc_rev as R
from scipy.stats import ttest_1samp
S = R.load_settings(); M = R.Mode(); out = {}
for name in R.MAIN:
    run = R.Runner(S[name], M); s_ = S[name]
    for a in (0.10, 0.15):
        ks = R.k_start(a, R.DELTA, M.e0); rec = dict(moved=[], gain=[], rep=[], dil=[], both=0, tot=0)
        for (tr, cal, te), s, ref in zip(run.splits, run.scores, run.refs):
            if s is None: continue
            rec["tot"] += 1
            thr = np.quantile(ref, 1 - R._grid_lams(len(run.sub) // 3, ks, M.n_grid, M.c))
            rg = R.Region(s_, "margin", 0.10, tr, ref, M, s_.m[cal])
            jf = R.fst_region(s_, s, cal, thr, a, R.DELTA, None); jc = R.fst_region(s_, s, cal, thr, a, R.DELTA, rg)
            if jf is None or jc is None: continue
            rec["both"] += 1; n = len(te)
            acc_f = s[te] >= thr[jf]; rep_f = (~acc_f) & rg.mask(te, s)
            acc_c = s[te] >= thr[jc]; rep_c = (~acc_c) & rg.mask(te, s)
            cf, cc = acc_f.sum() / n, (acc_c.sum() + rep_c.sum()) / n
            rec["moved"].append(jf != jc); rec["gain"].append((cc - cf) * 100)
            rec["rep"].append(rep_f.sum() / n * 100); rec["dil"].append((cc - cf - rep_f.sum() / n) * 100)
        d = np.array(rec["dil"]); h = 1.96 * d.std(ddof=1) / np.sqrt(len(d))
        out[f"{name}|{a}"] = dict(moved=float(np.mean(rec["moved"])), gain=float(np.mean(rec["gain"])), rep=float(np.mean(rec["rep"])), dil=float(d.mean()),
                                  dil_lo=float(d.mean() - h), dil_hi=float(d.mean() + h), p=float(ttest_1samp(d, 0).pvalue) if d.std() > 0 else 1.0,
                                  both=rec["both"], tot=rec["tot"])
    print(name, flush=True)
json.dump(out, open("results_dilution.json", "w"), indent=1)
