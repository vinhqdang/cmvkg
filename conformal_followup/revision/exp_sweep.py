"""Datasets where the detector does not apply (R1.9: what happens to items without usable evidence).
Filtering only, image-grouped folds, confidence score. Writes results_sweep.json."""
import json, numpy as np, ccrc_rev as R
ids = R._J("revision/image_ids.json"); out = {}
for f, lab in [("exp10_mme.json", "MME (700)"), ("exp11_gqa.json", "GQA testdev (700)"), ("exp11_hallusion.json", "HallusionBench (447)")]:
    d = R._J(f); s = R.Setting(lab, d["p_yes"], d["answer"], d["gold"], d["owl"], ids[f], d["obj"])
    run = R.Runner(s, R.Mode(), X=np.column_stack([s.p, np.abs(s.p - .5) * 2]))
    row = dict(n=s.n, images=s.n_img, mu=float(s.mu), grounded=int(s.g.sum()))
    for a in (0.10, 0.15, 0.20):
        r = R.summarise(run.run(a, gate=None), a); row[str(a)] = dict(cov=r["cov"], sd=r["cov_sd"], abort=r["abort"], exc=r["exc_te"])
    out[lab] = row; print(lab, {k: (round(v["cov"] * 100, 1), round(v["abort"] * 100)) for k, v in row.items() if isinstance(v, dict)}, flush=True)
json.dump(out, open("results_sweep.json", "w"), indent=1)
