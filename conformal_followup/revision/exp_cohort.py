"""Cohort flow (R1.9): what enters each experiment, what is lost, and why. Writes results_cohort.json."""
import json, numpy as np, ccrc_rev as R
S = R.load_settings(); out = {}
def icc(y, img):
    u, inv = np.unique(img, return_inverse=True); k, n = len(u), len(y)
    m = np.bincount(inv); gm = np.bincount(inv, weights=y) / m; ssb = (m * (gm - y.mean()) ** 2).sum(); ssw = ((y - gm[inv]) ** 2).sum()
    msb, msw = ssb / (k - 1), ssw / (n - k); n0 = (n - (m ** 2).sum() / n) / (k - 1)
    return float((msb - msw) / (msb + (n0 - 1) * msw))
imgs = {}
for name in R.MAIN:
    s = S[name]; sp = R.make_splits(s, 100)
    obj_none = int(sum(o is None for o in s.obj)) if s.obj is not None else None
    out[name] = dict(n_items=s.n, n_images=s.n_img, items_per_image=s.n / s.n_img, n_grounded=int(s.g.sum()), n_ungrounded=int((~s.g).sum()),
                     n_unparsed_object=obj_none, mu=float(s.mu), mu_grounded=float(1 - s.ok[s.g].mean()), acc_detector_grounded=float(s.okr[s.g].mean()),
                     fold_images=[float(np.mean([len(set(s.img[f[i]])) for f in sp])) for i in range(3)],
                     fold_items=[float(np.mean([len(f[i]) for f in sp])) for i in range(3)],
                     icc_model_error=icc((1 - s.ok).astype(float), s.img), icc_detector_error=icc((1 - s.okr[s.g]).astype(float), s.img[s.g]))
    imgs[name] = set(s.img)
names = list(imgs)
out["_image_overlap"] = {f"{a} & {b}": len(imgs[a] & imgs[b]) for i, a in enumerate(names) for b in names[i + 1:]}
out["_unique_images_total"] = len(set().union(*imgs.values()))
json.dump(out, open("results_cohort.json", "w"), indent=1)
for k, v in out.items():
    print(k, v if not isinstance(v, dict) else {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items()})
