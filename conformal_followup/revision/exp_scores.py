"""Score comparison and baselines under P4 (R1: all tables on one protocol). Filtering only (no repair) for the
score ladder; plus detector-only selective prediction. Writes results_scores.json.
AURC is computed on each split's test fold from the score fitted on that split's fit fold."""
import json, numpy as np, ccrc_rev as R
S = R.load_settings(); M = R.Mode(); out = {}

def aurc(run, S_):
    v = []
    for (tr, cal, te), s in zip(run.splits, run.scores):
        if s is None: continue
        o = np.argsort(-s[te], kind="stable"); err = (1 - S_.ok[te][o]).cumsum() / np.arange(1, len(te) + 1)
        v.append(err.mean())
    return float(np.mean(v))

def feats(s, kind):
    p, conf = s.p, np.abs(s.p - .5) * 2
    if s.clip is not None:                       # unsupervised standardisation (no labels used)
        s = R.take(s, np.arange(s.n)); s.clip = (s.clip - s.clip.mean()) / s.clip.std()
    base = s.features("impute")
    if kind == "conf": return conf[:, None]
    if kind == "learned": return np.column_stack([p, conf])
    if kind == "clip": return np.column_stack([p, conf, s.clip])
    if kind == "det": return base
    if kind == "both": return np.column_stack([base, s.clip])
    if kind == "clip_only": return s.clip[:, None]
    if kind == "sc": return s.sc[:, None]
    if kind == "conf_sc": return np.column_stack([p, conf, s.sc])
    if kind == "det_sc": return np.column_stack([base, s.sc])

def block(name, kinds):
    s = S[name]
    for kind in kinds:
        run = R.Runner(s, M, X=feats(s, kind))
        for a in (0.05, 0.10, 0.15, 0.20):
            f = R.summarise(run.run(a, gate=None), a)
            out[f"{name}|{kind}|{a}"] = dict(cov=f["cov"], sd=f["cov_sd"], abort=f["abort"], exc=f["exc_te"], aurc=aurc(run, s))
        print(name, kind, {a: round(out[f"{name}|{kind}|{a}"]["cov"] * 100, 1) for a in (0.1,)}, round(out[f"{name}|{kind}|0.1"]["aurc"], 4), flush=True)
block("POPE-1500 LLaVA", ["clip_only", "conf", "learned", "clip", "det", "both"])
block("POPE-adv LLaVA", ["sc", "conf_sc", "det_sc", "learned", "det"])

# detector-only selective prediction: emit the detector's answer, ranked by its own evidence margin
for name in R.MAIN:
    s = S[name]; T = R.take(s, np.arange(s.n))
    T.a, T.ok = s.b.copy(), s.okr.copy()                                  # the emitted answer is the detector's
    X = np.column_stack([np.maximum(s.m, 0), s.g.astype(float), np.where(s.g, s.o, 0.0)])
    run = R.Runner(T, M, X=X)
    for a in (0.10, 0.15):
        f = R.summarise(run.run(a, gate=None), a)
        out[f"{name}|detector_only|{a}"] = dict(cov=f["cov"], sd=f["cov_sd"], abort=f["abort"], exc=f["exc_te"], acc=float(s.okr[s.g].mean()))
    print(name, "detector-only", round(out[f"{name}|detector_only|0.1"]["cov"] * 100, 1), flush=True)
json.dump(out, open("results_scores.json", "w"), indent=1)
