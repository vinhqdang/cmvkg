"""
Revised CCRC protocol (P4) -- one implementation behind every revised number.

What changed relative to canonical_fst.py / ccrc_v3.py (each change answers a
specific reviewer point; see revision/RESPONSE.md):

  * IMAGE-GROUPED folds. POPE contributes six questions per image, so item-level
    splits put questions about the same image in the fit, calibration and test
    folds at once. Folds are now unions of whole images.            (R1.6)
  * FIXED VALUE GRID. Thresholds on the correctness score, and the repair gate on
    the evidence margin, are numeric VALUES derived from the FIT fold before any
    calibration outcome is read -- not empirical quantiles of the calibration fold
    that is then used to count errors. This is the object Theorem 1 covers. The
    score-value grid is read off out-of-fold scores of the fit fold, so that it
    reflects out-of-sample score behaviour.                         (R1.1)
  * ANALYTIC START. The sequence starts where e0 errors can be tolerated:
    k_start(e0) = min{k : CP_upper(e0, k; delta) <= alpha}  (22, 38, 52, 65 at
    alpha = delta = 0.10 for e0 = 0..3), and the first hypothesis is placed where
    c * k_start items are expected to be emitted. e0 and c are design constants,
    fixed before the final runs and ablated, never tuned on the test fold.  (R1.5)
  * GATE SELECTION WITHOUT PEEKING. `gate="fitsel"` picks the repair gate (or no
    repair at all) on the FIT fold alone, so calibration keeps the full delta;
    `gate="union"` certifies a pre-specified set of gates at delta/|Q| each.
    A fixed q is available but is not certified when it was chosen with
    knowledge of the benchmark.                                      (R1.4)
  * ALL-ITEM ANALYSIS. Items without detector evidence stay in the cohort: their
    grounding features are zeroed, a missing flag is added to the score, and they
    can never be repaired (m = -1).                                  (R1.9)
  * `Mode.legacy()` reproduces the previous protocol (item-level folds, quantiles
    of the calibration fold, e0 = 0, grounded items only); regress_legacy.py checks
    that it recovers the previously reported Table-5 cells.

Everything is deterministic in (setting, mode, reps, seed).
"""
import json, os, functools, warnings
import numpy as np
from scipy.stats import beta
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DELTA = 0.10
THETA = 0.15            # channel-2 operating point on the OWLv2 score (unchanged)
REPS = int(os.environ.get("CCRC_REPS", "400"))
KEYS = ("k", "e", "k_acc", "k_rep", "e_acc", "e_rep", "changed", "w2r", "r2w", "agree_right", "agree_wrong",
        "fp_corr", "fn_corr", "fp_tot", "fn_tot")


# ------------------------------------------------------------------ bounds
@functools.lru_cache(maxsize=None)
def cp_upper(e, k, d):
    """One-sided Clopper-Pearson upper limit for e errors in k trials."""
    if k == 0: return 1.0
    return 1.0 if e >= k else float(beta.ppf(1 - d, e + 1, k - e))


@functools.lru_cache(maxsize=None)
def cp_lower(e, k, d):
    if k == 0 or e == 0: return 0.0
    return float(beta.ppf(d, e, k - e + 1))


@functools.lru_cache(maxsize=None)
def k_start(alpha, delta, e0=0):
    """Smallest emitted count at which e0 observed errors still certify risk <= alpha."""
    k = e0 + 1
    while cp_upper(e0, k, delta) > alpha: k += 1
    return k


# -------------------------------------------------------------------- data
def _J(f): return json.load(open(os.path.join(ROOT, f)))


class Setting:
    """One benchmark x backbone cell with everything the protocol needs."""

    def __init__(self, name, p, ans, gold, owl, img, obj=None, clip=None, sc=None):
        self.name = name
        self.p = np.asarray(p, float); self.a = np.asarray(ans, int)
        self.y = np.asarray(gold, int); self.o = np.asarray(owl, float)
        self.img = np.asarray(img).astype(str); self.obj = obj
        self.clip = None if clip is None else np.asarray(clip, float)
        self.sc = None if sc is None else np.asarray(sc, float)
        self.n = len(self.y)
        self.g = self.o >= 0                      # detector returned a value
        self.b = (self.o >= THETA).astype(int)    # channel-2 answer
        self.ok = (self.a == self.y).astype(int)  # original answer correct
        self.okr = (self.b == self.y).astype(int) # channel-2 answer correct
        self.m = np.where(self.g, np.abs(self.o - THETA), -1.0)   # evidence margin
        self.mu = 1 - self.ok.mean()
        self.n_img = len(set(self.img))

    def features(self, missing="impute", legacy_norm=False):
        """Score features. `drop` keeps legacy semantics (rows with g=False are
        removed by the caller); `impute` zeroes grounding features, adds a flag."""
        p, a, o, g = self.p, self.a, self.o, self.g
        conf = np.abs(p - .5) * 2
        if legacy_norm:                            # min-max over kept items (legacy)
            dn = np.zeros(self.n); lo, hi = o[g].min(), o[g].max()
            dn[g] = (o[g] - lo) / (hi - lo + 1e-9)
        else:
            dn = np.where(g, o, 0.0)
        signed = np.where(a == 1, dn, 1 - dn)
        if missing == "drop":
            return np.column_stack([p, conf, dn, signed])
        signed = np.where(g, signed, 0.0)
        return np.column_stack([p, conf, dn, signed, (~g).astype(float)])


def load_settings():
    S = {}
    r, o = _J("raw_scores.json"), _J("owlv2_scores.json")
    ids = _J("revision/image_ids.json")
    S["POPE-1500 LLaVA"] = Setting("POPE-1500 LLaVA", r["p_yes"], r["answer"], r["gold"],
                                   o["ground_det"], ids["raw_scores.json"], clip=r["ground"])
    d = _J("exp7_pope.json")
    S["POPE-adv LLaVA"] = Setting("POPE-adv LLaVA", d["p_yes"], d["answer"], d["gold"], d["owl"],
                                  ids["exp7_pope.json"], d["obj"], sc=d["sc_yesfrac"])
    d = _J("exp8_qwen_pope.json")
    S["POPE-adv Qwen2-VL"] = Setting("POPE-adv Qwen2-VL", d["p_yes"], d["answer"], d["gold"], d["owl"],
                                     ids["exp8_qwen_pope.json"], d["obj"])
    d = _J("exp9_vcd_pope.json")
    S["POPE-adv LLaVA+VCD"] = Setting("POPE-adv LLaVA+VCD", d["p_vcd"], d["answer"], d["gold"], d["owl"],
                                      ids["exp9_vcd_pope.json"], d["obj"])
    d = _J("exp12_amber_all.json"); a_ids = ids["exp12_amber_all.json"]
    p, ans, gold, owl, obj = list(d["p_yes"]), list(d["answer"]), list(d["gold"]), list(d["owl"]), list(d["obj"])
    ext = os.path.join(ROOT, "exp17_amber_ext.json")
    if os.path.exists(ext):                       # items 500.. of the same shuffled list (colab_exp17_amber_ext.py)
        e = _J("exp17_amber_ext.json")
        p += e["p_yes"]; ans += e["answer"]; gold += e["gold"]; owl += e["owl"]; obj += e["obj"]; a_ids = list(a_ids) + [str(x) for x in e["image"]]
    S["AMBER(d) LLaVA"] = Setting("AMBER(d) LLaVA", p, ans, gold, owl, a_ids, obj)
    big = load_pope_extended()
    if big is not None: S["POPE-3000 LLaVA"] = big
    return S


def load_pope_extended():
    """POPE adversarial rows 0..2999 (500 images): the cached 1500 rows plus the rows extracted by
    colab_exp16_pope_ext.py. Returns None when the extension file is absent."""
    f = os.path.join(ROOT, "exp16_pope_ext.json")
    if not os.path.exists(f): return None
    r, o, e = _J("raw_scores.json"), _J("owlv2_scores.json"), _J("exp16_pope_ext.json")
    ids = _J("revision/image_ids.json")["raw_scores.json"] + [str(x) for x in e["image_source"]]
    return Setting("POPE-3000 LLaVA", r["p_yes"] + e["p_yes"], r["answer"] + e["answer"], r["gold"] + e["gold"],
                   o["ground_det"] + e["ground_det"], ids)


MAIN = ["POPE-1500 LLaVA", "POPE-adv LLaVA", "POPE-adv Qwen2-VL", "POPE-adv LLaVA+VCD", "AMBER(d) LLaVA"]
if os.path.exists(os.path.join(ROOT, "exp16_pope_ext.json")): MAIN.insert(1, "POPE-3000 LLaVA")

_ARR = ("p", "a", "y", "o", "img", "g", "b", "ok", "okr", "m")


def take(S, idx):
    """Sub-Setting restricted to rows `idx` (rows may repeat)."""
    T = Setting.__new__(Setting); T.__dict__.update(S.__dict__)
    for k in _ARR: setattr(T, k, getattr(S, k)[idx])
    for k in ("clip", "sc"):
        v = getattr(S, k); setattr(T, k, None if v is None else v[idx])
    T.n = len(idx); T.n_img = len(set(T.img)); T.mu = 1 - T.ok.mean()
    return T


def resample_images(S, rng):
    """Image-level bootstrap replicate: images drawn with replacement; a duplicated
    image keeps its identifier, so it is always assigned to a single fold."""
    uniq, inv = np.unique(S.img, return_inverse=True)
    draw = rng.integers(0, len(uniq), len(uniq))
    members = [np.flatnonzero(inv == u) for u in range(len(uniq))]
    return take(S, np.concatenate([members[u] for u in draw]))


# ------------------------------------------------------------------ splits
def make_splits(S, reps=REPS, seed=0, grouping="image"):
    """Paired three-way splits (fit / calibrate / test), by image or by item.

    Deterministic in (setting, reps, seed, grouping): every arm of every
    comparison sees byte-identical folds.  Each fold receives a third of the
    grouping units (images for `image`, items for `item`).
    """
    rng = np.random.default_rng(seed)
    out = []
    if grouping == "item":
        t = S.n // 3
        for _ in range(reps):
            idx = rng.permutation(S.n)
            out.append((np.sort(idx[:t]), np.sort(idx[t:2 * t]), np.sort(idx[2 * t:])))
        return out
    uniq, inv = np.unique(S.img, return_inverse=True)
    t = len(uniq) // 3
    for _ in range(reps):
        perm = rng.permutation(len(uniq))
        fold = np.empty(len(uniq), int)
        fold[perm[:t]] = 0; fold[perm[t:2 * t]] = 1; fold[perm[2 * t:]] = 2
        f = fold[inv]
        out.append((np.flatnonzero(f == 0), np.flatnonzero(f == 1), np.flatnonzero(f == 2)))
    return out


# --------------------------------------------------------- score fitting
def fit_logistic(X, y, C=1.0, iters=50):
    """L2 logistic regression (intercept unpenalised), Newton-Raphson. Matches the fully
    converged sklearn.linear_model.LogisticRegression(C=1) optimum to ~2e-7 on these data
    (revision/check_logistic.py; sklearn's default tolerance leaves ~5e-3); ~100x faster,
    which is what makes the image-level bootstrap affordable."""
    n, d = X.shape
    Xb = np.column_stack([X, np.ones(n)])
    w = np.zeros(d + 1)
    reg = np.eye(d + 1) / C; reg[-1, -1] = 0.0
    for _ in range(iters):
        z = Xb @ w; p = 1 / (1 + np.exp(-np.clip(z, -35, 35)))
        grad = Xb.T @ (p - y) + reg @ w
        H = (Xb * (p * (1 - p))[:, None]).T @ Xb + reg + 1e-10 * np.eye(d + 1)
        step = np.linalg.solve(H, grad); w -= step
        if np.max(np.abs(step)) < 1e-9: break
    return w


def predict(w, X):
    return 1 / (1 + np.exp(-np.clip(np.column_stack([X, np.ones(len(X))]) @ w, -35, 35)))


# --------------------------------------------------------------- procedure
class Mode:
    """Protocol switches. `Mode.legacy()` is the previous protocol; defaults are P4."""

    def __init__(self, grouping="image", grid="value", gate_kind="value", e0=1,
                 gate="fitsel", qs=(0.05, 0.10, 0.25), q=0.10, missing="impute",
                 legacy_norm=False, n_grid=40, oof=True, k_folds=5, c=1.5,
                 qs_sel=(0.05, 0.10, 0.25, 0.50), fallback=True, rule="margin"):
        self.__dict__.update(locals()); del self.__dict__["self"]

    @staticmethod
    def legacy():
        return Mode(grouping="item", grid="rank", gate_kind="rank", e0=0, gate="fixed",
                    q=0.10, missing="drop", legacy_norm=True)


def _grid_lams(n_cal_expected, ks, n, c=1.5):
    lam0 = min(0.9, max(0.05, c * ks / max(n_cal_expected, 1)))
    return np.linspace(lam0, 1.0, n)


def fst(s_cal, ok_cal, thr, alpha, delta, rep_cal=None, okr_cal=None):
    """Ascending fixed sequence over the pre-specified threshold values `thr`
    (ascending permissiveness = descending value). Stops at the first
    non-rejection; returns the last passing index or None (abort)."""
    best = None
    for j, t in enumerate(thr):
        acc = s_cal >= t
        k = int(acc.sum()); e = k - int(ok_cal[acc].sum())
        if rep_cal is not None:
            rep = (~acc) & rep_cal
            k += int(rep.sum()); e += int(rep.sum()) - int(okr_cal[rep].sum())
        if k == 0: break
        if cp_upper(e, k, delta) <= alpha: best = j
        else: break
    return best


class Region:
    """A repair region fixed by fit-fold information only: mask(idx, s) -> bool mask.
    Also carries the replacement answer b and whether it is right (okr)."""

    def __init__(self, S, rule, q, tr, ref, mode, m_cal):
        self.S, self.rule, self.q = S, rule, q
        if rule == "margin":
            fit_m = S.m[tr] if mode.gate_kind == "value" else m_cal
            fit_m = fit_m[fit_m >= 0]
            self.gv = np.quantile(fit_m, 1 - q) if len(fit_m) else np.inf
        elif rule in ("sym", "pos", "neg"):
            pos_tr = tr[S.g[tr] & (S.o[tr] >= THETA)]; neg_tr = tr[S.g[tr] & (S.o[tr] < THETA)]
            self.mp = (S.o - THETA) / (1 - THETA); self.mn = (THETA - S.o) / THETA
            self.gp = np.quantile(self.mp[pos_tr], 1 - q) if len(pos_tr) else np.inf
            self.gn = np.quantile(self.mn[neg_tr], 1 - q) if len(neg_tr) else np.inf
        elif rule == "self":
            self.gs = np.quantile(ref, q)           # bottom-q of out-of-fold scores
        else:
            raise ValueError(rule)

    def mask(self, idx, s):
        S = self.S
        if self.rule == "margin": return S.m[idx] >= self.gv
        if self.rule == "self":   return s[idx] <= self.gs
        g, o = S.g[idx], S.o[idx]
        pos = g & (o >= THETA) & (self.mp[idx] >= self.gp)
        neg = g & (o < THETA) & (self.mn[idx] >= self.gn)
        return pos if self.rule == "pos" else neg if self.rule == "neg" else (pos | neg)

    def answers(self, idx):
        S = self.S
        if self.rule == "self": return 1 - S.a[idx], 1 - S.ok[idx]       # flip; right iff model wrong
        return S.b[idx], S.okr[idx]


def evaluate(S, s, idx, thr_val, region):
    """Emitted-set counts of the policy (thr_val, region) on items `idx`; region None = filtering."""
    acc = s[idx] >= thr_val
    ok_a = S.ok[idx]; a_i = S.a[idx]
    if region is None:
        rep = np.zeros_like(acc); b_i = a_i; ok_r = ok_a
    else:
        rep = (~acc) & region.mask(idx, s); b_i, ok_r = region.answers(idx)
    k_acc, k_rep = int(acc.sum()), int(rep.sum())
    e_acc = k_acc - int(ok_a[acc].sum()); e_rep = k_rep - int(ok_r[rep].sum())
    ch = rep & (b_i != a_i)
    y_i = S.y[idx]
    fp, fn = (a_i == 1) & (y_i == 0), (a_i == 0) & (y_i == 1)          # the model's two error types
    return dict(n=len(idx), fp_corr=int((rep & fp & (b_i == 0)).sum()), fn_corr=int((rep & fn & (b_i == 1)).sum()),
                fp_tot=int(fp.sum()), fn_tot=int(fn.sum()), k=k_acc + k_rep, e=e_acc + e_rep, k_acc=k_acc, k_rep=k_rep,
                e_acc=e_acc, e_rep=e_rep, changed=int(ch.sum()),
                w2r=int((ch & (ok_a == 0) & (ok_r == 1)).sum()),
                r2w=int((ch & (ok_a == 1) & (ok_r == 0)).sum()),
                agree_right=int((rep & (b_i == a_i) & (ok_a == 1)).sum()),
                agree_wrong=int((rep & (b_i == a_i) & (ok_a == 0)).sum()))


def fst_region(S, s, idx, thr, alpha, delta, region):
    """FST on items `idx` for the policy family indexed by thr with a fixed repair region."""
    if region is None: return fst(s[idx], S.ok[idx], thr, alpha, delta)
    ok_r = region.answers(idx)[1]
    return fst(s[idx], S.ok[idx], thr, alpha, delta, rep_cal=region.mask(idx, s), okr_cal=ok_r)


class Runner:
    """Runs filtering and CCRC on identical paired splits; caches scores per split."""

    def __init__(self, S, mode, reps=REPS, seed=0, X=None):
        self.S, self.mode, self.reps = S, mode, reps
        self.sub = np.flatnonzero(S.g) if mode.missing == "drop" else np.arange(S.n)
        self.X = S.features(mode.missing, mode.legacy_norm) if X is None else X
        if mode.missing == "drop":
            self.splits = [tuple(self.sub[f] for f in sp)
                           for sp in make_splits(take(S, self.sub), reps, seed, mode.grouping)]
        else:
            self.splits = make_splits(S, reps, seed, mode.grouping)
        self.scores, self.refs = [], []
        for tr, cal, te in self.splits:
            if len(np.unique(S.ok[tr])) < 2: self.scores.append(None); self.refs.append(None); continue
            w = fit_logistic(self.X[tr], S.ok[tr])
            s = np.full(S.n, np.nan); s[self.sub] = predict(w, self.X[self.sub])
            self.scores.append(s)
            self.refs.append(self._oof(tr) if (mode.grid == "value" and mode.oof) else s[tr])

    def _oof(self, tr):
        """Out-of-fold scores on the fit fold (image-grouped K-fold inside the fit fold)."""
        S, m = self.S, self.mode
        rng = np.random.default_rng(int(tr[0]) * 7919 + len(tr))
        uniq, inv = np.unique(S.img[tr], return_inverse=True)
        f = (rng.permutation(len(uniq)) % m.k_folds)[inv]; ref = np.empty(len(tr))
        Xt, yt = self.X[tr], S.ok[tr]
        for k in range(m.k_folds):
            te_k, tr_k = f == k, f != k
            if te_k.sum() == 0: continue
            if len(np.unique(yt[tr_k])) < 2: ref[te_k] = yt[tr_k].mean(); continue
            ref[te_k] = predict(fit_logistic(Xt[tr_k], yt[tr_k]), Xt[te_k])
        return ref

    def run(self, alpha, delta=DELTA, gate="mode", q=None, qs=None, e0=None, rule=None, c=None):
        """One (alpha, arm) cell over all splits.
        gate: None = filtering only | 'fixed' | 'union' | 'fitsel' | 'mode' (Mode default)."""
        S, m = self.S, self.mode
        gate = m.gate if gate == "mode" else gate
        e0 = m.e0 if e0 is None else e0
        q = m.q if q is None else q
        qs = m.qs if qs is None else qs
        rule = m.rule if rule is None else rule
        c = m.c if c is None else c
        arms = [None] if gate is None else ([q] if gate in ("fixed", "fitsel") else list(qs))
        dl = delta if gate in (None, "fixed", "fitsel") else delta / len(arms)   # per-hypothesis level
        ks = k_start(alpha, dl, e0)                                              # start uses that level
        out = []
        for (tr, cal, te), s, ref in zip(self.splits, self.scores, self.refs):
            if s is None: continue
            lams = _grid_lams(len(self.sub) // 3, ks, m.n_grid, c)
            thr = np.quantile(ref if m.grid == "value" else s[cal], 1 - lams)
            mk = lambda qq: None if qq is None else Region(S, rule, qq, tr, ref, m, S.m[cal])
            arms_i = arms
            if gate == "fitsel":
                # choose the gate (or no repair) on the FIT fold only: out-of-fold scores, fit-fold labels.
                # Independent of the calibration fold, so calibration keeps the full delta.
                best_q, best_k = (None if m.fallback else q), -1
                sfit = np.full(S.n, np.nan); sfit[tr] = ref
                for qq in ([None] if m.fallback else []) + sorted(m.qs_sel):    # filtering listed first: wins ties
                    rg = mk(qq)
                    jf = fst_region(S, sfit, tr, thr, alpha, dl, rg)
                    if jf is None: continue
                    kf = evaluate(S, sfit, tr, thr[jf], rg)["k"]
                    if kf > best_k: best_q, best_k = qq, kf
                arms_i = [best_q]
            best = None
            for qq in arms_i:
                rg = mk(qq)
                j = fst_region(S, s, cal, thr, alpha, dl, rg)
                if j is None: continue
                kc = evaluate(S, s, cal, thr[j], rg)["k"]
                if best is None or kc > best[0]: best = (kc, j, rg, qq)
            if best is None:
                out.append(dict(abort=True, cov=0.0, n=len(te), q=None, j=None, risk=np.nan,
                                **{k: 0 for k in KEYS}))
                continue
            kc, j, rg, qq = best
            r = evaluate(S, s, te, thr[j], rg)
            r.update(abort=False, cov=r["k"] / r["n"], q=qq, j=j, k_cal=kc,
                     risk=(r["e"] / r["k"]) if r["k"] else np.nan)
            out.append(r)
        return out


# ---------------------------------------------------------------- summaries
def summarise(res, alpha, d_lcb=0.10):
    ab = np.array([r["abort"] for r in res], bool)
    cov = np.array([r["cov"] for r in res], float)
    risk = np.array([r["risk"] for r in res], float)
    ok = ~ab
    lcb = np.array([cp_lower(r["e"], r["k"], d_lcb) for r in res])
    return dict(cov=float(cov.mean()), cov_sd=float(cov.std(ddof=1)) if len(cov) > 1 else 0.0,
                abort=float(ab.mean()),
                risk_mean=float(np.nanmean(risk)) if ok.any() else float("nan"),
                exc_te=float((risk[ok] > alpha + 1e-12).mean()) if ok.any() else float("nan"),
                conf_viol=float((lcb[ok] > alpha).mean()) if ok.any() else float("nan"),
                rep_mass=float(np.mean([r["k_rep"] / r["n"] for r in res])),
                n_splits=len(res))


def paired_gain(a, b):
    """Mean and sd of per-split coverage difference (b - a), in points."""
    d = (np.array([r["cov"] for r in b]) - np.array([r["cov"] for r in a])) * 100
    return float(d.mean()), float(d.std(ddof=1))


# ------------------------------------------------- externally fixed gate (R1.4)
def external_gate(dev, alpha, mode=None, delta=DELTA):
    """Choose the repair gate on a DIFFERENT benchmark (`dev`), once, before the target's data are
    touched: out-of-fold scores over the whole development set, the same value grid construction, the
    same FST at full delta, and the candidate (filtering or q in mode.qs_sel) that certifies the most.
    Returns q (None = do not repair). A constant fixed in advance, so the target keeps the full delta."""
    mode = mode or Mode()
    X = dev.features(mode.missing, mode.legacy_norm)
    rn = np.random.default_rng(7)
    uniq, inv = np.unique(dev.img, return_inverse=True)
    f = (rn.permutation(len(uniq)) % mode.k_folds)[inv]; ref = np.empty(dev.n)
    for k in range(mode.k_folds):
        tk, rk = f == k, f != k
        ref[tk] = predict(fit_logistic(X[rk], dev.ok[rk]), X[tk])
    ks = k_start(alpha, delta, mode.e0)
    thr = np.quantile(ref, 1 - _grid_lams(dev.n, ks, mode.n_grid, mode.c))
    idx = np.arange(dev.n); best_q, best_k = None, -1
    for qq in [None] + sorted(mode.qs_sel):
        rg = None if qq is None else Region(dev, mode.rule, qq, idx, ref, mode, dev.m)
        j = fst_region(dev, ref, idx, thr, alpha, delta, rg)
        if j is None: continue
        kf = evaluate(dev, ref, idx, thr[j], rg)["k"]
        if kf > best_k: best_q, best_k = qq, kf
    return best_q


def external_gate_matched(dev, alpha, n_cal, reps=20, subsamples=20, seed=11, mode=None, delta=DELTA,
                          qs=(0.05, 0.10, 0.25, 0.50), X_of=None):
    """External gate at the TARGET calibration size. The full-size version of `external_gate` treats the
    whole development set as one huge calibration sample and so favours gates that only pay off with many
    items. Here the development benchmark is subsampled (by image) to 3 * n_cal items, the full protocol is
    run on image-grouped splits of each subsample, and the candidate (filtering or q) with the highest mean
    certified test coverage is kept. The target's size is known in advance and none of its data are used,
    so the result is a constant that is independent of the target calibration sample."""
    mode = mode or Mode()
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(dev.img, return_inverse=True)
    cov = {q: [] for q in [None] + list(qs)}
    for _ in range(subsamples):
        perm = rng.permutation(len(uniq)); keep = []; tot = 0
        for u in perm:
            idx = np.flatnonzero(inv == u); keep.append(idx); tot += len(idx)
            if tot >= 3 * n_cal: break
        sub = take(dev, np.concatenate(keep))
        run = Runner(sub, mode, reps=reps, seed=int(rng.integers(1e9)), X=None if X_of is None else X_of(sub))
        for q in cov:
            res = run.run(alpha, delta, gate=None if q is None else "fixed", q=q)
            cov[q].append(np.mean([r["cov"] for r in res]) if res else 0.0)
    best = max(cov, key=lambda q: (np.mean(cov[q]), q is None))
    return best
