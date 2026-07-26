"""
Canonical risk-control protocol -- ONE implementation, imported by every script.

Before this module existed, eight scripts used an exhaustive max-over-thresholds
search:

    for t in np.unique(score_cal):
        ... if cp_upper(e, k, DELTA) <= alpha and k > best: best, tau = k, t

which returns the *maximum-coverage passing threshold*. That rule has no prefix
structure, no k_min start and no multiplicity correction: it tests every one of
the O(n_cal) distinct thresholds and keeps the best, so its selection event is a
union over all of them. It is anti-conservative, and measurably so (see
CANONICAL_NUMBERS.md: population-risk exceedance 7.0-9.6% against delta=0.10,
versus 2.0-5.9% for the fixed sequence below).

The canonical procedure, as in ccrc_v3.py and missing_numbers.py:

  1. a fixed, pre-specified ASCENDING grid of lambda (permissiveness), starting
     at lam0 -- chosen so the very first hypothesis is testable, i.e. so the
     emitted count clears k_min for the bound in use (`kmin_for`);
  2. test the hypotheses in that fixed order, at FULL delta each (a fixed
     sequence needs no multiplicity correction precisely because the order is
     pre-specified and it stops at the first non-rejection);
  3. STOP at the first non-rejection and return the LAST passing index.

`lam_grid` is ccrc_v3's grid; `kmin_for` is missing_numbers' bound-specific
k_min. For Clopper-Pearson the two agree exactly:
    ub(0,k;d) = 1 - d^(1/k) <= alpha  <=>  k >= ln(delta)/ln(1-alpha),
which is Lemma 1's k_min = 22 at alpha = delta = 0.10.

Everything else here is reporting discipline that the eight scripts lacked:

  * `SPLITS`   -- paired three-way splits, so every arm sees identical folds and
                  gains can be tested pairwise.
  * `gain_stats` -- per-split sd, 95% CI for the mean gain, two-sided paired
                  t-test and Wilcoxon; see the caveat in `gain_stats.__doc__`.
  * `abort_rate` -- fraction of splits on which the fixed sequence certifies
                  nothing and the deployed system covers 0%.
  * `exceedance` -- the validity audit is Pr[realised risk > alpha] against
                  delta, NEVER mean risk. Mean risk <= alpha neither implies nor
                  is implied by the quantile guarantee.
"""
import os
import numpy as np
from scipy.stats import beta, t as tdist, wilcoxon

# Reps: the reviewer measured that 60 reps gives a Monte-Carlo SE of 0.8-4.2 pp
# on the coverage gains, so quoting a tenth of a point off 60 reps is illusory.
# Every quantity in CANONICAL_NUMBERS.md uses at least this many.
# CANON_REPS exists only to smoke-test the scripts cheaply; leave it unset for any
# number that will be reported.
REPS = int(os.environ.get("CANON_REPS", "400"))
DELTA = 0.10


# ----------------------------------------------------------------- bounds
def cp_upper(e, k, d):
    """One-sided Clopper-Pearson upper confidence limit on a binomial rate."""
    if k == 0: return 1.0
    return 1.0 if e == k else float(beta.ppf(1 - d, e + 1, k - e))


def hoeffding_upper(e, k, d):
    if k == 0: return 1.0
    return min(1.0, e / k + np.sqrt(np.log(1.0 / d) / (2.0 * k)))


def bernstein_upper(e, k, d):
    """Maurer-Pontil empirical Bernstein."""
    if k < 2: return 1.0
    p = e / k
    var = p * (1 - p) * k / (k - 1)
    return min(1.0, p + np.sqrt(2 * var * np.log(2 / d) / k)
                      + 7 * np.log(2 / d) / (3 * (k - 1)))


def betting_upper(e, k, d, grid=None):
    """Waudby-Smith & Ramdas capital-process upper confidence limit."""
    if k == 0: return 1.0
    x = np.concatenate([np.ones(e), np.zeros(k - e)])
    if grid is None: grid = np.linspace(1e-3, 1.0, 500)
    thr = np.log(1.0 / d)
    for m in grid:
        lam = min(0.5 / max(1.0 - m, 1e-3), 1.0)
        if np.sum(np.log1p(lam * (m - x))) >= thr: return float(m)
    return 1.0


_KMIN_CACHE = {}


def kmin_for(ub, alpha, delta, kmax=4000):
    """Smallest k at which the bound can certify at all: ub(0,k,delta) <= alpha.

    For Clopper-Pearson this reproduces ceil(ln delta / ln(1-alpha)); for the
    looser bounds it is much larger, which is why a fixed sequence started at
    the CP value aborts immediately when a loose bound is used.
    """
    key = (getattr(ub, "__name__", str(ub)), alpha, delta)
    if key in _KMIN_CACHE: return _KMIN_CACHE[key]
    v = kmax
    for k in range(1, kmax + 1):
        if ub(0, k, delta) <= alpha: v = k; break
    _KMIN_CACHE[key] = v
    return v


BOUNDS = [("Hoeffding", hoeffding_upper), ("Emp. Bernstein", bernstein_upper),
          ("Betting (e-CRC)", betting_upper), ("Clopper-Pearson", cp_upper)]


# ------------------------------------------------------------------ grid
GRID_N = 40


def lam_grid(n_cal, alpha, delta, ub=cp_upper, n=GRID_N):
    """ccrc_v3's grid, with missing_numbers' bound-specific k_min.

    lam0 is placed at 1.5 * k_min / n_cal so the first hypothesis is comfortably
    testable. Skipping an untestable hypothesis mid-sequence would break
    fixed-sequence FWER control, so the grid must *begin* above k_min rather
    than stepping over the early points.
    """
    k_min = kmin_for(ub, alpha, delta)
    lam0 = min(0.9, max(0.05, 1.5 * k_min / max(n_cal, 1)))
    return np.linspace(lam0, 1.0, n)


# ------------------------------------------------------------------- FST
def fst(score_cal, ok_cal, alpha, delta=DELTA, ub=cp_upper,
        rep_mask_cal=None, ok_rep_cal=None, return_index=False):
    """Canonical ascending-lambda fixed sequence.

    Stops at the FIRST non-rejection; returns the LAST passing lambda (or None
    if nothing is certified, i.e. an ABORT). `rep_mask_cal` / `ok_rep_cal` add a
    repair block, so the hypothesis tested is the MIXTURE risk over
    accepted + repaired -- which is what CCRC certifies.
    """
    k_min = kmin_for(ub, alpha, delta)
    grid = lam_grid(len(score_cal), alpha, delta, ub)
    best, best_i = None, -1
    for i, lam in enumerate(grid):
        acc = score_cal >= np.quantile(score_cal, 1 - lam)
        if rep_mask_cal is None:
            k = int(acc.sum()); e = int((1 - ok_cal[acc]).sum())
        else:
            rep = (~acc) & rep_mask_cal
            k = int(acc.sum() + rep.sum())
            e = int((1 - ok_cal[acc]).sum() + (1 - ok_rep_cal[rep]).sum())
        if k < k_min: break            # untestable; the fixed sequence must stop
        if ub(e, k, delta) <= alpha: best, best_i = float(lam), i
        else: break                    # first non-rejection: stop
    return (best, best_i) if return_index else best


# --------------------------------------------------------------- protocol
def splits(n, reps=REPS, seed=0):
    """Paired three-way disjoint splits: fit score / calibrate lambda / test.

    Deterministic in (n, reps, seed), so every arm of every comparison sees
    exactly the same folds and gains are paired.
    """
    rng = np.random.default_rng(seed)
    out = []
    t = n // 3
    for _ in range(reps):
        idx = rng.permutation(n)
        out.append((idx[:t], idx[t:2 * t], idx[2 * t:]))
    return out


SPLITS = splits


_SCORE_CACHE = {}


def fitted_scores(X, ok, reps=REPS, seed=0, max_iter=1000):
    """Logistic score fitted on each split's FIT fold, cached.

    The score depends only on (X, ok, fit fold), never on alpha, delta or the
    repair gate, so refitting it inside an alpha/q loop is pure waste -- and it is
    the dominant cost (~0.2 s per fit). Returns a list aligned with
    `splits(len(ok), reps, seed)`; entries are None where the fit fold is
    single-class and the split must be skipped.

    Cached on the CONTENT of X and ok, so two callers with the same features and
    labels share the work and are guaranteed identical scores.
    """
    import hashlib
    from sklearn.linear_model import LogisticRegression
    X = np.ascontiguousarray(np.asarray(X, float))
    ok = np.ascontiguousarray(np.asarray(ok))
    key = (hashlib.sha1(X).hexdigest(), hashlib.sha1(ok).hexdigest(),
           X.shape, reps, seed, max_iter)
    if key in _SCORE_CACHE: return _SCORE_CACHE[key]
    out = []
    for tr, cal, te in splits(len(ok), reps, seed):
        if len(np.unique(ok[tr])) < 2:
            out.append(None); continue
        out.append(LogisticRegression(max_iter=max_iter)
                   .fit(X[tr], ok[tr]).predict_proba(X)[:, 1])
    _SCORE_CACHE[key] = out
    return out


# ------------------------------------------------------------- statistics
class Gain:
    """Paired gain summary. `.line()` renders one row."""

    def __init__(self, d, name=""):
        d = np.asarray(d, float)
        self.name = name
        self.d = d
        self.n = len(d)
        self.mean = float(d.mean()) if self.n else float("nan")
        self.sd = float(d.std(ddof=1)) if self.n > 1 else float("nan")
        self.se = self.sd / np.sqrt(self.n) if self.n > 1 else float("nan")
        if self.n > 1 and self.sd > 0:
            h = float(tdist.ppf(0.975, self.n - 1)) * self.se
            self.lo, self.hi = self.mean - h, self.mean + h
            from scipy.stats import ttest_1samp
            self.p_t = float(ttest_1samp(d, 0.0).pvalue)          # two-sided
            try:
                self.p_w = float(wilcoxon(d, alternative="two-sided").pvalue)
            except ValueError:
                self.p_w = 1.0
        else:
            self.lo = self.hi = self.mean
            self.p_t = self.p_w = float("nan")

    def line(self, w=26, nd=2):
        return (f"{self.name:<{w}s}{self.mean:+8.{nd}f}{self.sd:9.{nd}f}"
                f"  [{self.lo:+7.{nd}f},{self.hi:+7.{nd}f}]{self.p_t:10.3g}{self.p_w:10.3g}"
                f"{self.n:7d}")

    @staticmethod
    def header(w=26):
        return (f"{'gain':<{w}s}{'mean':>8s}{'sd/split':>9s}"
                f"{'  95% CI (mean)':>19s}{'p(t)':>10s}{'p(Wilc)':>10s}{'reps':>7s}")


def gain_stats(base, arm, name=""):
    """Paired gain arm - base, per split.

    p-values are TWO-SIDED throughout. They test whether the mean gain over the
    *split distribution* differs from zero, conditional on this dataset. They do
    NOT license a population claim about new data: the 400 splits resample the
    same n items, so the CI shrinks like 1/sqrt(reps) while dataset-level
    uncertainty does not shrink at all. Read them as "is this gain a stable
    property of the protocol on this data", nothing more.
    """
    return Gain(np.asarray(arm, float) - np.asarray(base, float), name)


def abort_rate(lams):
    """Fraction of splits on which the fixed sequence certified nothing.

    An abort is not a missing measurement: the deployed system emits nothing and
    covers 0%, which is exactly how it enters the coverage mean.
    """
    lams = list(lams)
    return float(np.mean([l is None for l in lams])) if lams else float("nan")


def exceedance(risks, alpha):
    """Pr[realised risk > alpha] -- the validity audit.

    The guarantee is Pr[Risk <= alpha] >= 1 - delta, so the audit statistic is a
    TAIL FRACTION, not a mean. Mean risk <= alpha neither implies nor is implied
    by it. Splits that aborted contribute no emitted set and so no risk; they are
    excluded from the denominator (including them as risk 0 could only lower the
    figure). Report `abort_rate` beside this, always.
    """
    r = np.asarray([x for x in risks if x is not None], float)
    return float((r > alpha + 1e-12).mean()) if len(r) else float("nan")


def fmt_pct(x, nd=1):
    return "  n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}%"
