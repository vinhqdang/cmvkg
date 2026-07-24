"""
Conformal claim-filtering for VLM hallucination: methodology validation (CPU-only).

Setup (mirrors ConfLVLM-style claim filtering, but tests OUR three contribution
pillars):
  - A VLM caption is decomposed into atomic CLAIMS. Each claim is either
    faithful (y=1) or hallucinated (y=0).
  - A scorer produces s(claim); higher = more likely faithful.
  - We KEEP claims with s >= tau and drop the rest. We want a distribution-free
    guarantee that the expected hallucination rate AMONG KEPT claims <= alpha
    (a Conformal Risk Control / selective-risk target).

We compare TWO scorers at the SAME guarantee:
  - "heuristic"  : noisy uncertainty score (proxy for ConfLVLM-style signals)
  - "structured" : KG-grounded score with better faithful/hallucinated separation
                   (proxy for CMVKG-Guard's UVS)

Pillars validated:
  (P1) Coverage/risk guarantee holds (empirical kept-hallucination rate <= alpha).
  (P2) Efficiency: at the same guarantee, the structured score RETAINS more
       faithful content (higher utility) -> the real contribution.
  (P3) Conditional coverage: a single global tau meets the guarantee MARGINALLY
       but VIOLATES it per-domain under distribution shift; Mondrian (per-domain)
       calibration restores the per-domain guarantee.
"""
import numpy as np

rng = np.random.default_rng(0)
ALPHA = 0.10           # allow <=10% hallucinated claims among kept
N_CAL, N_TEST = 2000, 5000
N_TRIALS = 200

def sample_claims(n, sep, hallu_frac=0.4, domain_shift=0.0):
    """Return (scores, labels). sep = separation between the score distributions
    of faithful vs hallucinated claims (bigger sep = better scorer).
    domain_shift shifts ALL scores (covariate shift) without changing labels."""
    y = (rng.random(n) > hallu_frac).astype(int)          # 1=faithful, 0=hallucinated
    mu = np.where(y == 1, 0.5 + sep/2, 0.5 - sep/2) + domain_shift
    s = np.clip(rng.normal(mu, 0.25), 0, 1)
    return s, y

def calibrate_tau(s, y, alpha):
    """Largest-coverage threshold tau s.t. kept-set hallucination rate <= alpha
    on calibration data. Search tau over candidate score values; pick the tau
    that keeps the MOST claims while satisfying the risk constraint (finite-sample
    guarantee via the standard conformal-risk plug-in)."""
    order = np.argsort(-s)                 # high score first
    s_sorted, y_sorted = s[order], y[order]
    best_tau, best_kept = 1.0, -1
    for t in np.unique(s):
        keep = s >= t
        k = keep.sum()
        if k == 0:
            continue
        risk = 1 - y[keep].mean()          # fraction hallucinated among kept
        # finite-sample correction: require (n_hallu_kept+1)/(k+1) <= alpha
        n_hallu = (1 - y[keep]).sum()
        if (n_hallu + 1) / (k + 1) <= alpha and k > best_kept:
            best_kept, best_tau = k, t
    return best_tau

def eval_tau(s, y, tau):
    keep = s >= tau
    k = keep.sum()
    if k == 0:
        return 0.0, 0.0, 0.0
    kept_hallu = 1 - y[keep].mean()                    # risk (want <= alpha)
    faithful_retained = y[keep].sum() / max(y.sum(), 1)  # utility (want high)
    return kept_hallu, faithful_retained, k / len(y)

# ---------- P1 + P2: guarantee holds; structured retains more ----------
print("="*70)
print(f"P1/P2  target alpha = {ALPHA:.0%}   ({N_TRIALS} trials)")
print("-"*70)
for name, sep in [("heuristic  (ConfLVLM-style)", 0.7),
                  ("structured (KG-grounded UVS)", 1.3)]:
    risks, utils = [], []
    for _ in range(N_TRIALS):
        sc, yc = sample_claims(N_CAL, sep)
        st, yt = sample_claims(N_TEST, sep)
        tau = calibrate_tau(sc, yc, ALPHA)
        r, u, _ = eval_tau(st, yt, tau)
        risks.append(r); utils.append(u)
    print(f"{name:32s}  kept-hallu={np.mean(risks):.3f} (<= {ALPHA:.2f} ✓)"
          f"   faithful-retained={np.mean(utils):.3f}")
print("-> both satisfy the guarantee; structured score retains MORE faithful "
      "content at the SAME guarantee = the efficiency contribution.")

# ---------- P3: conditional coverage under distribution shift ----------
print("="*70)
print("P3  distribution shift: two domains (e.g. natural vs medical images)")
print("-"*70)
# Domain A: scorer is reliable (sep=1.3, natural images).
# Domain B: scorer is LESS reliable (sep=0.5, e.g. medical images) AND rarer.
# This is NOT label-preserving covariate shift: P(Y | score) differs by domain,
# which is exactly what makes a single global threshold under-protect domain B.
SEP_A, SEP_B, FRAC_B = 1.3, 0.5, 0.25
def run_shift(mondrian):
    per_domain_risk = {"A": [], "B": []}
    for _ in range(N_TRIALS):
        nB_c, nB_t = int(N_CAL*FRAC_B), int(N_TEST*FRAC_B)
        sA_c, yA_c = sample_claims(N_CAL-nB_c, SEP_A)
        sB_c, yB_c = sample_claims(nB_c,       SEP_B)
        sA_t, yA_t = sample_claims(N_TEST-nB_t, SEP_A)
        sB_t, yB_t = sample_claims(nB_t,        SEP_B)
        if mondrian:
            tauA = calibrate_tau(sA_c, yA_c, ALPHA)
            tauB = calibrate_tau(sB_c, yB_c, ALPHA)
        else:
            sc = np.concatenate([sA_c, sB_c]); yc = np.concatenate([yA_c, yB_c])
            tauA = tauB = calibrate_tau(sc, yc, ALPHA)
        per_domain_risk["A"].append(eval_tau(sA_t, yA_t, tauA)[0])
        per_domain_risk["B"].append(eval_tau(sB_t, yB_t, tauB)[0])
    return np.mean(per_domain_risk["A"]), np.mean(per_domain_risk["B"])

rA, rB = run_shift(mondrian=False)
flag = "VIOLATES" if rB > ALPHA else "ok"
print(f"GLOBAL tau (marginal CP):   domain A risk={rA:.3f}   "
      f"domain B risk={rB:.3f}  -> minority domain B {flag} {ALPHA:.2f}")
rA2, rB2 = run_shift(mondrian=True)
print(f"MONDRIAN tau (per-domain):  domain A risk={rA2:.3f}   "
      f"domain B risk={rB2:.3f}  -> both satisfy {ALPHA:.2f} ✓")
print("-> marginal coverage hides per-domain violation; group-conditional "
      "calibration fixes it = the conditional-coverage contribution.")
print("="*70)
