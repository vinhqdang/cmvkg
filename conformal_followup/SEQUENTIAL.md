# Sequential CCRC — extending certified correction to autoregressive decoding

Status: **theory draft + simulation plan.** The one-shot guarantee (`ALGORITHM.md`) is
proved and validated. This document works out what breaks in the sequential
(generative) case and which procedure survives it.

---

## 1. What breaks

One-shot CCRC calibrates a policy on i.i.d. items and deploys it per item. Calibration
is on-policy by construction, so exchangeability holds and split conformal applies.

In generative decoding we intervene *inside* a trajectory. Two distinct failures:

**(F1) Self-induced distribution shift.** Replacing token $t$ changes the prefix, so
every later token is drawn from $p(\cdot\mid y_{<t}^{\text{edited}}, v)$ — a
distribution the model would never have produced. Calibration scores were collected
under the *un-intervened* policy. This is **not** exogenous covariate shift: the shift
is a function of our own threshold, so weighted conformal (Tibshirani et al. 2019) does
not apply. It is an instance of **feedback covariate shift** (Fannjiang et al., PNAS
2022; see also arXiv 2405.06627 for validity under agent-induced shift).

**(F2) Sequential multiplicity.** A caption yields hundreds of correlated per-token
decisions. A per-item guarantee says nothing about the emitted *sequence*, and the
number of decisions is itself data-dependent (we may stop early).

Existing work sidesteps both: ConfLVLM filters claims post-hoc; Conformal Language
Modeling (Quach et al., ICLR 2024) calibrates sampling/stopping/rejection but never
edits mid-sequence; arXiv 2606.29054 is task-level and explicitly does not address
token-level multiplicity or self-induced shift.

## 2. The procedure that survives

### 2.1 Prefix-conditional validity (kills F1)

Do not seek a marginal guarantee over trajectories. Instead make each decision valid
**conditional on the realized prefix**. With state features
$\phi_t=(\text{grounding regime},\ \text{claim type},\ \text{position bucket})$ and a
calibration pool stratified into buckets $B(\phi)$ (Mondrian conformal), define

$$p_t=\frac{1+\#\{j\in B(\phi_t)\,:\,s_j\le s_t\}}{1+|B(\phi_t)|}.$$

Under exchangeability *within a bucket*, $p_t$ is a valid conditional p-value. **How we
arrived at the prefix is irrelevant** — including that we edited it — because the
conditioning absorbs it. This is the step that makes intervention admissible.

*Cost:* buckets must be coarse enough to stay populated; conditional validity is paid
for in calibration data (the standard Mondrian trade-off).

### 2.2 E-process aggregation (kills F2)

Convert each $p_t$ into a conditionally-valid e-value, $\mathbb{E}[e_t\mid\mathcal{F}_{t-1}]\le 1$
under the null "token $t$ is grounded", and accumulate $M_T=\prod_{t\le T}e_t$. By
Ville's inequality,

$$\Pr\Big[\sup_T M_T\ge 1/\delta\Big]\le\delta .$$

The key property: an e-process is valid w.r.t. a **filtration**, and our corrections are
$\mathcal{F}_{t-1}$-predictable — they are part of the history. So conditional validity
survives adaptive intervention *and* data-dependent stopping. Split conformal cannot do
this because it needs exchangeability *across items*; e-processes need only the
conditional property. (Betting/e-CRC machinery: Waudby-Smith & Ramdas; e-CRC appears in
arXiv 2606.29054 but only at task level.)

### 2.3 Certified repair, sequential form

Reuse the one-shot design rule that we validated empirically: repair only at a strict,
**pre-specified** evidence gate $q$, certified by the *independent* channel, and require
the replacement to lie in the model's own top-$k$ (bounding distortion). Emit a hedge or
stop when neither accept nor repair certifies.

### 2.4 On-policy fixed-point calibration (residual F1)

Even with conditional validity, the *bucket populations* drift: the deployed policy
writes different text, so which $\phi$ occur (and their score distributions) changes.
Iterate to a fixed point:

$$\Theta^{(k+1)}=\mathcal{C}\big(\text{scores collected under }\pi(\Theta^{(k)})\big)$$

At a fixed point the calibration pool is on-policy, so the bucket quantiles are
consistent for the deployed policy. Convergence argument: if the map is monotone
(stricter thresholds → fewer accepted hallucinations → stricter thresholds), a
Knaster–Tarski / monotone-convergence argument applies. **Monotonicity is an assumption
we have not verified**; if it holds only approximately, the honest claim is convergence
to a neighbourhood.

## 3. Target theorem

> **Claim.** At a fixed point of §2.4, for any correction policy predictable w.r.t. the
> decoding filtration, the **emitted** sequence satisfies
> $\Pr[\text{hallucinated-claim rate} > \alpha]\le\delta$, simultaneously over all
> stopping times.

Novelty of the *target*: a guarantee on the **corrected** generation, not on a filtered
subset. Combined with Prop. 3 of arXiv 2606.29054 (filtering must abstain ≥
$(\mu-\alpha)/(1-\alpha)$), a sequential certified-correction procedure is the only
route to sequence-level certification without heavy deletion.

## 4. Where this can fail (to test before writing any proof)

1. **Leakage.** $e_t$ must be conditionally valid *given the corrected prefix*. If the
   repair policy is fit on the same fold used to calibrate buckets, it leaks. Enforce the
   disjoint-fold discipline already established (`combiner_ablation.py`).
2. **Bucket starvation.** Deep in a caption, states become rare; $|B(\phi_t)|$ collapses
   and $p_t$ loses resolution. Mitigation: coarsen with depth, or pool across positions.
3. **Non-monotone fixed point.** Correction can *increase* later hallucination (a wrong
   repair derails the caption). Then the map is not monotone and the iteration may
   oscillate. **This is the most likely failure and must be measured first.**
4. **E-value power.** $e_t=1/p_t$ is valid but weak; Kelly-style betting is tighter but
   needs a working null model per bucket.

## 4b. RESULT: monotonicity FAILS (measured, n=121)

The falsification test in §5 was run (`colab_exp14_monotonicity.py`,
`exp14_monotonicity.json`). Paired matched-prefix design: caption an AMBER image, find
the first hallucinated object mention at position t, build two prefixes identical up to
t (one keeping the hallucinated word, one substituting the best-detected `truth`
object), continue decoding equally from each, and count hallucinated mentions in the
continuations only.

| downstream hallucinated mentions | original prefix | repaired prefix |
|---|---|---|
| mean | 1.207 | **1.413** |
| paired diff (rep − orig) | **+0.207 ± 0.091** | t=2.27, p=0.025; Wilcoxon p=0.035 |
| per 100 words | 3.61 | 4.30 |
| cases | 25 better / **45 worse** / 51 tie | |

**Repairing a claim significantly INCREASES downstream hallucination.** (A pilot with
n=5 suggested the opposite; it was underpowered — p=0.62 — and is retained in the log as
a cautionary record.)

### Consequences

1. **The fixed-point / monotone-convergence argument (§2.4) is dead.** The correction map
   is not monotone, so Knaster–Tarski does not apply and the iteration has no guaranteed
   fixed point. Do not attempt that proof.
2. **Use the coupling fallback** for validity:
   risk(corrected) ≤ risk(original) + Pr(repair wrong), each term controlled separately
   with 2-D LTT. Weaker, tractable, and still gives a guarantee on a corrected output.
3. **Myopic repair is measurably harmful in sequence** — and this now *forces* a
   lookahead algorithm rather than merely motivating one. A greedy per-claim rule (CCRC
   as it stands) optimises local risk while paying an unpriced downstream externality of
   +0.21 hallucinated claims. Any correct sequential procedure must **price the
   continuation**, i.e. search over rewrite trajectories under a risk ledger
   (Conformal Trajectory Search).

### Interpretation and an important limitation

Plausible mechanism: substituting a word the model did not choose pushes the prefix
off-policy, and off-policy prefixes degrade downstream faithfulness. Note the repair
here is the *globally* best-detected `truth` object, which may be contextually
inappropriate mid-sentence ("...sitting on a **person**...") — so the honest claim is
that **naive, context-blind substitution carries a downstream cost**, not that all
correction must. A context-aware repair (constrained to the model's own top-k, as CCRC's
one-shot version requires) may reduce or remove it. That is a testable follow-up and a
second reason to prefer trajectory search, which selects repairs by their *continuation*
quality rather than their local evidence alone.

## 5. Original plan (kept for the record)

Do **not** write the proof first. Run the falsification test for failure mode 3:

> Generate captions with LLaVA under (a) no intervention and (b) CCRC-style correction
> at a strict gate. Measure whether correcting token $t$ raises or lowers the
> hallucination rate of tokens $>t$ (compare matched prefixes).

- If correction **lowers** downstream hallucination → monotonicity plausible → the
  fixed-point argument is worth formalizing.
- If it **raises** it → the fixed point may not exist; fall back to the *coupling*
  variant: bound $\text{risk}(\text{corrected}) \le \text{risk}(\text{original}) +
  \Pr[\text{repair wrong}]$ and control both terms with 2-D LTT. Weaker but tractable,
  and still yields the novel "guaranteed correction" claim.

This experiment needs caption generation (GPU) + CHAIR-style claim labelling, and is the
gating item for the sequential contribution.
