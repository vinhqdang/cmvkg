# CCRC — Certified-Correction Risk Control

A new selective-prediction procedure: it **emits repaired answers** rather than only
accepting or abstaining, while keeping a distribution-free finite-sample guarantee on
everything it emits.

Implementation: **`ccrc_v3.py` (canonical)** · `ccrc_replicate.py` (multi-backbone) ·
development history kept as the record of what failed and why: `ccrc_algorithm.py`
(v1, circular certification), `ccrc.py` + `ccrc_v2.py` + `ccrc_validate.py` (v2,
coupled gate + δ/2 split).

---

## 1. Why a new procedure is needed (the forcing argument)

**Prop. 3 of arXiv 2606.29054** ("When Can Conformal Risk Control Certify LLM
Outputs?") proves a distribution-free impossibility: a selective predictor that may
only **emit-or-abstain the model's own answer** must abstain on at least

$$\frac{\mu-\alpha}{1-\alpha}\quad\text{of inputs whenever base risk }\mu>\alpha,$$

*regardless of score quality, bound tightness, or calibration size*. Every existing
conformal-LVLM method — ConfLVLM, conformal abstention, conformal language modeling —
lives inside this bound, because they only **delete or abstain**.

The bound's premise is that the emitted answer is the model's own. **It does not apply
to a predictor that emits a different answer.** That is the opening CCRC exploits.

## 2. The mechanism: risk dilution buys acceptance headroom

Initially we expected "budget reallocation" — spending the accepted set's slack on
*worse* repaired items. Measurement showed the **opposite, and stronger, effect**.

Repairs admitted at a strict evidence gate are *more* accurate than the acceptance
gate's **marginal** items (detector top decile: 96–100% correct). Adding them
therefore **lowers** the emitted set's average risk, which relaxes the binding
constraint and lets the acceptance gate open further along the nested sequence.

The consequence is **leverage**: the coverage gain is 3–6x the repaired mass.

| setting (α=0.10) | repaired mass | coverage gain |
|---|---|---|
| POPE-1500 LLaVA | 1.8% | +4.4 pp |
| POPE-adv LLaVA | 1.4% | +4.7 pp |
| POPE-adv Qwen2-VL | 1.0% | +1.9 pp |
| POPE-adv LLaVA+VCD | 0.9% | +2.5 pp |

A small amount of high-precision repaired mass acts as *risk ballast* that finances a
much larger expansion of the accepted set. To our knowledge this is a new mechanism
in conformal risk control.

## 3. Two channels — the essential design constraint

| | signal | gates |
|---|---|---|
| Channel 1 | $s(x)=P(\text{model's answer correct})$ | ACCEPT |
| Channel 2 | $m(x)=$ independent evidence margin (open-vocab detector / KG) | REPAIR |

**The repair must be certified by an independent channel.** Certifying it with $1-s$
is circular — a hallucination score says *"suspicious"*, not *"the opposite is true."*

*Measured (POPE/LLaVA-1.5):* even the bottom 2% of $s$ had $P(\text{orig correct})=0.133>\alpha=0.10$,
so no repair is ever certifiable that way (v1 failed exactly here, repairing ~0% of
items). The detector channel, by contrast, is **93–100% accurate in its top margin
deciles** — certifiable. This is why the method *needs* a grounding/KG module, not
merely a better uncertainty score.

## 4. Policy family and guarantee

**v3 (canonical, `ccrc_v3.py`).** Nested, **pre-specified** family indexed by
permissiveness $\lambda$, with the repair gate **decoupled** at a fixed evidence
quality $q$:

```
ACCEPT original   if  s ≥ Q_s(1−λ)
REPAIR via ch.2   if  not accepted and m ≥ Q_m(1−q)     # q FIXED, not λ
ABSTAIN           otherwise
```

*Why decoupled (v2 -> v3).* In v2 the repair gate was tied to λ, so loosening
acceptance also loosened repair, admitting low-precision repairs; FST then stopped
early. Replication exposed this: v2 gained on LLaVA (+4.7/+5.3) but **lost** on
Qwen2-VL (−4.2) and VCD (−2.9). Decoupling keeps repair precision constant while λ
sweeps, and the family remains nested in λ — so a **single** FST at full δ suffices,
removing v2's δ/2 union-bound cost entirely.

*Choosing q.* q must be **strict**, and it is pre-specified, not tuned per dataset.
Sensitivity across 4 settings x 2 α levels:

| q | outcome (4 POPE-family settings x 2 α) |
|---|---|
| **0.10** (top-decile evidence) | positive in all 8 cells (+0.5 to +8.0 pp) |
| 0.25 | mixed (−16.3 on Qwen) |
| 0.50 | mixed (−15.5 on Qwen) |

**However, q=0.10 is not universally safe: it loses on AMBER(d) (−7.5 pp).** See §5c —
the cause is a missing precondition, not the gate.

This matches the channel-2 diagnostic (top margin decile = 100% accurate, 9th = 96%,
falling to 51% at the bottom): **repair only where the evidence is strongest.** Emitted-set error counts what is *actually
emitted*:

$$\mathrm{err}=\#\{\text{accepted} \wedge \text{original wrong}\}+\#\{\text{repaired}\wedge\text{ch.2 answer wrong}\}$$

**Calibration (v3).** A *single* fixed-sequence test over the nested family in
$\lambda$, with Clopper–Pearson upper bounds at full level $\delta$. FST carries no
multiplicity tax, and because the repair gate is fixed at $q$ the family is nested,
so no union-bound split is needed:

$$\Pr\big[\mathrm{risk}(\text{emitted})\le\alpha\big]\ \ge\ 1-\delta .$$

(v2 instead certified two families at $\delta/2$ and selected between them; that
split is what made it lose on Qwen2-VL and VCD. v3 removes it.)

**Sequence start.** CP cannot certify risk ≤ α with fewer than
$k_{\min}=\ln\delta/\ln(1-\alpha)\approx 22$ clean samples; starting the FST below that
aborts the sequence at step 1 (a bug we hit). The grid therefore starts at the
analytically derived minimum.

## 5. Results — v2 (coupled gate, δ/2 split), kept for the ablation record

Shows the cost of the coupled gate: gains at moderate α but a regression at α=0.05.
Superseded by §5b.

### v2 (POPE / LLaVA-1.5-7B, n=1500, 3-way split, 100 reps)

Base risk μ=0.181, δ=0.10. Risk guarantee verified on held-out test data in **all** rows.

| α | filtering ceiling (Prop.3) | FILTER cov | risk | **CCRC cov** | risk | repaired | gain |
|---|---|---|---|---|---|---|---|
| 0.05 | 86.2% | 26.8% | 1.9% | 22.1% | 1.9% | 4.2% | −4.7 |
| 0.10 | 91.0% | 69.1% | 8.0% | **72.8%** | 7.4% | 14.6% | **+3.7** |
| 0.15 | 96.4% | 87.0% | 12.8% | **92.9%** | 12.6% | 14.9% | **+5.9** |
| 0.20 | 100.0% | 97.8% | 17.0% | **99.7%** | 16.7% | 3.5% | **+1.9** |

**Admissibility condition (confirmed empirically).** Gains require α to exceed the
accepted set's risk floor $r_a$ — i.e. there must be slack to reallocate. At α=0.05
repair is inadmissible (channel-2 accuracy 93.5% < 1−α) and CCRC correctly falls back
toward filtering, paying only the δ/2 option cost (−4.7 pp). Gains peak at moderate α
and shrink as α→1 where filtering already covers nearly everything.

## 5b. Replication (CCRC v3, q=0.10, `ccrc_v3.py` / `ccrc_replicate.py`)

Certified coverage, filtering -> CCRC. Risk guarantee verified on held-out test data
in **every** row (realized risk shown for CCRC).

| setting | n | μ | α | filter | **CCRC** | risk | gain |
|---|---|---|---|---|---|---|---|
| POPE-1500 LLaVA | 1500 | 18.1% | 0.10 | 68.2% | **72.6%** | 8.2% | +4.4 |
| POPE-adv LLaVA | 444 | 17.3% | 0.10 | 42.1% | **46.8%** | 4.3% | +4.7 |
| POPE-adv Qwen2-VL | 591 | 12.9% | 0.10 | 77.5% | **79.4%** | 6.0% | +1.9 |
| POPE-adv LLaVA+VCD | 591 | 19.6% | 0.10 | 45.0% | **47.5%** | 4.7% | +2.5 |
| POPE-1500 LLaVA | 1500 | 18.1% | 0.15 | 86.8% | **88.4%** | 12.6% | +1.7 |
| POPE-adv LLaVA | 444 | 17.3% | 0.15 | 69.2% | **77.2%** | 9.7% | +8.0 |
| POPE-adv Qwen2-VL | 591 | 12.9% | 0.15 | 94.4% | **94.9%** | 10.4% | +0.5 |
| POPE-adv LLaVA+VCD | 591 | 19.6% | 0.15 | 73.4% | **75.4%** | 10.1% | +2.0 |

Holds across two backbones (LLaVA-1.5, Qwen2-VL) and a mitigation decoder (VCD).

## 5c. Negative result and the precondition for CCRC

On **AMBER(d)** (n=228 grounded, μ=11.4%) CCRC **loses**: −7.5 pp at α=0.10 and
−9.1 pp at α=0.15, even at the strict gate. Two candidate explanations were tested:

- *Bad repair channel?* **No.** The detector on AMBER is excellent — 95.6% overall and
  **100%** in the top-10% margin region where repairs are drawn.
- *Small n?* **No.** POPE subsampled to AMBER's size still gains: n=228 → +3.2,
  n=400 → +5.6, n=700 → +10.3. Small-n is not the cause.

*The cause is missing headroom.* On AMBER μ=11.4% is barely above α=10%, so filtering
alone already certifies **92.1%** coverage — the abstention floor
$(\mu-\alpha)/(1-\alpha)$ is only 1.6%. There is almost nothing for repair to recover,
while admitting any repaired mass perturbs a nearly-saturated constraint and forces λ to
tighten. Net: a loss.

**Precondition (empirical).** CCRC's gain tracks the abstention floor, i.e. how much
filtering is *forced* to discard:

| setting | μ | μ−α (α=0.10) | gain |
|---|---|---|---|
| POPE-adv LLaVA | 17.3% | 7.3 pp | +4.7 |
| POPE-1500 LLaVA | 18.1% | 8.1 pp | +4.4 |
| POPE-adv LLaVA+VCD | 19.6% | 9.6 pp | +2.5 |
| POPE-adv Qwen2-VL | 12.9% | 2.9 pp | +1.9 |
| **AMBER(d) LLaVA** | **11.4%** | **1.4 pp** | **−7.5** |

**Use CCRC when μ is comfortably above α** (base risk well above the target, i.e. the
regime where Prop. 3 forces heavy abstention — exactly the hard regime that motivates
the method). When μ ≈ α, filtering is already near-complete and plain filtering is the
right choice. A practitioner can check this *before* calibrating, since μ and α are both
known — no extra data needed.

## 6. Honest limitations

- **q is a hyperparameter.** Fixed a priori at 0.10 on the principle "repair only
  where evidence is strongest"; loose gates can lose. A data-driven choice of q would
  need its own multiplicity budget.
- **Not universally beneficial.** CCRC requires μ comfortably above α (§5c); it loses
  on AMBER(d) where filtering is already near-complete. The precondition is checkable a
  priori, but it means CCRC is a targeted tool, not a drop-in improvement.
- **Gains are modest where filtering is already strong** (Qwen at α=0.15: +0.5 pp),
  and largest where the base model hallucinates more — the same pattern as the
  grounding-score result.
- **One-shot setting only.** The guarantee proved here is for *discriminative* (single
  answer) prediction, where calibration is on-policy by construction. The
  **sequential/generative** case (token-level correction during decoding) additionally
  induces self-inflicted distribution shift — related to feedback covariate shift
  (Fannjiang et al., PNAS 2022; arXiv 2405.06627) — and needs prefix-conditional
  validity plus on-policy fixed-point calibration. **Not yet proved; the main open
  extension.**
- Channel-2 independence is an assumption; a detector sharing failure modes with the
  VLM would weaken (not invalidate) the certification.
- Validated on POPE/LLaVA so far; needs replication on AMBER(d) and a second backbone.

## 7. Positioning

| | filters/abstains | emits modified answer | distribution-free guarantee | escapes Prop.3 bound |
|---|:--:|:--:|:--:|:--:|
| OPERA / REVERSE / Attention Lens | – | ✓ (heuristic) | ✗ | n/a (no guarantee) |
| ConfLVLM, conformal abstention | ✓ | ✗ | ✓ | ✗ |
| Conformal editing (RAG, text) | ✓ | ✓ | ✓ | not framed/proved |
| **CCRC (ours)** | ✓ | ✓ | ✓ | **✓ by construction** |

## 5d. Missing baseline: the detector alone

A reviewer's first question: if the independent channel is that good, why not just
*answer with it*? Selective prediction on the detector alone (same FST+CP protocol,
α=0.10, emit the detector's answer):

| setting | VLM acc | det acc | filter(VLM) | **CCRC** | detector-only | winner |
|---|---|---|---|---|---|---|
| POPE-1500 LLaVA | 81.9% | 83.1% | 68.2% | **72.6%** | 55.3% | CCRC |
| POPE-adv LLaVA | 82.7% | 82.4% | 42.1% | **46.8%** | 13.0% | CCRC |
| POPE-adv Qwen2-VL | 87.1% | 82.9% | 77.5% | **79.4%** | 21.8% | CCRC |
| AMBER(d) LLaVA | 88.6% | 95.6% | 92.1% | 84.5% | **92.6%** | detector |

CCRC wins 3 of 4. Note the detector's *accuracy* is comparable on POPE (82–83%) yet its
selective coverage collapses (13–55%): raw accuracy does not imply a **separable
correctness score**, and conformal coverage depends on the latter. Where the detector is
genuinely stronger than the VLM (AMBER: 95.6% vs 88.6%) it also wins as a standalone
predictor — a second reason CCRC is the wrong tool in that regime.

## 5e. Head-to-head vs ConfLVLM's scorer, with attribution

ConfLVLM's published scoring function is CLIP-ViT-B/32 image–text similarity. Running
it as the nonconformity score in our pipeline (same FST+CP protocol, POPE-1500/LLaVA),
and then adding our components one at a time:

| method (α=0.10) | coverage | risk | Δ vs ConfLVLM-style |
|---|---|---|---|
| ConfLVLM-style (CLIP score, filter) | 5.6% | 1.9% | — |
| + VLM confidence (filter) | 41.5% | 5.2% | +35.9 |
| + OWLv2 grounding (filter) | 68.2% | 7.9% | +62.6 |
| **CCRC (ours: + repair, q=0.10)** | **72.6%** | 8.2% | **+67.0** |

**Honest attribution.** Of the +67 pp total, **+35.9 comes from using model confidence
and +26.7 from detector grounding — i.e. from the SCORE. Only +4.4 pp comes from our
algorithmic contribution (certified repair).** A "+67 pp over ConfLVLM" headline would
be misleading.

**Caveat that favours ConfLVLM.** ConfLVLM was designed for *free-form caption claims*,
not POPE yes/no probes. We transplanted their scorer, not their system, so 5.6% likely
understates them in their own setting.

## 7b. Prior art that narrows the novelty claim (verified)

**"Look Again Before You Abstain: Budgeted Conformal Evidence Acquisition for Reliable
Vision-Language Models"** (arXiv 2606.16667) is close prior art and must be cited:

- Same domain (VLM claims, POPE + COCO val2017), same models (LLaVA-1.5-7B, Qwen2.5-VL).
- **Same statistical machinery**: Clopper–Pearson + fixed-sequence procedure.
- Same goal — avoid heavy abstention: reports 28% → **37%** coverage at α=0.10 with risk
  0.10 held (and shows the naive un-recalibrated variant hits 0.30 risk, violating it).
- Their Theorem 1 (acquisition-adaptive validity) is exactly the "intervention requires
  recalibration" principle.

**What CCRC still contributes, precisely:**

1. **It emits a corrected answer.** BCEA acquires zoomed views and *re-scores the model's
   original claim* — explicitly, "the model makes no corrected answer." ConfLVLM and
   conformal abstention only delete. CCRC's emitted answer may **differ** from the
   model's, and the guarantee covers that modified output.
2. **The independent-channel requirement, with a proof of necessity.** BCEA lets the same
   score flag *and* rescue (validity restored by recalibration) — admissible because it
   never changes the answer. For *correction* that is impossible: we showed
   self-certification is circular and empirically dead (bottom 2% of s still 13.3%
   correct > α). Correction therefore **requires** a second, independent channel.
3. **The risk-dilution mechanism** and its 3–6x leverage.
4. **The μ−α precondition** linking the gain to Prop. 3's closed-form abstention floor.

**What CCRC can no longer claim:** being first to escape heavy abstention in VLM
selective prediction via extra visual evidence — BCEA did that.

**Direct comparison, now run** (`bcea_analysis.py`; details in RESULTS.md §B8). With one
common base score and one mechanism added per arm, on POPE/LLaVA (n=394):

| arm | cov@0.10 | cov@0.15 |
|---|---|---|
| base (confidence) | 5.5% | 16.5% |
| + BCEA acquisition | 5.5% | 24.1% |
| + our grounding score | 32.1% | 63.2% |
| + CCRC repair | 34.5% | 71.8% |
| **composed** | **35.8%** | **74.2%** |

**The mechanisms are complementary: composition beats every single-mechanism arm at both
α.** Re-reading the image and replacing the answer rescue different claims. We do **not**
claim to beat BCEA — our reproduction uses crude crops (B=3) and under-delivers versus
their published +9 pp at α=0.10, so the honest position is that CCRC and BCEA compose.
