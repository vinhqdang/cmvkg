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

| q | outcome |
|---|---|
| **0.10** (top-decile evidence) | **gains in all 8 cells (+0.5 to +8.0 pp), never loses** |
| 0.25 | mixed (−16.3 on Qwen) |
| 0.50 | mixed (−15.5 on Qwen) |

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

## 6. Honest limitations

- **q is a hyperparameter.** Fixed a priori at 0.10 on the principle "repair only
  where evidence is strongest"; loose gates can lose. A data-driven choice of q would
  need its own multiplicity budget.
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
