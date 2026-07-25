# CCRC — Certified-Correction Risk Control

A new selective-prediction procedure: it **emits repaired answers** rather than only
accepting or abstaining, while keeping a distribution-free finite-sample guarantee on
everything it emits.

Implementation: `ccrc.py` (canonical) · development history: `ccrc_algorithm.py` (v1,
failed), `ccrc_v2.py`, `ccrc_validate.py`.

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

## 2. The mechanism: risk-budget reallocation

A calibrated filtering policy **systematically underspends its budget**: it emits a
high-precision subset whose realized risk sits well below α (measured: **8.0% at
α=10%**). CCRC converts that unused slack into coverage by admitting **repaired**
items whose individual error exceeds the accepted set's, but whose blended risk stays
within budget:

$$\frac{c_a r_a + c_r r_r}{c_a+c_r}\le\alpha ,\qquad r_r>\alpha>r_a$$

Slack on the accepted set *finances* repairs. This is, to our knowledge, a new
mechanism in conformal risk control.

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

Nested, **pre-specified** family indexed by permissiveness $\lambda$:

```
ACCEPT original   if  s ≥ Q_s(1−λ)
REPAIR via ch.2   if  not accepted and m ≥ Q_m(1−λ)     [family F_rep]
ABSTAIN           otherwise
```
`F_filt` is the same with repair disabled. Emitted-set error counts what is *actually
emitted*:

$$\mathrm{err}=\#\{\text{accepted} \wedge \text{original wrong}\}+\#\{\text{repaired}\wedge\text{ch.2 answer wrong}\}$$

**Calibration.** Fixed-sequence testing (FST) over each nested family — no multiplicity
tax — with Clopper–Pearson upper bounds; each family at level $\delta/2$; deploy
whichever certifies more coverage. By the union bound:

$$\Pr\big[\mathrm{risk}(\text{emitted})\le\alpha\big]\ \ge\ 1-\delta .$$

Adaptive family selection is what keeps CCRC from being worse than filtering when
repair is inadmissible.

**Sequence start.** CP cannot certify risk ≤ α with fewer than
$k_{\min}=\ln\delta/\ln(1-\alpha)\approx 22$ clean samples; starting the FST below that
aborts the sequence at step 1 (a bug we hit). The grid therefore starts at the
analytically derived minimum.

## 5. Results (POPE / LLaVA-1.5-7B, n=1500, 3-way split, 100 reps)

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

## 6. Honest limitations

- **Strict-α regression.** At α=0.05 the δ/2 union-bound cost is not recovered.
  A single pre-specified sequence spanning both families would remove the split but
  requires a data-independent ordering we have not yet constructed.
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
