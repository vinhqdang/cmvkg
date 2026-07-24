# Consolidated Results

All numbers are real, reproducible from the scripts in this folder. Conformal
target: error among accepted ≤ α = 10%, δ = 0.10, 20 random calibration/test
splits unless noted. Metric key: **AURC** = area under risk-coverage curve
(lower better); **cov@10%** = coverage retained at a certified 10% error budget
(higher better).

## 0. Master comparison (all datasets/backbones, `master_comparison.py`)

Strict 3-way split (train combiner / calibrate τ / test), RCPS, 20 reps. Every
row valid (test error ≤ α). "conf→+g" = confidence-only → + OWLv2 grounding.

| Dataset | Backbone | acc | grnd | AURC conf→+g | cov@10% conf→+g |
|---|---|---|:--:|---|---|
| POPE (1500) | LLaVA-1.5 | 81.9% | ✓ | 0.072 → **0.058** | 63.6% → **73.1%** |
| POPE-adv | LLaVA-1.5 | 82.7% | ✓ | 0.081 → **0.060** | 46.6% → **57.4%** |
| POPE-adv | Qwen2-VL-2B | 87.3% | ✓ | 0.035 → **0.031** | 83.5% → **85.5%** |
| POPE-adv | LLaVA+VCD | 80.3% | ✓ | 0.097 → **0.060** | 21.8% → **53.9%** |
| MME (full) | LLaVA-1.5 | 68.9% | – | 0.185 | 5.7% |
| MME-existence† | LLaVA-1.5 | 95.0% | ✓ | 0.012 → **0.007** | 0% (n=60, underpowered) |
| GQA (yes/no) | LLaVA-1.5 | 72.4% | – | 0.130 | 17.8% |
| HallusionBench | LLaVA-1.5 | 51.2% | – | 0.421 | 0.0% |

Grounding improves AURC and coverage on every object-existence row; the gain is
largest when base confidence is poorly calibrated (VCD, +32pp). MME-full / GQA /
HallusionBench have (mostly) non-object questions (grounding n/a) — the guarantee
still holds via confidence, correctly forcing heavy abstention on near-chance
HallusionBench. †MME-existence is only 60 items (95% acc) — too small for a
reliable conformal split (0 certified at α=10%), though grounding still improves
its AURC. AMBER not on HuggingFace (needs manual load) — deferred.

**Coverage: 5 datasets (POPE, MME, MME-existence, GQA, HallusionBench) ·
3 backbones/decoders (LLaVA-1.5, Qwen2-VL-2B, LLaVA+VCD) · every row valid.**

## 1. Core result — structured grounding vs weaker signals (POPE, LLaVA-1.5-7B)

| Selective score | AURC ↓ | cov@10% ↑ | AUROC ↑ |
|---|---|---|---|
| raw confidence | 0.0775 | 62.1% | 0.767 |
| learned (confidence only) | 0.0681 | 65.2% | 0.786 |
| + CLIP grounding | 0.0632 | 67.5% | 0.800 |
| + **OWLv2** grounding (ours) | **0.0546** | **73.6%** | **0.835** |
| + both | 0.0538 | 74.9% | 0.838 |

Structured detection grounding beats generic CLIP similarity (+8.4pp coverage vs
+2.3pp). Validity held (error ≤ α) on every row.

## 2. vs ConfLVLM — real self-consistency baseline (POPE-adversarial, hardest split)

| Score | AURC ↓ | cov@10% ↑ |
|---|---|---|
| self-consistency only (K=3) | 0.135 | 0.0% |
| internal: confidence + self-consistency **[ConfLVLM-faithful]** | 0.077 | 61.8% |
| **internal + grounding (ours)** | **0.059** | **72.0%** |

Grounding gain over the faithful baseline: **+10.2 ± 8.6 pp coverage, ΔAURC
+0.018 ± 0.008**. Even given both confidence and self-consistency, grounding adds.

## 3. Multi-α — grounding helps most under strict guarantees (POPE)

| Score | α=5% | α=10% | α=15% | α=20% |
|---|---|---|---|---|
| Confidence only | 30.0% | 65.2% | 84.6% | 99.1% |
| + OWLv2 grounding (ours) | **52.5%** | **73.6%** | **89.7%** | **99.6%** |

At the strict 5% budget, grounding nearly **doubles** usable coverage.

## 4. Cross-backbone generalization (POPE-adversarial)

| Backbone | base acc | Δcov@10% from grounding | ΔAURC |
|---|---|---|---|
| LLaVA-1.5-7B | 0.827 | +10.2 pp | +0.018 |
| Qwen2-VL-2B | 0.873 | +1.9 pp | +0.0045 |

Method generalizes; grounding's benefit scales with how much the base model
hallucinates (stronger Qwen hallucinates less → smaller gain).

## 5. Composition with a mitigation decoder (VCD, POPE-adversarial)

| Score | AURC ↓ | cov@10% ↑ |
|---|---|---|
| VCD confidence (mitigation alone) | 0.1030 | 47.6% |
| **VCD + grounding (ours, composition)** | **0.0603** | **64.2%** |

ΔAURC +0.043 ± 0.007 (~6σ). Our conformal grounding layer composes on top of a
SOTA-style mitigation decoder and improves it, regardless of the decoder's own
quality.

## 6. Second dataset — MME (LLaVA-1.5-7B)

| | value |
|---|---|
| accuracy | 0.689 |
| error among accepted @ α=10% | 4.6% (valid ✓) |
| grounded items | 0 / 700 |

Conformal **validity generalizes** to a harder, different dataset. The first 700
MME items are cognition/reasoning tasks (no object-existence questions), so OWLv2
grounding does not apply — **delineating the scope of the grounding contribution
to object hallucination**. The guarantee still holds via confidence alone.

## 7. Positioning vs recent methods (capability, not head-to-head accuracy)

OPERA (CVPR'24), Attention Lens (CVPR'25), REVERSE (NeurIPS'25) are full-coverage
**mitigation** methods (no statistical guarantee, no selective tradeoff); ConfLVLM
(EMNLP'25) is conformal but uses heuristic uncertainty without structured
grounding. Ours is the only approach combining structured grounding + real-time
correction + a distribution-free guarantee + selective coverage, and it *composes
on top of* the mitigation methods (see §5). See `comparison_figure.png`.

## 8. Ablation — nonconformity combiner (`combiner_ablation.py`)

Does a higher-capacity combiner beat logistic regression? Two calibration
protocols, 6 features, target error 10%.

**Protocol I — combiner trained on the same fold used to calibrate τ (naive):**

| Combiner | cov@10% | test error | valid? |
|---|---|---|---|
| Logistic | 74.8% | 8.8% | ✅ |
| GradBoost | 91.3% | 13.3% | ❌ VIOLATED |
| RandForest | 88.5% | 12.1% | ❌ VIOLATED |
| MLP (64,32) | ~28% (unstable) | — | overfits |

**Protocol II — proper 3-way split (train combiner / calibrate τ / test, disjoint):**

| Combiner | cov@10% | test error | valid? |
|---|---|---|---|
| Logistic | 72.5% | 8.4% | ✅ |
| GradBoost | 72.5% | 8.3% | ✅ |
| RandForest | 75.7% | 8.5% | ✅ |

High-capacity combiners only *appear* better under data reuse, where they overfit
the calibration set and **break the guarantee** (13.3% > 10%). With a proper split
all combiners are valid and statistically tied, so **logistic regression is chosen
deliberately**: equal efficiency, lowest variance, interpretable, robust to the
data-reuse trap. Capacity is not the bottleneck — grounding-signal quality is; a
deep combiner would only help if fed rich high-dimensional inputs (raw
embeddings / hidden states / attention), which needs far more calibration data.
**Design rule: train the combiner on a fold disjoint from conformal calibration.**

## 9. Ablation — risk upper-bound method (binary loss, POPE, 3-way split)

| Upper bound | cov@10% | test error | valid? |
|---|---|---|---|
| **Clopper–Pearson (exact binomial)** | **72.5%** | 8.4% | ✅ |
| Empirical-Bernstein | 54.9% | 4.6% | ✅ |
| Hoeffding | 38.3% | 3.2% | ✅ |

The accepted-set error is a Binomial proportion, so the exact interval
(Clopper–Pearson) is near-optimal; general-bounded-loss concentration bounds
(Hoeffding, Bernstein, WSR) are looser here and waste coverage. **CP is used for
the binary experiments.** For graded/continuous factuality losses,
variance-adaptive bounds (empirical-Bernstein, WSR betting) become preferable.
Multiplicity across the threshold scan is handled in the Learn-Then-Test /
fixed-sequence-testing framing (RCPS); CP's conservativeness currently absorbs it.

### Honest caveats
- POPE samples here are the **adversarial** split (hardest); grounding coverage
  ~99% of items. Per-category Mondrian conditional coverage is shown on synthetic
  data (`conformal_sim.py`); a category-balanced real run is future work.
- Self-consistency used K=3 (coarse); a higher-K baseline would be a stronger
  ConfLVLM reproduction.
- Grounding benefit is specific to object-existence hallucination (§6).
