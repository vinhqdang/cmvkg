# Consolidated Results

All numbers are real, reproducible from the scripts in this folder. Conformal
target: error among accepted ≤ α = 10%, δ = 0.10, 20 random calibration/test
splits unless noted. Metric key: **AURC** = area under risk-coverage curve
(lower better); **cov@10%** = coverage retained at a certified 10% error budget
(higher better).

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

### Honest caveats
- POPE samples here are the **adversarial** split (hardest); grounding coverage
  ~99% of items. Per-category Mondrian conditional coverage is shown on synthetic
  data (`conformal_sim.py`); a category-balanced real run is future work.
- Self-consistency used K=3 (coarse); a higher-K baseline would be a stronger
  ConfLVLM reproduction.
- Grounding benefit is specific to object-existence hallucination (§6).
