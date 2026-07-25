# Consolidated Results — running log

Everything is real, reproducible from the scripts in this folder. Default protocol:
3-way disjoint split (train combiner / calibrate / test), fixed-sequence testing with
Clopper–Pearson bounds, δ=0.10, 20–100 repetitions. Metric key: **AURC** = area under
risk–coverage curve (lower better); **cov@α** = coverage certified at error budget α;
**risk** = realized error on held-out test data (validity audit).

Two research threads:
- **Part A — the score.** Does a structured, detection/KG-grounded nonconformity score
  beat model-internal uncertainty for conformal selective prediction? *(Yes.)*
- **Part B — the algorithm (CCRC).** Can we emit *corrected* answers under a
  distribution-free guarantee, escaping the filtering abstention bound? *(Yes, in a
  characterised regime.)* See `ALGORITHM.md` for the full write-up.

---

# PART A — grounding as the nonconformity score

## A1. Core result (POPE, LLaVA-1.5-7B, 1500 items)

| Selective score | AURC ↓ | cov@10% ↑ | AUROC ↑ |
|---|---|---|---|
| raw confidence | 0.0775 | 62.1% | 0.767 |
| learned (confidence only) | 0.0681 | 65.2% | 0.786 |
| + CLIP grounding | 0.0632 | 67.5% | 0.800 |
| + **OWLv2** grounding | **0.0546** | **73.6%** | **0.835** |
| + both | 0.0538 | 74.9% | 0.838 |

Structured detection grounding beats generic CLIP similarity (+8.4 pp vs +2.3 pp).
*(`local_analysis.py`, `local_analysis_owlv2.py`.)*

## A2. vs a faithful ConfLVLM baseline (POPE-adversarial, hardest split)

Real self-consistency (K=3 samples) as the baseline's uncertainty signal.

| Score | AURC ↓ | cov@10% ↑ |
|---|---|---|
| self-consistency only | 0.135 | 0.0% |
| confidence + self-consistency **[ConfLVLM-faithful]** | 0.077 | 61.8% |
| **+ grounding (ours)** | **0.059** | **72.0%** |

Gain **+10.2 ± 8.6 pp** coverage, ΔAURC **+0.018 ± 0.008**. *(`local_selfconsistency.py`.)*

## A3. Multi-α — grounding helps most under strict guarantees (POPE)

| Score | α=5% | α=10% | α=15% | α=20% |
|---|---|---|---|---|
| Confidence only | 30.0% | 65.2% | 84.6% | 99.1% |
| + OWLv2 grounding | **52.5%** | **73.6%** | **89.7%** | **99.6%** |

At α=5% grounding nearly **doubles** usable coverage. *(`local_multi_alpha.py`.)*

## A4. Master comparison — 5 datasets × 3 backbones/decoders

Every row valid (test error ≤ α). "conf→+g" = confidence-only → + OWLv2 grounding.

| Dataset | Backbone | acc | grnd | AURC conf→+g | cov@10% conf→+g |
|---|---|---|:--:|---|---|
| POPE (1500) | LLaVA-1.5 | 81.9% | ✓ | 0.072 → **0.058** | 63.6 → **73.1%** |
| POPE-adv | LLaVA-1.5 | 82.7% | ✓ | 0.081 → **0.060** | 46.6 → **57.4%** |
| POPE-adv | Qwen2-VL-2B | 87.3% | ✓ | 0.035 → **0.031** | 83.5 → **85.5%** |
| POPE-adv | LLaVA+VCD | 80.3% | ✓ | 0.097 → **0.060** | 21.8 → **53.9%** |
| MME (full) | LLaVA-1.5 | 68.9% | – | 0.185 | 5.7% |
| MME-existence† | LLaVA-1.5 | 95.0% | ✓ | 0.012 → **0.007** | n=60, underpowered |
| GQA (yes/no) | LLaVA-1.5 | 72.4% | – | 0.130 | 17.8% |
| HallusionBench | LLaVA-1.5 | 51.2% | – | 0.421 | 0.0% |
| AMBER(d) mixed | LLaVA-1.5 | 78.6% | partial | see Part B | — |

Grounding improves every object-existence row; largest where base confidence is poorly
calibrated (VCD, +32 pp). MME-full / GQA / HallusionBench have non-object questions
(grounding n/a) — the guarantee still holds via confidence, correctly forcing near-total
abstention on the near-chance HallusionBench. †n=60, too small to certify.
*(`master_comparison.py`.)*

## A5. Composition with a mitigation decoder (VCD, POPE-adv)

| Score | AURC ↓ | cov@10% ↑ |
|---|---|---|
| VCD confidence (mitigation alone) | 0.1030 | 47.6% |
| **VCD + grounding (ours)** | **0.0603** | **64.2%** |

ΔAURC +0.043 ± 0.007 (~6σ). Our layer composes on top of a SOTA-style mitigation
decoder regardless of that decoder's own quality.

## A6. Cross-backbone summary

| Backbone | base acc | Δcov@10% from grounding |
|---|---|---|
| LLaVA-1.5-7B | 0.827 | +10.2 pp |
| Qwen2-VL-2B | 0.873 | +1.9 pp |

The method generalizes; **the benefit scales with how much the base model hallucinates.**

---

# PART B — CCRC (certified correction)

Full write-up, theory and positioning: **`ALGORITHM.md`**. Summary:

## B1. Motivation
Prop. 3 of arXiv 2606.29054 proves any **emit-or-abstain** predictor must abstain on
≥ (μ−α)/(1−α). CCRC emits a *modified* answer, so the bound's premise fails.

## B2. Development history (each failure is documented)
| version | flaw | fix |
|---|---|---|
| v1 | certified repair with `1−s` → **circular**; repaired ~0% | certify with an **independent channel** |
| v2 | repair gate **coupled** to λ; needed δ/2 split → lost on Qwen (−4.2), VCD (−2.9) | **decouple** gate at fixed q → single FST at full δ |
| **v3** | — | canonical (`ccrc_v3.py`) |

## B3. CCRC v3 results (q=0.10, guarantee verified in every row)

| setting | n | μ | α | filter | **CCRC** | risk | gain |
|---|---|---|---|---|---|---|---|
| POPE-1500 LLaVA | 1500 | 18.1% | 0.10 | 68.2% | **72.6%** | 8.2% | +4.4 |
| POPE-adv LLaVA | 444 | 17.3% | 0.10 | 42.1% | **46.8%** | 4.3% | +4.7 |
| POPE-adv Qwen2-VL | 591 | 12.9% | 0.10 | 77.5% | **79.4%** | 6.0% | +1.9 |
| POPE-adv LLaVA+VCD | 591 | 19.6% | 0.10 | 45.0% | **47.5%** | 4.7% | +2.5 |
| POPE-adv LLaVA | 444 | 17.3% | 0.15 | 69.2% | **77.2%** | 9.7% | +8.0 |
| **AMBER(d) LLaVA** | 228 | 11.4% | 0.10 | 92.1% | 84.5% | — | **−7.5** |

## B4. Mechanism: risk dilution → acceptance headroom
Strict-gate repairs are *more* accurate than the acceptance gate's marginal items, so
they lower emitted risk and let λ open further. **Coverage gain is 3–6× the repaired
mass** (1.4% repaired → +4.7 pp coverage).

## B5. Precondition (from the AMBER negative result)
Gain tracks the abstention floor μ−α. Isolated by controlled tests: *not* a bad channel
(detector 95.6% on AMBER, 100% in the repair region) and *not* small n (POPE subsampled
to 228 still gains +3.2).

| μ−α (α=0.10) | gain |
|---|---|
| ≥ 3 pp | +1.9 … +4.7 |
| ~1.4 pp (AMBER) | −7.5 |

**Use CCRC only when μ is comfortably above α** — checkable a priori.

## B6. Baselines
**Detector-only** (emit the detector's answer): CCRC wins 3 of 4 —
POPE-1500 72.6 vs 55.3, POPE-adv 46.8 vs 13.0, Qwen 79.4 vs 21.8; AMBER **loses**
84.5 vs 92.6. Note detector accuracy ≈ VLM on POPE (82–83%) yet its selective coverage
collapses: **accuracy ≠ separable correctness score.**

**ConfLVLM scorer head-to-head** (POPE-1500, α=0.10), with honest attribution:

| method | coverage | Δ |
|---|---|---|
| ConfLVLM-style (CLIP scorer, filter) | 5.6% | — |
| + VLM confidence | 41.5% | +35.9 |
| + OWLv2 grounding | 68.2% | +26.7 |
| **CCRC (+repair)** | **72.6%** | **+4.4** |

Of +67 pp total, **+62.6 is the score and only +4.4 is our algorithm.** Caveat:
ConfLVLM was designed for free-form caption claims, so this understates them.

## B7. Ablations
- **Combiner** (`combiner_ablation.py`): deep/tree combiners do *not* help and can
  silently break validity under data reuse (GBM 13.3% error at a 10% target). With a
  proper 3-way split all combiners tie (~73%); logistic chosen deliberately.
- **Risk bound** (§9): Clopper–Pearson 72.5% > empirical-Bernstein 54.9% > Hoeffding
  38.3% coverage, all valid. Exact binomial dominates for a binary loss.

---

# Prior art / positioning (verified)

- **arXiv 2606.29054** — filtering impossibility (Prop. 3). Motivates CCRC.
- **arXiv 2606.16667 (BCEA)** — *close prior art.* Same domain/machinery (POPE,
  LLaVA/Qwen, CP+FST), same goal; **28% → 37% coverage at α=0.10.** But it *re-scores the
  model's original claim* with zoomed views and explicitly "makes no corrected answer."
  CCRC's distinct claim is **emitting a different answer**, which *requires* an
  independent channel (self-certification proved dead). **Complementary, not superseded.**
- **arXiv 2511.17908** — checked, does *not* contain conformal editing; the earlier lead
  was a synthesized search summary, not a real result.
- **ConfLVLM (EMNLP 2025)**, conformal abstention, conformal language modeling — filter or
  abstain only.
- **OPERA / REVERSE / Attention Lens** — full-coverage mitigation, no guarantee; our layer
  composes on top (A5). Dataset inventory: `baseline_datasets.md`.

## B8. BCEA race + composition (`colab_exp13_bcea.py`, `bcea_analysis.py`)

Faithful BCEA reproduction (post-acquisition score over B=3 zoomed views + blank-image
baseline). **Fair design: one common base score (VLM confidence), one mechanism added per
arm** — an earlier version gave the filter arm our grounding features while BCEA got only
its acquisition score, which understated BCEA and was not a valid comparison.

POPE / LLaVA-1.5-7B, n=394 grounded, μ=0.183. All arms valid.

| arm | cov@α=0.10 | Δ vs base | cov@α=0.15 | Δ vs base |
|---|---|---|---|---|
| BASE filter (confidence only) | 5.5% | — | 16.5% | — |
| + **BCEA** acquisition (re-score) | 5.5% | +0.1 | 24.1% | +7.5 |
| + our grounding score (filter) | 32.1% | +26.6 | 63.2% | +46.7 |
| + **CCRC** repair (ours, full) | 34.5% | +29.1 | 71.8% | +55.3 |
| **COMPOSED** (BCEA + grounding + repair) | **35.8%** | **+30.4** | **74.2%** | **+57.7** |

**Headline: the two mechanisms are complementary — composition beats every
single-mechanism arm at both α.** Re-reading the image (BCEA) and replacing the answer
from an independent channel (CCRC) rescue *different* claims, so they stack. This is the
right framing for the paper: BCEA is a collaborator, not a competitor.

Secondary: our grounding score contributes far more than BCEA's acquisition here
(+26.6/+46.7 vs +0.1/+7.5), and repair adds a consistent +2.5/+8.6 pp on top of it.

**Fidelity caveat (favours BCEA).** Our BCEA arm is likely weaker than the published
method: our zoomed views are crude (center crop + left/right halves, B=3), whereas their
acquisition may use more and better-targeted views. Their paper reports +9 pp at α=0.10;
we reproduce +7.5 pp at α=0.15 but only +0.1 pp at α=0.10. **We therefore do not claim to
beat BCEA** — only that composition helps and that our grounding score is strong in this
setup.

# Open items
2. **Sequential/generative extension** — theory drafted (`SEQUENTIAL.md`); gating
   falsification experiment (does correcting token *t* raise or lower downstream
   hallucination?) not yet run.
3. AMBER generative subset (CHAIR-style) untouched; per-category Mondrian conditional
   coverage still only on synthetic data (`conformal_sim.py`).

# File index
| file | purpose |
|---|---|
| `ALGORITHM.md` / `SEQUENTIAL.md` | CCRC write-up / sequential theory |
| `ccrc_v3.py` | **canonical CCRC** |
| `ccrc_replicate.py` | multi-backbone replication |
| `ccrc.py`, `ccrc_v2.py`, `ccrc_algorithm.py`, `ccrc_validate.py` | v1/v2 failure record |
| `conformal_sim.py` | synthetic 3-pillar validation |
| `local_analysis*.py`, `local_multi_alpha.py`, `local_selfconsistency.py`, `local_backbone_analysis.py` | Part A analyses |
| `master_comparison.py`, `combiner_ablation.py`, `risk_coverage_vs_conflvlm.py` | comparisons/ablations |
| `colab_exp*.py` | GPU extraction runs |
| `exp*.json`, `raw_scores.json`, `owlv2_scores.json` | extracted per-item signals |
| `*.png` | figures |
