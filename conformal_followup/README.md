# Conformal Selective Prediction for VLM Hallucination (follow-up study)

A research follow-up to **CMVKG-Guard** exploring **conformal prediction** for
hallucination control in Vision-Language Models. Instead of a heuristic
verification threshold, we ask: can we give the accept/abstain (or accept/correct)
decision a **distribution-free, finite-sample statistical guarantee**, and does a
**structured grounding signal** (as in CMVKG-Guard's UVS) make the resulting
selective predictor *more efficient* than model self-confidence alone?

## Thesis

1. **Validity** — wrap the decision in a risk-controlling procedure (RCPS /
   split-conformal with a Clopper-Pearson bound) so the error rate among
   *accepted* outputs is provably ≤ α.
2. **Efficiency** — the *nonconformity score* is what matters once coverage is
   guaranteed. A structured, visually-grounded score should retain more content
   (higher coverage / lower AURC) at the same guarantee than self-confidence.
3. **Conditional coverage** — marginal guarantees hide per-domain failures under
   distribution shift; group-conditional (Mondrian) calibration restores them.

## Relation to prior work (honest positioning)

The generic idea "conformal prediction for VLM hallucination" is **not new** —
see ConfLVLM (EMNLP 2025, claim-level conformal factuality), Inductive Conformal
Prediction for LVLMs (arXiv 2504.17671), Conformal Abstention (NeurIPS 2024),
Proof-of-Perception (arXiv 2603.00324). The intended contribution here is
narrower and buildable on our accepted system: **(a)** a structured KG-grounded
nonconformity score vs heuristic uncertainty, measured by selective-prediction
*efficiency*; **(b)** a guarantee on the *corrected* output during decoding, not
just post-hoc filtering; **(c)** group-conditional coverage under visual shift.

## Contents

| File | What it does | Compute |
|---|---|---|
| `conformal_sim.py` | Synthetic validation of all three pillars (P1 validity, P2 efficiency, P3 conditional coverage) | CPU |
| `colab_exp3.py` | Real experiment: LLaVA-1.5-7B (4-bit) on POPE, extracts VLM yes/no probability + CLIP grounding per item. Run via `colab run --gpu T4 --timeout 3000 colab_exp3.py 1500` | GPU (T4) |
| `raw_scores.json` | Extracted per-item signals for 1500 POPE probes (p_yes, CLIP grounding, answer, gold, correctness) — lets all downstream analysis run on CPU | — |
| `local_analysis.py` | Learned-combiner analysis over 20 splits: isolates grounding's contribution to conformal selective prediction | CPU |

## Results (real data, POPE, LLaVA-1.5-7B, 20 random splits)

VLM raw POPE accuracy = 0.819. Target risk α = 0.10, δ = 0.10. Error among
accepted stayed ≤ α (valid) for every row.

| Selective score | AURC ↓ | Coverage@10% ↑ | AUROC ↑ |
|---|---|---|---|
| raw confidence (baseline) | 0.0775 | 62.1% | 0.767 |
| learned (confidence only) | 0.0681 | 65.2% | 0.786 |
| learned + CLIP grounding | 0.0632 | 67.5% | 0.800 |
| learned + **OWLv2** grounding | **0.0546** | **73.6%** | **0.835** |
| learned + both | 0.0538 | 74.9% | 0.838 |

**Paired gains over confidence-only (same splits):**

| Grounding signal | ΔCoverage@10% | ΔAURC |
|---|---|---|
| CLIP-B/32 (weak, full-image similarity) | +2.3 ± 1.6 pp | +0.0049 ± 0.0017 |
| **OWLv2 (detection-based)** | **+8.4 ± 3.4 pp** | **+0.0135 ± 0.0014** |
| both | +9.8 ± 2.9 pp | +0.0143 ± 0.0023 |

**Finding.** The nonconformity *score* is the crux. A structured, detection-based
grounding signal (OWLv2 — the analogue of CMVKG-Guard's L1) lets us confidently
answer **73.6%** of questions vs **65.2%** on model confidence alone, at the same
≤10% error guarantee — a ~13% relative jump in usable coverage. Generic CLIP
similarity gives only a small gain; the *quality* of the grounding signal is what
converts into conformal efficiency.

### Honest caveats
- A **naive min-max fusion made coverage *worse***; the gain only appears with a
  **learned combiner**. Fusion design matters.
- Single benchmark (POPE), single backbone (LLaVA-1.5-7B). A validated core, not
  yet a paper — needs a second benchmark and risk-coverage curves across α.
- OWLv2 is a strong stand-in for CMVKG-Guard's detection layer; wiring in the full
  UVS (detection + KG + reasoning) is the next step.

Reproduce the grounding comparison with `python local_analysis_owlv2.py`
(reads `raw_scores.json` + `owlv2_scores.json`, CPU-only).

## Reproduce

```bash
# 1. (GPU) extract signals — needs an authenticated google-colab-cli
colab run --gpu T4 --timeout 3000 colab_exp3.py 1500   # writes RAW_JSON to stdout
# 2. (CPU) synthetic pillars
python conformal_sim.py
# 3. (CPU) main analysis (reads raw_scores.json)
python local_analysis.py
```
