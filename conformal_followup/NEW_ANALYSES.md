# Three new CPU-only analyses for the manuscript revision

All three run from the already-extracted per-item signals; no GPU, no new inference.
Every arm uses the canonical CCRC protocol imported directly from `ccrc_v3.py`: 3-way
disjoint split (fit score / calibrate λ / test), ascending-λ fixed-sequence test stopping
at the first non-rejection, Clopper–Pearson upper bounds, δ = 0.10, logistic-regression
correctness score on the same four features as `ccrc_v3.build()`.

| file | task |
|---|---|
| `sym_gate.py` | Task 1 — the polarity-symmetric repair gate |
| `power_seq.py` | Task 2 — power / equivalence for the sequential experiment |
| `qsel_transfer.py` | Task 3 — pricing the selection of the repair-gate quantile *q* |

Reproduce everything:

```bash
cd /home/user/cmvkg/conformal_followup
python3 sym_gate.py --reps 500          # Task 1  (~20 min, 500 paired splits)
python3 sym_gate.py --diag-only         # Task 1  structural diagnostic + nesting check (~1 min)
python3 power_seq.py                    # Task 2  (~5 s)
python3 qsel_transfer.py --reps 500     # Task 3  (~25 min, 500 paired splits)
```

Nothing was committed. No existing file was modified.

### Reproduction fidelity, and an incidental finding about the published gains

The `one-dir` arm below is a re-implementation of `ccrc_v3.py`'s q=0.10 policy that shares
splits with the new arms so contrasts can be paired. It reproduces the canonical procedure —
but at **500** splits rather than the 60 in `ccrc_v3.py`, and the published gains shrink:

| setting | α | `ccrc_v3.py`, 60 splits | this file, 500 splits | 95% CI at 500 splits |
|---|---|---|---|---|
| POPE-1500 LLaVA | 0.10 | 68.2 → 72.6 (+4.4) | 68.37 → 71.71 (**+3.34**) | [+2.72, +3.96] |
| POPE-adv LLaVA | 0.10 | 42.1 → 46.8 (+4.7) | 43.58 → 46.76 (**+3.18**) | [+1.85, +4.51] |
| POPE-adv Qwen2-VL | 0.10 | 77.5 → 79.4 (+1.9) | 77.79 → 79.10 (**+1.31**) | [+0.45, +2.17] |
| POPE-adv LLaVA+VCD | 0.10 | 45.0 → 47.5 (+2.5) | 48.31 → 50.31 (**+2.01**) | [+1.17, +2.85] |
| AMBER(d) LLaVA | 0.10 | 92.1 → 84.5 (−7.5) | 91.54 → 81.49 (**−10.05**) | [−12.87, −7.23] |
| POPE-1500 LLaVA | 0.15 | 86.8 → 88.4 (+1.7) | 86.55 → 88.48 (**+1.93**) | [+1.82, +2.05] |
| POPE-adv LLaVA | 0.15 | 69.2 → 77.2 (+8.0) | 69.01 → 76.87 (**+7.86**) | [+5.97, +9.76] |
| POPE-adv Qwen2-VL | 0.15 | 94.4 → 94.9 (+0.5) | 94.69 → 95.09 (**+0.40**) | [+0.34, +0.46] |
| POPE-adv LLaVA+VCD | 0.15 | 73.4 → 75.4 (+2.0) | 75.43 → 76.72 (**+1.29**) | [+0.57, +2.00] |
| AMBER(d) LLaVA | 0.15 | 95.7 → 86.6 (−9.1) | 95.86 → 83.90 (**−11.96**) | [−14.81, −9.11] |

Equivalence was checked directly: run at `--reps 60`, `sym_gate.py`'s `filter` and `onedir`
columns reproduce `ccrc_v3.py`'s `filter` and `q=0.1` columns to the printed precision
(Qwen2-VL 77.5/79.4, VCD 45.0/47.5, AMBER 92.1/84.5). The differences below are Monte-Carlo
over splits, not a difference in procedure.

Every α=0.10 gain in Table 5b/B3 is 0.5–1.5 pp **higher** at 60 splits than at 500, and both
AMBER losses are 2.5–2.9 pp **smaller**. Nothing is wrong with the code; 60 splits is simply
not enough to pin a mean gain of a few pp. The revision should re-run the headline table at
≥ 400 splits and attach the intervals — the qualitative conclusions all survive, but "+4.4"
should read "+3.3 [+2.7, +4.0]".

**Caveat on all CIs and p-values in this document.** They are computed over repeated
**splits of a fixed dataset**, so they are conditional on the data: they quantify how stable
a contrast is under the split randomisation, not how it would vary over fresh draws of POPE
or AMBER. They are the right statistic for "is arm A better than arm B on this data", and the
wrong statistic for "does arm A generalise". With 500 highly dependent splits the CIs are
narrow and many p-values are astronomically small; the split-to-split standard deviation
(`sd` column in the script output, e.g. 7.08 pp for the POPE-1500 α=0.10 gain) is the number
that conveys realistic spread.

---

## TASK 1 — The polarity-symmetric repair gate

### What was built

The current margin `m = |o − 0.15|` gives the two polarities unequal dynamic range: a
"present" claim can reach `m = 0.85`, an "absent" claim can never exceed `m = 0.15`, and
the measured q=0.10 gate sits at **0.383 / 0.357 / 0.366 / 0.366 / 0.209** — above the
entire negative-side range in four of five settings. Hence 100% positive-polarity repairs.

Two symmetric constructions, both with **q held FIXED at exactly the pre-specified 0.10**
(only λ is calibrated, as in v3):

```
mp = (o − THR) / (1 − THR)      normalised margin for a "present" claim, in [0,1]
mn = (THR − o) / THR            normalised margin for an "absent"  claim, in [0,1]

sym-2gate   two pre-specified gates, one per polarity, each at the same fixed q:
            t_pos = Q_{1−q}( mp | cal, o ≥ THR )     t_neg = Q_{1−q}( mn | cal, o < THR )
            REPAIR if (o ≥ THR and mp ≥ t_pos) or (o < THR and mn ≥ t_neg)

sym-norm    single gate on the two-sided normalised margin m_sym = mp if o≥THR else mn
            REPAIR if m_sym ≥ Q_{1−q}( m_sym | cal )
```

Plus two decomposition arms that isolate *why* anything changes — `pos-2gate` (positive
branch only, which isolates the re-scaling of the positive gate) and `neg-2gate` (negative
branch only, the new capability by itself).

### Validity: nesting is preserved, no δ split needed

In every arm the repair mask `M` depends only on the calibration fold and on `o` — never
on λ. So `E_λ = A_λ ∪ (M \ A_λ) = A_λ ∪ M` is a λ-nested increasing family (union of a
nested increasing family with a λ-constant set). A **single** ascending fixed-sequence test
at **full δ = 0.10** is therefore valid, exactly as for the one-directional gate; no union
bound and no δ/2 split is introduced. Verified numerically over a λ-grid on 30 random
splits per setting (`python3 sym_gate.py --diag-only`): **all arms OK, zero violations.**

A budget note that matters for interpretation: `sym-2gate` admits the top-q of *each*
polarity, so the total gate mass is still ≈ q of items. It **redistributes** a fixed repair
budget across polarities; it does not enlarge it.

### Results (500 paired splits, δ = 0.10; all arms share splits and the fitted score)

Certified coverage %, gain in pp over filtering with 95% CI and two-sided paired-t p.

| setting | α | filtering | one-dir CCRC | **sym-2gate** | sym vs one-dir | p |
|---|---|---|---|---|---|---|
| POPE-1500 LLaVA | 0.10 | 68.37 | **71.71** (+3.34 [+2.72,+3.96], p=1.3e-23) | 70.60 (+2.23 [+1.62,+2.85], p=4.0e-12) | **−1.10** [−1.19,−1.02] | 2.3e-93 |
| POPE-1500 LLaVA | 0.15 | 86.55 | **88.48** (+1.93 [+1.82,+2.05]) | 87.23 (+0.68 [+0.61,+0.74]) | **−1.26** [−1.36,−1.15] | 1.0e-86 |
| POPE-adv LLaVA | 0.10 | 43.58 | 46.76 (+3.18 [+1.85,+4.51], p=3.3e-06) | **51.37** (+7.79 [+6.17,+9.42], p=1.7e-19) | **+4.61** [+2.77,+6.45] | 1.2e-06 |
| POPE-adv LLaVA | 0.15 | 69.01 | 76.87 (+7.86 [+5.97,+9.76]) | **77.25** (+8.24 [+6.23,+10.26]) | +0.38 [−0.55,+1.30] | 0.42 (ns) |
| POPE-adv Qwen2-VL | 0.10 | 77.79 | 79.10 (+1.31 [+0.45,+2.17], p=3.0e-03) | **80.66** (+2.87 [+2.01,+3.73], p=1.1e-10) | **+1.56** [+0.56,+2.56] | 0.0024 |
| POPE-adv Qwen2-VL | 0.15 | 94.69 | 95.09 (+0.40 [+0.34,+0.46]) | 95.08 (+0.39 [+0.06,+0.72]) | −0.01 [−0.33,+0.31] | 0.94 (ns) |
| POPE-adv LLaVA+VCD | 0.10 | 48.31 | 50.31 (+2.01 [+1.17,+2.85]) | **51.42** (+3.11 [+2.23,+3.99], p=1.3e-11) | **+1.10** [+0.05,+2.15] | 0.040 |
| POPE-adv LLaVA+VCD | 0.15 | 75.43 | 76.72 (+1.29 [+0.57,+2.00]) | 76.47 (+1.04 [+0.42,+1.65]) | −0.25 [−0.87,+0.37] | 0.43 (ns) |
| AMBER(d) LLaVA | 0.10 | 91.54 | 81.49 (**−10.05** [−12.87,−7.23]) | **92.25** (+0.71 [+0.56,+0.86]) | **+10.76** [+7.95,+13.57] | 2.6e-13 |
| AMBER(d) LLaVA | 0.15 | 95.86 | 83.90 (**−11.96** [−14.81,−9.11]) | **96.18** (+0.32 [+0.27,+0.37]) | **+12.28** [+9.42,+15.13] | 3.0e-16 |

`sym-norm` is reported in full by the script; it is uniformly weaker than `sym-2gate`
(e.g. POPE-1500 α=0.10 +1.25 vs +2.23; Qwen α=0.10 +2.33 vs +2.87) because pooling the two
normalised margins under one gate makes the *negative* side much easier to pass: the
negative-side margin distribution is heavily concentrated near 1 (its 90th percentile is
0.947–0.973 across the five settings) while the positive side's is not (0.555–0.582). A
single global top-q therefore fills up with negatives — `sym-norm`'s repaired mass is
**100% absent-polarity in every cell where it repairs anything** (present-polarity mass
0.00% in all ten cells), the exact mirror of the current defect.
It **inverts** the polarity bias instead of removing it. It is the wrong construction;
`sym-2gate`, which fixes one gate per polarity and so cannot be captured by either side, is
the one to report.

### Repaired mass and its polarity split (fraction of the test fold)

| setting | α | one-dir total / present / absent | sym-2gate total / present / absent |
|---|---|---|---|
| POPE-1500 LLaVA | 0.10 | 1.80 / **1.80** / 0.00 | 0.88 / 0.78 / **0.10** |
| POPE-1500 LLaVA | 0.15 | 1.33 / **1.33** / 0.00 | 0.48 / 0.48 / 0.00 |
| POPE-adv LLaVA | 0.10 | 1.51 / **1.51** / 0.00 | 0.86 / 0.53 / **0.33** |
| POPE-adv LLaVA | 0.15 | 1.33 / **1.33** / 0.00 | 0.41 / 0.36 / **0.05** |
| POPE-adv Qwen2-VL | 0.10 | 0.96 / **0.96** / 0.00 | 0.56 / 0.42 / **0.14** |
| POPE-adv Qwen2-VL | 0.15 | 0.35 / **0.35** / 0.00 | 0.15 / 0.15 / 0.00 |
| POPE-adv LLaVA+VCD | 0.10 | 0.86 / **0.86** / 0.00 | 0.53 / 0.22 / **0.30** |
| POPE-adv LLaVA+VCD | 0.15 | 0.43 / **0.43** / 0.00 | 0.22 / 0.17 / **0.05** |
| AMBER(d) LLaVA | 0.10 | 1.69 / **1.69** / 0.00 | 0.52 / 0.51 / **0.01** |
| AMBER(d) LLaVA | 0.15 | 0.39 / **0.39** / 0.00 | 0.31 / 0.31 / 0.00 |

The gate does what it was designed to do: negative-polarity repairs are now admitted
(0.05–0.33% of items), and their precision is **99.9–100%** in every cell where any are
admitted. The one-directional column reproduces the reviewer's 100%-positive finding
exactly.

### Realised risk, validity audit, abort rate

Audit statistic is the **fraction of splits with realised risk > α**, which must be ≤ δ =
0.10 — never mean risk. `exc_te` = held-out test fold; `exc_all` = the selected policy
re-scored on all *n* items (the low-noise proxy for its population risk).

| setting | α | arm | mean risk | exc_te | exc_all | abort |
|---|---|---|---|---|---|---|
| POPE-1500 LLaVA | 0.10 | filter / one-dir / sym | 7.63 / 7.75 / 7.79 | 0.146 / 0.142 / 0.150 | 0.030 / 0.032 / 0.024 | 0.010 / 0.000 / 0.000 |
| POPE-1500 LLaVA | 0.15 | filter / one-dir / sym | 12.48 / 12.51 / 12.48 | 0.162 / 0.156 / 0.160 | 0.026 / 0.020 / 0.024 | 0.000 |
| POPE-adv LLaVA | 0.10 | filter / one-dir / sym | 4.01 / 4.25 / 4.58 | 0.078 / 0.080 / 0.084 | 0.018 / 0.024 / 0.024 | **0.188 / 0.176 / 0.066** |
| POPE-adv LLaVA | 0.15 | filter / one-dir / sym | 9.06 / 9.81 / 10.00 | 0.114 / 0.114 / 0.116 | 0.012 / 0.018 / 0.016 | 0.096 / 0.008 / 0.000 |
| POPE-adv Qwen2-VL | 0.10 | filter / one-dir / sym | 6.40 / 6.47 / 6.63 | 0.148 / 0.144 / 0.150 | 0.022 / 0.022 / 0.024 | 0.002 / 0.012 / 0.000 |
| POPE-adv Qwen2-VL | 0.15 | filter / one-dir / sym | 11.04 / 11.03 / 11.06 | 0.100 / 0.094 / 0.098 | 0.000 | 0.000 |
| POPE-adv LLaVA+VCD | 0.10 | filter / one-dir / sym | 5.24 / 5.35 / 5.46 | 0.118 / 0.114 / 0.120 | 0.022 / 0.026 / 0.026 | 0.058 / 0.046 / 0.016 |
| POPE-adv LLaVA+VCD | 0.15 | filter / one-dir / sym | 10.42 / 10.53 / 10.53 | 0.118 / 0.120 / 0.120 | 0.022 | 0.008 / 0.004 / 0.002 |
| AMBER(d) LLaVA | 0.10 | filter / one-dir / sym | 4.70 / 3.91 / 4.70 | 0.076 / 0.058 / 0.074 | 0.012 / 0.006 / 0.012 | 0.000 / **0.130** / 0.000 |
| AMBER(d) LLaVA | 0.15 | filter / one-dir / sym | 7.68 / 6.50 / 7.67 | 0.032 / 0.026 / 0.032 | 0.000 | 0.000 / **0.128** / 0.000 |

Two things in this table need saying in the paper.

1. **`exc_all` ≤ 0.032 everywhere** — the guarantee holds for the symmetric gate, and the
   symmetric gate never degrades it relative to filtering or to the one-directional gate.
2. **`exc_te` reaches 0.146–0.162 on POPE-1500 and 0.148 on Qwen — above δ = 0.10 — for
   the FILTERING baseline as well as for every repair arm.** This is not caused by repair
   and not caused by the symmetric gate; it is the held-out estimator's binomial noise at
   n_te = n/3 around a population risk that CP deliberately parks just under α. `exc_all`
   is the right proxy and is well inside δ. `ccrc_v3.py` already flags only on `exc_all`,
   but the manuscript should state this explicitly rather than let a reader compute
   `exc_te` and conclude the guarantee fails.
3. AMBER's one-directional loss is largely an **abort** phenomenon: 13% of splits certify
   nothing at all. The symmetric gate drives that to 0.000. The same effect appears on
   POPE-adv LLaVA at α=0.10 (abort 0.188 → 0.066).

### Decomposition — where does the effect actually come from?

`sym-2gate` changes two things at once relative to the one-directional gate, so the two are
separated on identical splits.

| setting | α | from newly-admitted NEGATIVE repairs (sym − pos-only) | from RE-SCALING the positive gate (pos-only − one-dir) |
|---|---|---|---|
| POPE-1500 LLaVA | 0.10 | **+0.21** [+0.05,+0.37] p=0.012 | −1.31 [−1.49,−1.14] |
| POPE-1500 LLaVA | 0.15 | +0.00 [−0.00,+0.00] p=0.32 (ns) | −1.26 [−1.36,−1.15] |
| POPE-adv LLaVA | 0.10 | **+6.81** [+5.17,+8.44] p=2.2e-15 | −2.20 [−3.51,−0.89] |
| POPE-adv LLaVA | 0.15 | **+4.79** [+3.22,+6.36] p=3.9e-09 | −4.41 [−5.84,−2.99] |
| POPE-adv Qwen2-VL | 0.10 | **+1.87** [+1.10,+2.65] p=2.6e-06 | −0.31 [−1.10,+0.47] (ns) |
| POPE-adv Qwen2-VL | 0.15 | +0.23 [−0.09,+0.55] p=0.16 (ns) | −0.24 [−0.29,−0.20] |
| POPE-adv LLaVA+VCD | 0.10 | **+2.60** [+1.76,+3.45] p=3.0e-09 | −1.50 [−2.33,−0.68] |
| POPE-adv LLaVA+VCD | 0.15 | **+0.74** [+0.13,+1.35] p=0.018 | −0.99 [−1.70,−0.28] |
| AMBER(d) LLaVA | 0.10 | +0.01 [+0.00,+0.02] p=0.045 | **+10.75** [+7.94,+13.56] |
| AMBER(d) LLaVA | 0.15 | +0.00 p=1.0 (ns) | **+12.28** [+9.42,+15.13] |

So the effect has two entirely different sources depending on the setting. On the three
POPE-adversarial cells at α=0.10 the whole of the improvement is genuinely attributable to
negative-polarity repairs (+1.87 to +6.81 pp, all p < 1e-5). On AMBER the entire +10.8/+12.3
comes from *re-scaling*: the two-gate construction makes the positive gate much stricter
(repair mass 1.69% → 0.51%), which stops perturbing AMBER's nearly-saturated constraint and
so removes the flagship negative result. That is a real and useful finding, but it is **not
a polarity result** and must not be sold as one.

### The decisive negative finding: it still never repairs a false positive

The reason the symmetric gate was wanted is to repair the canonical false-positive existence
hallucination. It does not. Counting repaired items where the VLM said "yes", the gold is
"no", and the substituted detector answer is "no":

| setting | α | FP existence hallucinations *corrected*, one-dir | **sym-2gate** | missed objects corrected, one-dir → sym |
|---|---|---|---|---|
| POPE-1500 LLaVA | 0.10 / 0.15 | 0.000% | **0.000%** | 1.095% → 0.416% / 1.072% → 0.416% |
| POPE-adv LLaVA | 0.10 / 0.15 | 0.000% | **0.000%** | 0.492% → 0.158% / 0.596% → 0.174% |
| POPE-adv Qwen2-VL | 0.10 / 0.15 | 0.000% | **0.000%** | 0.057% → 0.000% / 0.005% → 0.000% |
| POPE-adv LLaVA+VCD | 0.10 / 0.15 | 0.000% | **0.000%** | 0.374% → 0.154% / 0.381% → 0.158% |
| AMBER(d) LLaVA | 0.10 / 0.15 | 0.000% | **0.000%** | 1.282% → 0.374% / 0.392% → 0.308% |

The admitted negative-polarity repairs are 100% *correct* but they land almost entirely on
items where the VLM had **already** answered "no" correctly. They change no answer. They are
pure risk ballast — which is exactly the v3 dilution mechanism, now operating on the other
polarity, and nothing more.

`python3 sym_gate.py --diag-only` shows why, structurally and without any calibration. Ranking
the detector-negative items by confidence and asking where the repairable false positives sit:

| POPE-1500 LLaVA, negative side | detector accuracy | FP hallucinations reachable (cumulative) |
|---|---|---|
| top 10% most-confident absent (o ∈ [0.001,0.008]) | 97.4% | 0 |
| top 20% | 94.9% | 1 |
| top 50% | 83.3% | 9 |
| top 80% | 74.7% | 23 |
| top 90% | 66.7% | 39 |
| all (o up to 0.150) | 36.7% in the last decile | 46 of 85 |

Same shape on POPE-adv LLaVA (top decile: 100% accurate, **0** of 29 reachable; last two
deciles: 42–48% accurate, where 7 of them live), Qwen2-VL (0 of 20 in the top six deciles)
and VCD (0 of 78 in the top two deciles).

**On the negative side, repair opportunity and repair precision are anti-correlated.** An
object scoring o ≈ 0.001 is so obviously absent that the VLM also says "no"; the VLM only
hallucinates presence when there is *some* visual evidence, which is precisely where the
detector's own margin is smallest and its accuracy collapses to 35–55% — far below 1−α, so
those items can never be certified. No choice of a *strict* negative gate can reach the
hallucinations, and any gate loose enough to reach them is uncertifiable. This is a property
of the signal, not of the gate's parameterisation.

The single exception is **AMBER(d)**, where the detector is 100% accurate all the way down
the negative side and a loose negative gate could reach 10–12 of the 14 false positives at
full precision. AMBER is exactly the setting where CCRC's μ−α precondition fails, so the one
place a symmetric gate could genuinely repair false positives is the one place the method
should not be used.

### Verdict — Task 1

**Mixed, and the honest headline is a negative one.**

- The symmetric gate is **valid** (nesting preserved, single FST at full δ, `exc_all` ≤ 0.032
  everywhere) and it does admit negative-polarity repairs at 99.9–100% precision.
- It **helps** at α=0.10 on the three POPE-adversarial cells (+1.10 to +4.61 pp over the
  one-directional gate, p = 0.040 / 0.0024 / 1.2e-06) and that improvement *is* attributable
  to the new negative repairs (+1.87 to +6.81 pp).
- It **does nothing** at α=0.15 on POPE (−1.26 / +0.38 / −0.01 / −0.25; three of four ns).
- It **hurts** on POPE-1500 at both α (−1.10, −1.26).
- Its largest apparent effect, converting AMBER's −10.05/−11.96 into +0.71/+0.32, is
  **entirely from re-scaling the positive gate**, not from polarity, and mostly works by
  eliminating a 13% abort rate.
- Most importantly: **the symmetric gate repairs zero false-positive existence
  hallucinations in every setting at both α.** The manuscript's "obvious remedy" does not
  fix the structural limitation it was meant to fix. The correct revision is to state the
  limitation, report that the symmetric gate was evaluated and does not remove it, and give
  the anti-correlation between negative-side opportunity and negative-side precision as the
  reason — that is a stronger and more defensible claim than the one-line "not evaluated"
  currently in the paper.

---

## TASK 2 — Power analysis for the sequential experiment

`exp14_monotonicity.json`, n = 121 paired matched-prefix continuations. Primary endpoint
d = `n_hallu_rep` − `n_hallu_orig`.

### Paired statistics

| quantity | value |
|---|---|
| mean(rep − orig) | **+0.2066** |
| sd / se | 0.9993 / 0.0908 |
| t(120) | 2.274 |
| **two-sided paired t p** | **0.0247** ← the value to quote |
| one-sided paired t p (H1: μ > 0, direction pre-specified by the hypothesis) | 0.0124 |
| 95% CI | [+0.0267, +0.3865] |
| 90% CI | [+0.0560, +0.3572] |
| bootstrap 95% CI (2e4) | [+0.0331, +0.3884] |
| pairs worse / better / **tied** | 45 / 25 / **51** (42.1%) |
| Wilcoxon signed-rank, two-sided / one-sided | 0.0347 / 0.0173 |
| exact sign test, two-sided / one-sided | 0.0225 / 0.0112 |
| Hodges–Lehmann location | +0.0000 |
| Cohen's d_z | 0.2068 |

Convention: the one-sided test is `H0: μ ≤ 0` vs `H1: μ > 0`, i.e. in the direction the
manuscript hypothesises. The reported "+0.207 ± 0.091, p = 0.025" is the **two-sided**
paired t, and it reproduces exactly. Note the Hodges–Lehmann estimate is 0.000 — with 42% of
pairs tied, the rank-based location estimate is null even though the mean and every test
statistic are positive; that should be stated.

The manuscript's omitted row, and one it does not report:

| endpoint | mean diff | two-sided p | 95% CI |
|---|---|---|---|
| faithful mentions (`n_truth`) | +0.1736 | 0.0769 | [−0.0190, +0.3661] |
| continuation length | −0.5455 words | 0.0916 | [−1.1806, +0.0896] |

### Power

Observed sd of the paired difference = 0.9993, n = 121, se = 0.0908.

| | two-sided α=0.05 | one-sided α=0.05 |
|---|---|---|
| **MDE at 80% power** | **+0.2566** (21.3% of the baseline 1.207) | +0.2272 (18.8%) |
| MDE at 90% power | +0.2969 | +0.2674 |
| **achieved power at the observed +0.2066** | **61.6%** | 73.1% |
| d_z needed for 80% power (observed 0.2068) | 0.2568 | 0.2273 |
| n needed for 80% power at the observed effect | **186** (65 more pairs) | 146 (25 more) |
| n needed for 90% power | 248 (127 more) | — |

The observed effect (+0.207) is **below** the two-sided 80%-power MDE (+0.257). The study
was powered to detect effects ≥ 0.257 with 80% probability and detected a smaller one: this
is a marginally-powered positive, not a well-powered one. (Post-hoc "achieved power" is a
monotone restatement of the p-value and is not evidence about the true effect; it is
reported only because it was asked for. The MDE and the TOST bound are the informative
numbers.)

### TOST equivalence — what the experiment could and could not rule out

`H0` pair: μ ≤ −Δ and μ ≥ +Δ; equivalence declared iff both one-sided tests reject at 0.05
(equivalently, the 90% CI lies inside ±Δ).

| Δ | % of baseline | p_lower | p_upper | TOST p | equivalent? |
|---|---|---|---|---|---|
| 0.10 | 8.3% | 0.0005 | 0.8785 | 0.8785 | no |
| 0.15 | 12.4% | 0.0001 | 0.7328 | 0.7328 | no |
| 0.20 | 16.6% | 0.0000 | 0.5289 | 0.5289 | no |
| **0.25** | **20.7%** | 0.0000 | 0.3169 | **0.3169** | **no** |
| 0.30 | 24.9% | 0.0000 | 0.1530 | 0.1530 | no |
| **0.40** | 33.2% | 0.0000 | 0.0177 | 0.0177 | **YES** |
| 0.50 | 41.4% | 0.0000 | 0.0008 | 0.0008 | YES |
| 1.00 | 82.9% | 0.0000 | 0.0000 | 0.0000 | YES |

**Smallest equivalence margin establishable at 5%: Δ = 0.357** (29.6% of the baseline
hallucination count). Against the natural pre-specified margin of ±0.25 (a quarter of a
hallucinated object per repair, ≈20% of baseline) the experiment **cannot** establish
equivalence (TOST p = 0.317).

So the experiment simultaneously: rules out an exact null (p = 0.025), rules out true
effects larger in magnitude than 0.357, and **cannot distinguish +0.207 from anything in
[+0.027, +0.386] — a 14-fold range.** The increase is established as a **sign, not a
magnitude.**

### Per-word figure — the 0.0075 / 0.0115 / 0.0069 discrepancy is resolved

| variant | estimate | se | two-sided p | one-sided p |
|---|---|---|---|---|
| **(i) paired per-item rate difference, 119 pairs with len > 0** | **+0.0075** | 0.0029 | **0.0114** | 0.0057 |
| (ii) same, all 121 pairs, len clamped ≥ 1 | +0.0115 | 0.0050 | 0.0228 | 0.0114 |
| (iii) pooled ratio-of-totals Σh/Σlen | +0.0069 | 0.0027 | 0.0086 (bootstrap) | — |
| (iv) ratio-of-means | +0.0069 | — | — | — |
| (v) count effect ÷ mean original length | +0.0062 | — | — | — |

- The manuscript's **"+0.0075, p = 0.011" is variant (i) and is CORRECT — and its p-value is
  already two-sided.** 95% CI [+0.0017, +0.0132], n = 119.
- The reviewer's **+0.0115 / p = 0.023 is variant (ii)**. The only difference is that it
  retains the 2 degenerate pairs with `len_orig = 0`: id 35 (len_rep = 5, h_rep = 0 → +0.0000)
  and id 90 (**len_rep = 2, h_rep = 1 → +0.5000**). That single 2-word continuation supplies
  36% of variant (ii)'s entire numerator and also inflates its variance, which is why (ii)
  has both a larger point estimate and a larger p-value.
- The reviewer's **+0.0069 pooled is variant (iii)**, a ratio of totals; it is not a paired
  statistic and has no paired p-value (paired-cluster bootstrap: 95% CI [+0.0016, +0.0122],
  two-sided p = 0.0086).

**Correct two-sided value to report:** per-word hallucination rate **+0.0075 (95% CI
[+0.0017, +0.0132]), two-sided paired p = 0.011, n = 119 of 121 pairs (2 excluded for
`len_orig = 0`)**, with the pooled ratio-of-totals +0.0069 given for reference. The
discrepancy is a 2-pair inclusion rule, not an arithmetic error on either side — but it must
be disclosed, because the result is not robust to those two pairs (p = 0.011 excluding vs
p = 0.023 including).

### Hallucinated share of mentions — no significant change, confirmed

| quantity | value |
|---|---|
| mean(share_rep − share_orig), 111 pairs with ≥1 mention in both | **−0.0039** |
| **two-sided paired t p** | **0.8844** |
| 95% CI | [−0.0563, +0.0486] |
| pairs worse / better / tied | 35 / 32 / 44 |
| Wilcoxon two-sided p | 0.9725 |
| pooled share | 0.4332 → 0.4465 (+0.0132) |
| mention density | 0.0833 → 0.0963 mentions/word |

Confirmed: the hallucinated *share* of mentions does not move (−0.004, p = 0.884). Repair
makes continuations more object-dense (mention density +15% relative), lifting both
hallucinated and faithful counts.

### Verdict — Task 2

The structural conclusion survives, but only as a directional claim, and the paper should say
so.

- The effect is real in sign at conventional levels under **three** independent tests (t
  p = 0.025, Wilcoxon p = 0.035, sign test p = 0.023), which is the right robustness evidence
  to cite given 42% ties.
- It is **under-powered as a magnitude claim**: observed +0.207 sits below the two-sided
  80%-power MDE of +0.257, achieved power 62%, and 186 pairs would be needed for 80% power
  at this effect size. TOST cannot establish equivalence within ±0.25.
- The manuscript's per-word figure **+0.0075, p = 0.011** is correct and already two-sided;
  the reviewer's +0.0115 comes from including two `len_orig = 0` pairs, one of which
  contributes +0.5 on a 2-word continuation. Disclose the exclusion rule.
- The share-of-mentions null (−0.004, p = 0.884) is confirmed and must be reported alongside
  the count result, together with the faithful-mention rise (+0.174, p = 0.077) and the
  shorter continuations (−0.55 words, p = 0.092).
- Recommended wording: the experiment shows repair **does not reliably reduce** downstream
  hallucination and more likely increases it, which is enough to close the monotone
  fixed-point route (that argument needs only "not non-increasing"). It does **not** support
  any quantitative statement about how much repair degrades a continuation, and it does not
  support "repair makes the continuation worse" per mention.

---

## TASK 3 — The q-selection story

The claim under audit: q "is fixed a priori, not tuned", asserted next to a table showing
q = 0.10 is best of {0.10, 0.25, 0.50} on the same cells that produce the headline results.

### (1) The selection surface — gain over filtering (pp), 500 paired splits

| setting | α | q = 0.10 | q = 0.25 | q = 0.50 |
|---|---|---|---|---|
| POPE-1500 LLaVA | 0.10 | +3.34 [+2.7,+4.0] | +4.92 [+3.4,+6.4] | **+8.95** [+7.9,+10.0] |
| POPE-adv LLaVA | 0.10 | **+3.18** [+1.9,+4.5] | +0.42 [−2.2,+3.0] | +2.57 [−0.5,+5.7] |
| POPE-adv Qwen2-VL | 0.10 | **+1.31** [+0.4,+2.2] | −16.18 [−19.4,−13.0] | −14.71 [−17.9,−11.6] |
| POPE-adv LLaVA+VCD | 0.10 | **+2.01** [+1.2,+2.8] | −5.56 [−8.1,−3.0] | −8.59 [−11.3,−5.9] |
| AMBER(d) LLaVA | 0.10 | **−10.05** [−12.9,−7.2] | −37.03 [−41.2,−32.9] | −25.46 [−29.3,−21.6] |
| POPE-1500 LLaVA | 0.15 | +1.93 [+1.8,+2.0] | +5.34 [+5.2,+5.5] | **+6.67** [+6.5,+6.9] |
| POPE-adv LLaVA | 0.15 | +7.86 [+6.0,+9.8] | +10.34 [+8.1,+12.6] | **+14.71** [+12.5,+16.9] |
| POPE-adv Qwen2-VL | 0.15 | +0.40 [+0.3,+0.5] | −1.15 [−2.4,+0.1] | **+0.47** [−0.3,+1.3] |
| POPE-adv LLaVA+VCD | 0.15 | +1.29 [+0.6,+2.0] | +0.55 [−0.8,+1.9] | **+1.80** [+0.7,+2.9] |
| AMBER(d) LLaVA | 0.15 | −11.96 [−14.8,−9.1] | −12.64 [−15.5,−9.7] | **−2.23** [−3.7,−0.8] |

Cells with a positive gain (of 10): **q=0.10 → 8**, q=0.25 → 5, q=0.50 → 6.
Cell-wise argmax: at α=0.10 q=0.10 wins 4 of 5; **at α=0.15 q=0.50 wins all 5.**

The first correction the manuscript needs is factual: **q = 0.10 is not the best of the
three by gain.** Mean gain over the 8 POPE cells is +2.66 at q=0.10 against +4.89 for the
per-cell oracle. q = 0.10 is best on a *different* criterion — it is the only value that is
positive in every POPE cell at both α, i.e. it is the most robust, not the strongest. That
is a legitimate and much more defensible reason to prefer it, and it should replace the
current "best of {…}" framing. It is also, still, a criterion that was evaluated on the
headline cells.

### (2)/(3) Oracle vs leave-one-dataset-out transfer

- **fixed** — q = 0.10 as reported.
- **oracle** — q chosen per (dataset, α) on that cell's own gain. Upper bound on selection.
- **LODO-A** — q maximising mean gain over the *other* datasets, pooled over both α, applied
  to the held-out dataset.
- **LODO-C** — q maximising mean gain over the other datasets *at the same α* (α is
  user-specified, not tuned, so conditioning on it is legitimate).

| setting | α | fixed q=.10 | oracle | q* | LODO-A | q* | LODO-C | q* |
|---|---|---|---|---|---|---|---|---|
| POPE-1500 LLaVA | 0.10 | +3.34 | +8.95 | 0.5 | +3.34 | 0.1 | +3.34 | 0.1 |
| POPE-1500 LLaVA | 0.15 | +1.93 | +6.67 | 0.5 | +1.93 | 0.1 | **+6.67** | 0.5 |
| POPE-adv LLaVA | 0.10 | +3.18 | +3.18 | 0.1 | +3.18 | 0.1 | +3.18 | 0.1 |
| POPE-adv LLaVA | 0.15 | +7.86 | +14.71 | 0.5 | +7.86 | 0.1 | **+14.71** | 0.5 |
| POPE-adv Qwen2-VL | 0.10 | +1.31 | +1.31 | 0.1 | **−14.71** | 0.5 | +1.31 | 0.1 |
| POPE-adv Qwen2-VL | 0.15 | +0.40 | +0.47 | 0.5 | +0.47 | 0.5 | +0.47 | 0.5 |
| POPE-adv LLaVA+VCD | 0.10 | +2.01 | +2.01 | 0.1 | +2.01 | 0.1 | +2.01 | 0.1 |
| POPE-adv LLaVA+VCD | 0.15 | +1.29 | +1.80 | 0.5 | +1.29 | 0.1 | +1.80 | 0.5 |
| AMBER(d) LLaVA | 0.10 | −10.05 | −10.05 | 0.1 | −10.05 | 0.1 | −10.05 | 0.1 |
| AMBER(d) LLaVA | 0.15 | −11.96 | −2.23 | 0.5 | −11.96 | 0.1 | −2.23 | 0.5 |
| **mean, 10 cells** | | **−0.07** | **+2.68** | | **−1.66** | | **+2.12** | |
| **mean, 8 POPE cells** | | **+2.66** | **+4.89** | | **+0.67** | | **+4.19** | |

Headline numbers for the revision:

- **Selection exposure (oracle − LODO-A) = +4.35 pp** over 10 cells; **oracle − LODO-C =
  +0.56 pp** over 10 cells and **+0.70 pp** over the 8 POPE cells.
- **Transferring q by the sensible protocol (LODO-C: pick q on held-out datasets at the
  deployment α) gives a LARGER mean gain than the reported fixed q = 0.10: +4.19 vs +2.66 pp
  over the 8 POPE cells.** Selection is not inflating the headline — the headline is
  *conservative* relative to a properly transferred choice.
- **q does not transfer as a single number across α.** LODO-C selects q = 0.10 at α = 0.10
  and q = 0.50 at α = 0.15 on every fold. Pooling across α (LODO-A) is actively harmful: it
  picks q = 0.50 for the Qwen α=0.10 fold and turns +1.31 into **−14.71**, dragging the
  10-cell mean to −1.66 — worse than the fixed q=0.10's −0.07 by 1.60 pp.
- So the correct statement is *not* "q transfers" and *not* "the gain is inflated by
  selection". It is: **q must be indexed to α; conditional on α it transfers across datasets
  and would improve the reported numbers; and the reported q = 0.10 is a single robust value
  that costs ≈1.5 pp of mean gain at α=0.15 in exchange for never losing on a POPE cell.**

### (4) Multiplicity correction: all three q at δ/3, then select on the calibration fold

`k_min = ⌈ln δ / ln(1−α)⌉`, the smallest emitted set CP can certify:

| α | δ = 0.10 | δ/3 = 0.0333 | extra clean samples required |
|---|---|---|---|
| 0.10 | 22 | **33** | +11 |
| 0.15 | 15 | **21** | +6 |

| setting | α | filter | q=.10 @ δ | q=.10 @ δ/3 | **select @ δ/3** | gain(sel) | cost vs q=.10@δ | abort | exc_all |
|---|---|---|---|---|---|---|---|---|---|
| POPE-1500 LLaVA | 0.10 | 68.37 | 71.71 | 68.13 | **75.34** | +6.97 | **+3.64** | 0.000 | 0.012 |
| POPE-1500 LLaVA | 0.15 | 86.55 | 88.48 | 85.79 | **91.30** | +4.75 | **+2.81** | 0.000 | 0.010 |
| POPE-adv LLaVA | 0.10 | 43.58 | 46.76 | 41.81 | **50.72** | +7.14 | **+3.95** | 0.140 | 0.004 |
| POPE-adv LLaVA | 0.15 | 69.01 | 76.87 | 66.39 | **78.24** | +9.23 | **+1.36** | 0.006 | 0.002 |
| POPE-adv Qwen2-VL | 0.10 | 77.79 | 79.10 | 69.27 | 74.90 | −2.89 | **−4.20** | 0.010 | 0.008 |
| POPE-adv Qwen2-VL | 0.15 | 94.69 | 95.09 | 92.06 | 93.65 | −1.04 | −1.44 | 0.000 | 0.000 |
| POPE-adv LLaVA+VCD | 0.10 | 48.31 | 50.31 | 38.48 | 46.83 | −1.48 | **−3.49** | 0.120 | 0.006 |
| POPE-adv LLaVA+VCD | 0.15 | 75.43 | 76.72 | 68.10 | 74.19 | −1.24 | −2.53 | 0.000 | 0.006 |
| AMBER(d) LLaVA | 0.10 | 91.54 | 81.49 | 78.24 | 78.92 | −12.61 | −2.56 | 0.146 | 0.002 |
| AMBER(d) LLaVA | 0.15 | 95.86 | 83.90 | 83.14 | **91.38** | −4.48 | **+7.48** | 0.044 | 0.000 |
| **mean** | | | | | | | **+0.50** | | ≤ 0.012 |

Reading:

- The δ/3 tax on a *fixed* q is real and large: **−0.8 to −11.8 pp** of certified coverage for
  the same q = 0.10 (POPE-1500 71.71 → 68.13; Qwen 79.10 → 69.27; VCD 50.31 → 38.48;
  POPE-adv LLaVA at α=0.15 76.87 → 66.39).
- But the union-bound version *earns the tax back* by being allowed to select: mean cost of
  the fully-valid select-at-δ/3 procedure versus the reported q=0.10-at-full-δ is only
  **+0.50 pp** — i.e. roughly free on average. It gains +1.4 to +7.5 pp on five cells (both
  POPE-1500 cells, both POPE-adv LLaVA cells, AMBER at α=0.15) and loses 1.4 to 4.2 pp on the
  other five (Qwen, VCD, AMBER at α=0.10).
- Validity holds throughout: `exc_all` ≤ 0.012 in every row. Abort rate rises to 0.12–0.15 in
  three cells, which is the visible face of the larger `k_min`.
- Caveat on the selection rule, which should be stated if this version is adopted: selecting
  the q with the largest *certified calibration coverage* systematically prefers the loosest
  gate (it picks q = 0.50 on 61–87% of splits for POPE-1500 and POPE-adv LLaVA, and only
  22–23% for AMBER). It has no way to know that a loose gate transfers badly. That is the
  substantive reason a fixed strict q is preferable to a validly-selected one here, and it is
  a better argument than the current unsupported "fixed a priori".

### Verdict — Task 3

The claim as written is not defensible, but the exposure is small and the fix is cheap.

1. **Drop "q = 0.10 is the best of {0.10, 0.25, 0.50}".** It is false by mean gain (q = 0.50
   wins all five cells at α = 0.15 and the POPE-1500 cell at α = 0.10). Replace with: q = 0.10
   is the only candidate that is positive in every POPE cell at both α (8/10 vs 5/10 and
   6/10) — a robustness criterion, not a maximum-gain criterion.
2. **Price the selection honestly with the LODO number.** Selecting q on held-out datasets at
   the deployment α (LODO-C) yields **+4.19 pp** mean gain over the 8 POPE cells versus
   **+2.66 pp** for the reported fixed q = 0.10 and **+4.89 pp** for the per-cell oracle. The
   selection exposure is therefore **+0.70 pp**, and it runs in the *conservative* direction:
   the reported headline understates what a properly transferred q would give.
3. **State that q must be indexed to α.** Conditional on α, q transfers across datasets;
   pooled across α it does not (LODO-A costs 16.0 pp on the Qwen α=0.10 fold).
4. **The δ/3 multiplicity-corrected version is available and is a wash** (mean +0.50 pp vs
   the reported configuration, validity `exc_all` ≤ 0.012, `k_min` 22 → 33 at α = 0.10). It is
   worth reporting as an appendix arm, since it removes the objection entirely at negligible
   average cost — but note that it selects the loosest gate on most splits and therefore
   inherits q = 0.50's transfer problems.

---

## Cross-cutting note the revision should absorb

`exc_te`, the fraction of splits whose realised risk on the held-out fold exceeds α, is
**0.146–0.162** on POPE-1500 and **0.148** on Qwen at α = 0.10 — above δ = 0.10 — for the
plain **filtering** baseline as much as for any repair arm. `exc_all`, the same statistic with
the selected policy re-scored on all n items, is ≤ 0.032 everywhere. The gap is the held-out
estimator's binomial noise at n_te = n/3 around a population risk that Clopper–Pearson parks
just under α, not a validity failure — but the manuscript's "risk guarantee verified in every
row" language invites a reader to compute the test-fold number and conclude otherwise. State
which statistic is being audited and why.
