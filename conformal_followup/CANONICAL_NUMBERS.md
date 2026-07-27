# CANONICAL_NUMBERS.md — single source of truth

Every number the manuscript reports, recomputed under **one** protocol, organised by the
table or claim it feeds, old value beside new, deltas called out.

Generated from the scripts in this directory. Nothing here is hand-copied from
`RESULTS.md`, `ALGORITHM.md` or a previous draft; every figure has a script and a command.

---

## 0. How to read this file

**Section 1** states the protocol and what changed. **Section 2** is the list of places where
the corrected number *changes a conclusion* rather than a digit — read that first.
**Sections 3–12** go table by table. **Sections 13–15** are the three new analyses.
**Section 16** records where I could *not* reproduce the reviewer's numbers.

Provenance map — which script produces which table:

| manuscript object | script | command |
|---|---|---|
| Table 3 `tab:score` | `local_analysis_owlv2.py` | `python3 local_analysis_owlv2.py` |
| Table 4 `tab:selfcons` | `local_selfconsistency.py` | `python3 local_selfconsistency.py` |
| Table 5 `tab:ccrc` | `ccrc_gains_stats.py` | `python3 ccrc_gains_stats.py` |
| Table 6 `tab:datasets` | `master_comparison.py` | `python3 master_comparison.py` |
| Table 6 Qwen rows | `local_backbone_analysis.py` | `python3 local_backbone_analysis.py` |
| Table 7 `tab:baselines` | `missing_numbers.py` BLOCK 2 + `ccrc_gains_stats.py` | `python3 missing_numbers.py` |
| Table 8 `tab:attrib` | `missing_numbers.py` BLOCK 3 | ″ |
| Table 11 `tab:combiner` | `combiner_ablation.py` | `python3 combiner_ablation.py` |
| Table 12 `tab:bounds` | `missing_numbers.py` BLOCK 1 | `python3 missing_numbers.py` |
| Table 13 `tab:qsens` | `ccrc_gains_stats.py` | ″ |
| Table 14 `tab:multialpha` | `local_multi_alpha.py` | `python3 local_multi_alpha.py` |
| §4.4 self-repair | `self_repair_experiment.py` **(new — previously no script existed)** | `python3 self_repair_experiment.py` |
| §4.5 dilution / leverage | `ccrc_gains_stats.py` **(new decomposition)** | ″ |
| §5.2 AMBER negative result | `amber_diagnosis.py` **(new)** | `python3 amber_diagnosis.py` |
| §4.4 bottom-q accuracy | `missing_numbers.py` BLOCK 5 + `self_repair_experiment.py` | ″ |
| §5.5 subsample gains | `missing_numbers.py` BLOCK 6, `amber_diagnosis.py` | ″ |
| protocol justification | `protocol_audit.py` **(new)** | `python3 protocol_audit.py` |
| Fig 5 `risk_coverage.png` | `risk_coverage_vs_conflvlm.py` | ″ |
| Fig 6 `coverage_vs_alpha.png` | `local_multi_alpha.py` | ″ |
| Fig 8 `comparison_figure.png` | `make_comparison_figure.py` | ″ |
| Figs 2/4/7 | `make_paper_figures.py` | `python3 make_paper_figures.py` |

The shared protocol lives in **`canonical_fst.py`** — one implementation, imported by every
script, so a change to the procedure cannot silently apply to some tables and not others.

---

## 1. The protocol, and what changed

### 1.1 What the eight scripts used to do

`local_analysis_owlv2.py`, `master_comparison.py`, `local_multi_alpha.py`,
`local_selfconsistency.py`, `local_backbone_analysis.py`, `combiner_ablation.py`,
`risk_coverage_vs_conflvlm.py` and `make_comparison_figure.py` each selected the threshold by
an exhaustive maximum-coverage search:

```python
for t in np.unique(score_cal):
    m = score_cal >= t; k = m.sum(); e = (1 - ok_cal[m]).sum()
    if cp_upper(e, k, DELTA) <= alpha and k > best: best, tau = k, t
```

"Of all O(n_cal) thresholds whose Clopper–Pearson bound passes, keep the one that emits the
most." No prefix rule, no `k_min` start, no multiplicity correction. The selection event is a
union over the whole grid, so the per-threshold δ does not transfer to the selected
threshold. Meanwhile §7.1 of the manuscript claims *"Every reported number uses a three-way
disjoint split with fixed-sequence testing and Clopper–Pearson bounds."* Only `ccrc_v3.py` and
`missing_numbers.py` actually did that.

### 1.2 What they do now

All eight now call `canonical_fst.fst`, which is `ccrc_v3.py`'s procedure with
`missing_numbers.py`'s bound-specific `k_min`:

1. `lam_grid(n_cal, alpha, delta, ub)` — `ccrc_v3`'s grid: 40 points, `np.linspace(lam0, 1.0, 40)`
   with `lam0 = min(0.9, max(0.05, 1.5 * k_min / n_cal))`. Verified numerically identical to
   `ccrc_v3.lam_grid` at α ∈ {0.10, 0.15}.
2. `k_min = kmin_for(ub, alpha, delta)` — `missing_numbers`' bound-specific value. For
   Clopper–Pearson it reproduces Lemma 1 exactly: `k_min = ⌈ln δ / ln(1−α)⌉`.
   **Verified: 45 / 22 / 15 / 11 at α = 0.05 / 0.10 / 0.15 / 0.20 with δ = 0.10**, and
   **29 at α = 0.10, δ = 0.05** (the manuscript's "95% confidence" aside is right that
   k_min ≠ 22 there). Loose bounds: Hoeffding 116, empirical Bernstein 71, betting 44.
3. Ascending sweep, **stop at the FIRST non-rejection, return the LAST passing index**. If
   the emitted count falls below `k_min` the sequence stops (skipping an untestable
   hypothesis mid-sequence would break fixed-sequence FWER control).
4. A single test at **full δ** — legitimate precisely because the order is pre-specified.

`ccrc_v3.py` and `missing_numbers.py` were already correct and their **procedure is
untouched**. The only edit to `missing_numbers.py` is `REPS: 60 → 400` (§1.3), which changes
Monte-Carlo precision, not behaviour.

**Equivalence verified, not assumed.** `canonical_fst.fst` was checked against
`ccrc_v3.calibrate` on identical splits across all five settings × α ∈ {0.10, 0.15} ×
{filtering, q=0.10}: **0 disagreements in 2,400 comparisons** (λ̂ identical to 1e−12, including
agreement on which splits abort). The two implementations differ only in that `fst` stops
explicitly when the emitted count drops below `k_min` where `calibrate` skips a `k == 0` grid
point; with the `k_min`-anchored `lam0` that branch is never reached, which is why the
behaviours coincide exactly. So the ported scripts now run *literally* `ccrc_v3`'s procedure,
and `ccrc_v3.py`'s own numbers are unchanged by anything in this pass.

Nothing else about what any script measures changed: same α values, same scores, same feature
sets, same seeds, same data files, same item filters.

### 1.3 Repetitions and pairing

Every count is now **≥ 400** (`canonical_fst.REPS = 400`), against 20–60 before. At 60 reps
the Monte-Carlo SE on these gains is 0.8–4.2 pp, so a gain quoted to a tenth of a point was
noise. Splits come from `canonical_fst.SPLITS(n, reps, seed)` and are **paired**: every arm of
every comparison sees byte-identical folds, so gains are paired differences and
`ccrc_gains_stats.py`'s filtering and CCRC columns are computed on the same 400 splits.

Consistent with the reviewer's warning, **every 400-rep estimate came out less favourable
than its 20/60-rep predecessor.** Not one moved the other way.

### 1.4 Reporting discipline now applied everywhere

- **per-split sd** and a **95% CI for the mean gain** (Student-t on the paired differences);
- **two-sided** paired t-test, plus a two-sided Wilcoxon signed-rank as a robustness check
  (`p(t)` and `p(Wilc)` columns). All p-values in this file are two-sided;
- **ABORT RATE** — fraction of splits on which the fixed sequence certifies nothing and the
  deployed system covers 0%. Previously unreported everywhere. It reaches **32.5%** on
  Table 3's own baseline row and **100%** on two cells of Table 6;
- **validity audited as `Pr[realised risk > α]` against δ**, on the test fold (`exc(te)`) and
  re-scored on all items (`exc(all)`, the low-noise population proxy) — **never as mean
  risk**, which is not the guarantee and appears nowhere in the new output.

> **Caveat on the p-values, to be stated in the paper.** The 400 splits resample the *same*
> n items, so the CI shrinks like 1/√reps while dataset-level uncertainty does not shrink at
> all. These p-values test whether the mean gain over the *split distribution* differs from
> zero **conditional on this dataset**. They are not evidence the gain would replicate on new
> items. Reporting p = 1e-100 without this sentence would be worse than reporting nothing.

### 1.5 The leaky rule measured against the canonical one — `protocol_audit.py`

Both rules, identical paired splits, identical fitted score; the *only* difference is
threshold selection. Filtering arm, 400 splits:

| setting | n | cov leaky | cov FST | Δ cov | exc(te) leaky | exc(te) FST | **exc(all) leaky** | **exc(all) FST** | abort FST |
|---|---|---|---|---|---|---|---|---|---|
| **α = 0.10** ||||||||||
| POPE-1500 LLaVA | 1500 | 70.8% | 68.6% | −2.2 | 18.5% | 14.9% | **5.8%** | **3.5%** | 0.8% |
| POPE-adv LLaVA | 444 | 57.8% | 44.0% | −13.9 | 11.8% | 10.8% | **3.8%** | **2.5%** | 18.8% |
| POPE-adv Qwen2-VL | 591 | 82.6% | 77.9% | −4.6 | 17.8% | 16.0% | **3.0%** | **1.8%** | 0.2% |
| POPE-adv LLaVA+VCD | 591 | 57.0% | 48.8% | −8.2 | 16.5% | 13.0% | **3.2%** | **1.9%** | 5.8% |
| AMBER(d) LLaVA | 228 | 91.6% | 91.8% | +0.2 | 5.2% | 7.8% | **1.0%** | **1.5%** | 0.0% |
| **α = 0.15** ||||||||||
| POPE-1500 LLaVA | 1500 | 87.6% | 86.4% | −1.2 | 19.8% | 15.5% | 4.8% | 3.2% | 0.0% |
| POPE-adv LLaVA | 444 | 79.0% | 69.1% | −9.9 | 14.5% | 14.1% | 2.5% | 0.8% | 9.5% |
| POPE-adv Qwen2-VL | 591 | 95.7% | 94.8% | −0.9 | 10.8% | 10.5% | 0.0% | 0.0% | 0.0% |
| POPE-adv LLaVA+VCD | 591 | 79.4% | 75.8% | −3.6 | 15.8% | 12.6% | 3.0% | 2.0% | 0.8% |
| AMBER(d) LLaVA | 228 | 95.7% | 95.9% | +0.2 | 3.2% | 3.2% | 0.0% | 0.0% | 0.0% |

CCRC arm (q = 0.10), same protocol:

| setting | α | cov leaky | cov FST | Δ | exc(all) leaky | exc(all) FST | abort FST |
|---|---|---|---|---|---|---|---|
| POPE-1500 LLaVA | 0.10 | 73.3% | 71.8% | −1.6 | 4.8% | 3.5% | 0.0% |
| POPE-adv LLaVA | 0.10 | 60.0% | 47.2% | −12.8 | 4.0% | 3.3% | 17.5% |
| POPE-adv Qwen2-VL | 0.10 | 84.0% | 79.5% | −4.5 | 3.2% | 1.8% | 0.8% |
| POPE-adv LLaVA+VCD | 0.10 | 58.1% | 51.0% | −7.1 | 3.2% | 2.3% | 4.0% |
| AMBER(d) LLaVA | 0.10 | 93.4% | 81.6% | −11.8 | 1.0% | 0.9% | 13.0% |
| POPE-1500 LLaVA | 0.15 | 89.4% | 88.4% | −1.1 | 4.2% | 2.0% | 0.0% |
| POPE-adv LLaVA | 0.15 | 80.9% | 76.7% | −4.2 | 2.5% | 1.5% | 0.5% |
| POPE-adv Qwen2-VL | 0.15 | 96.1% | 95.2% | −0.9 | 0.0% | 0.0% | 0.0% |
| POPE-adv LLaVA+VCD | 0.15 | 80.0% | 77.2% | −2.8 | 3.0% | 2.0% | 0.2% |
| AMBER(d) LLaVA | 0.15 | 96.1% | 84.0% | −12.2 | 0.0% | 0.0% | 12.8% |

**Reading.** The leaky rule is measurably more anti-conservative: on 9 of the 10 filtering
cells its `exc(all)` is strictly higher, and it buys 0.9–13.9 pp of coverage it has not paid
for. But **both rules keep `exc(all)` below δ = 0.10 on every cell I ran** — see §16, where I
report that I could not reproduce the reviewer's 7.0–9.6% figure. The case for the port is
therefore *(a)* the rule has no valid theory behind it and contradicts the paper's own
protocol statement, *(b)* it is directionally anti-conservative, and *(c)* it silently hides
abort behaviour that turns out to be the dominant effect in three separate places. It is
**not** that the guarantee was empirically violated.

The **coverage cost of validity** is the price to be reported honestly:
paired FST − leaky, in pp, 400 splits, two-sided tests:

| cell | mean | sd/split | 95% CI | p(t) |
|---|---|---|---|---|
| α=0.10 filter, POPE-1500 | −2.22 | 6.76 | [−2.89, −1.56] | 1.5e−10 |
| α=0.10 filter, POPE-adv LLaVA | −13.87 | 21.34 | [−15.97, −11.78] | 1.8e−32 |
| α=0.10 filter, Qwen2-VL | −4.63 | 11.89 | [−5.80, −3.46] | 6.1e−14 |
| α=0.10 filter, LLaVA+VCD | −8.17 | 14.21 | [−9.57, −6.77] | 1.2e−26 |
| α=0.10 filter, AMBER | +0.15 | 2.77 | [−0.12, +0.43] | 0.265 (n.s.) |
| α=0.10 CCRC, POPE-1500 | −1.56 | 3.21 | [−1.87, −1.24] | 3.8e−20 |
| α=0.10 CCRC, AMBER | −11.80 | 31.82 | [−14.93, −8.67] | 7.3e−13 |

---

## 2. ⚠ CONCLUSION CHANGES — not just digits

These nine change what the paper *says*, not merely what it prints. Each is expanded in the
section named.

**C1. Table 3's headline attribution changes size and mechanism (§3).** The paper says
detection grounding adds "+8.4 points of certified coverage against +2.3 for CLIP" (and
Table 3's own rows imply +9.5 and +2.4 — already inconsistent). Canonically the gains are
**+27.0 pp** (detection) and **+21.8 pp** (CLIP) — because the confidence-only baseline
**aborts on 32.5% of splits** and raw confidence on **59.2%**. The finding is no longer "a
better score buys a few more points"; it is *"weak scores cannot certify at all under a valid
fixed sequence."* That is a stronger result, but it is a different one, and the +8.4/+2.3
framing must go.

**C2. The nine-values problem is resolved: POPE-1500 at α=0.10 is 68.6% (§3, §5, §7).**
Certified coverage for POPE-1500/LLaVA/α=0.10 appeared as 63.6, 65.2, 68.2, 72.0, 72.5, 73.1,
73.6, 78.9 and 79% across Tables 3/5/6/7/8 and Figs 5/7. Under the canonical protocol every
one of those slots takes the value **68.6% (filtering)** or **71.8% (CCRC)**, because
Tables 3, 5, 6, 8, 14 and Fig 5 all use the same four features on the same 1500 items and now
the same rule and the same 400 splits. The disputed **73.1** was the leaky rule at 20 reps;
the leaky rule at 400 reps gives 70.8, and the valid rule gives 68.6.

**C3. AMBER's negative result is an ABORT phenomenon, not absent headroom (§14).**
The paper says the cause is "the absence of headroom, exactly as Proposition 3 predicts" and
explicitly rejects sample size. Measured: conditional on both arms certifying, **CCRC WINS on
AMBER, +2.17 pp at α=0.10 and +0.47 pp at α=0.15**; filtering **never** aborts (0.0%) while
CCRC aborts on **13.0% / 12.8%** of splits; and the CCRC-only-abort split class alone accounts
for **119% / 103%** of the reported loss. Worse for the current explanation, **r_r = 3.6% <
r_a = 4.5%**, so dilution is running in CCRC's *favour* there. The stated diagnosis is wrong.

**C4. The manuscript's own refutation of the sample-size explanation does not survive (§14).**
"Subsampling POPE to the same n=228 still gains +3.2 points" is doubly unsupportable. First,
+3.2/+5.6/+10.3 come from `missing_numbers.py` BLOCK 6, which measures the **grounding-score**
gain (confidence-only vs confidence+detection), *not* the CCRC-vs-filtering gain — a different
quantity that cannot bear on the AMBER loss. Second, the right comparison, CCRC vs filtering
on POPE subsampled to n=228, canonically **loses −3.65 pp** with abort rates 43.5%/53.8%.

**C5. The "1.9–3.4× leverage multiplier" is mostly arithmetic, not leverage (§13).**
λ̂ is *identical* between filtering and CCRC on **70.6–87.7%** of splits at α=0.10 (and
**76.5–98.3%** at α=0.15). Decomposed, the genuine dilution term — extra accepted mass from
λ̂ actually moving — is only **+0.20 to +1.84 pp** (α=0.10) and **+0.00 to +0.65 pp**
(α=0.15). The rest of the gain is the repaired items being emitted, which happens
automatically whenever λ̂ does not fall and is not leverage at all. Honest multipliers are
**1.10–2.12×** (α=0.10) and **1.01–2.38×** (α=0.15), not 1.9–3.4×. On two cells the dilution
term is **not statistically distinguishable from zero** (POPE-adv LLaVA α=0.15, p=0.41;
AMBER α=0.15, p=0.32).

**C6. The self-repair numbers in §4.4 are wrong and must be replaced (§15).** The manuscript
quotes −43.9 / −34.7 / −77.5 / −3.9 with aborts 65 / 88 / 100 / 25%. Measured from the new
script: **−49.3 / −37.3 / −76.9 / −6.6** with aborts **73.0 / 88.8 / 98.8 / 25.2%**. These
match the reviewer's independent reimplementation almost exactly. Also: the AMBER self-repair
gain is **+0.9 pp with p = 0.25 — not distinguishable from zero** — so "self-repair does gain
on AMBER (+1.3 points)" cannot be stated as a fact at α=0.10 (it *is* solid at α=0.15, +2.9,
p=6e−78).

**C7. Qwen2-VL at α=0.15 now shows a significantly NEGATIVE grounding gain (§6).**
`local_backbone_analysis.py`: **−0.50 pp, 95% CI [−0.85, −0.14], p = 0.0061**. The paper's
"on the more accurate Qwen2-VL the same addition is worth 2.0 points" is not supported at that
risk level. At α=0.10 the gain is positive, +3.10 pp [+1.91, +4.30].

**C8. Abort rates must appear in Tables 4, 6 and 14 or those tables mislead (§4, §6, §9).**
Table 4's baseline aborts on **75.2%** of splits (self-consistency-only on **100%**);
Table 6's MME-existence and HallusionBench rows are **100%** abort; Table 14's
confidence-only column aborts on **65.8%** of splits at α=0.05 and **32.5%** at α=0.10.
Reporting a mean coverage of "11.0%" without saying it is three-quarters zeros is not a
defensible presentation.

**C9. Table 13's q-sensitivity summary is too kind (§10).** The paper reports q=0.10 as
"positive in all eight cells (+0.5 to +8.0)". Canonically, over all **ten** cells (both α, all
five settings including AMBER), q=0.10 is positive in **8 of 10**, range **−11.90 to +7.59**.
The two negatives are the AMBER cells. The eight-cell count silently excludes the loss.

### 2.1 Abstract, intro and §4.6 claims — every quantity, old → new

These are the sentences that need editing, not just the tables.

| where | claim as written | **canonical value** | verdict |
|---|---|---|---|
| Abstract, §1.4, §4.6, Fig 7B title | "$1.9$–$3.4\times$ leverage multiplier" | **1.10–2.12×** (α=0.10), **1.01–2.38×** (α=0.15); dilution-only term **+0.20 to +1.84 pp** / **+0.00 to +0.65 pp** | **rewrite** (C5) |
| Abstract, §1.4 | "$0.9$–$1.8\%$ of items repaired" | **0.97–1.97%** (α=0.10), **0.35–1.36%** (α=0.15) | widen; state α |
| §4.5 | "$0.9$–$1.8\%$ … buys $1.9$–$4.7$ points" | buys **+2.03 to +3.59 pp** (α=0.10, both-certify) | rewrite |
| Abstract | gains "$+1.9$ to $+8.0$" | **+0.40 to +7.59** | narrow |
| §5.3 | gains "$+0.5$ to $+8.0$ where the diagnostic is satisfied" | **+0.40 to +7.59** | narrow |
| Abstract | "up to $10.2$ points" | Table 6's largest grounding gain is now **+45.63 pp** (VCD); largest CCRC gain **+7.59 pp** | the two must not be conflated — one is a *score* gain, the other a *repair* gain |
| §5.1 | detection adds "$+8.4$ points against $+2.3$ for CLIP" | **+27.01 pp** vs **+21.76 pp** | **rewrite** (C1) |
| §5.1 | "at α=0.05 it nearly doubles usable coverage ($26.4\%\to47.1\%$)" | **9.0% → 27.3%**, a **3.0×** ratio; abort **65.8% → 24.0%** | rewrite (C8) |
| §5.1 | "on Qwen2-VL the same addition is worth $2.0$ points" | **+3.10 pp** at α=0.10; **−0.50 pp, p=0.0061** at α=0.15 | **restrict to α=0.10 and report the reversal** (C7) |
| §5.1 | detection grounding adds "$11.9\pm21.1$" on POPE-adv | **+23.84 ± 24.87**, CI [+21.39, +26.28] | update; keep the underpowered caveat, recast as abort |
| §5.1 | "The POPE-1500 result ($+9.6\pm4.7$) is the one we would rely on" | **+27.01 ± 28.66** | update; the sd is now larger, not smaller |
| Remark 2 (`rem:sep`) | detector "certifies far less coverage ($59.0\%$ vs $68.2\%$)" | filtering side is **68.6%**; detector side see §11 | update once §11 lands |
| §4.4 | self-repair "$-43.9$, $-34.7$, $-77.5$, $-3.9$" | **−49.3, −37.3, −76.9, −6.6** | **replace** (C6) |
| §4.4 | "aborts on $65\%$, $88\%$, $100\%$, $25\%$" | **73.0%, 88.8%, 98.8%, 25.2%** | **replace**; note 98.8 ≠ 100 |
| §4.4 | "on AMBER … self-repair does gain ($+1.3$ points at α=0.10)" | **+0.9 pp, p=0.25, not significant**; α=0.15 **+2.9 pp, p<1e−70** | **rewrite** (C6) |
| §4.4 | bottom-2% accuracy "$20.9\%$ / $44.6\%$" | **22.1% / 43.8%** | update |
| §4.4 | bottom-2% "Qwen … above chance at $64.3\%$" | **62.4%** | update; conclusion holds |
| §4.4 | "VCD at $2.1\%$, AMBER at $0.0\%$" | **4.9% / 0.9%** | update; conclusion holds |
| §5.2 | "subsampling POPE to $n=228$ still gains $+3.2$ points ($+5.6$ at 400, $+10.3$ at 700)" | wrong quantity (grounding gain, not CCRC gain); CCRC at n=228 **loses −3.65 pp** | **delete the argument** (C4) |
| §5.2 | AMBER: "CCRC *loses* $7.5$ points" | **−10.20 pp** (α=0.10), **−11.90 pp** (α=0.15) | update |
| §5.2 | "the cause is the absence of headroom" | conditional gain **+2.17 / +0.47 pp**; abort **13.0% / 12.8%** vs **0.0%**; **r_r < r_a** | **replace the diagnosis** (C3) |
| §5.4 | VCD "certified coverage from $0.4\%$ to $45.0\%$"; Table 6 says $21.8\to53.9$ | **1.7% → 47.3%**, gain **+45.63** | one value now |
| §5.4 | VCD "AURC from $0.1039$ to $0.0570$" | **0.1031 → 0.0630** | update |
| §7.1 | "Every reported number uses a three-way disjoint split with fixed-sequence testing and Clopper–Pearson bounds" | **now true** (it was not) | keep, and it is finally accurate |
| §7 | validity: exceedance "$11$–$15\%$ (test fold), $1$–$4\%$ (full set)" | **3.1–22.3%** and **0.0–5.1%** over the 40-cell grid | widen both (C-adjacent) |
| §1 | "95% confidence" (δ=0.05) while experiments use δ=0.10 | k_min would be **29**, not 22 | still needs fixing in prose |
| Table 13 caption | q=0.10 "positive in all eight ($+0.5$ to $+8.0$)" | **8 of 10**, **−11.90 to +7.59** | **rewrite** (C9) |

**C10. "Under a three-way split all combiners are statistically tied" is false (§11).**
Logistic regression beats gradient boosting by **11.90 pp** (p=8e−18) and random forest by
**15.41 pp** (p=2e−20) under the canonical protocol, because those combiners **abort on 14.5%
and 24.8%** of splits against logistic's **0.0%**. The recommendation survives; the reason
changes. Separately, the naive protocol's violation is far worse than reported when measured
correctly: **93.5%** of splits exceed α for gradient boosting (the paper quotes a 13.3% mean
risk), and random forest's `exc(all)` of 6.8% is an artefact of the contaminated evaluation
set — its `exc(te)` is **72.5%**.

Two further items that are digit-level but load-bearing:

- **Table 12's Clopper–Pearson row was never 71.6%** under any protocol I can reproduce; see
  §11. The manuscript's 71.6 is the third distinct value for the same quantity.
- **`fig_precondition.png` panel B's title** ("gain is 1.9–3.4× the repaired mass") is
  contradicted by §13 and has been rewritten in `make_paper_figures.py`.

---

## 3. Table 3 `tab:score` — the correctness score (POPE-1500, LLaVA, α=0.10)

`local_analysis_owlv2.py`, 400 paired three-way splits.

| Score | AURC old | **AURC new** | cov@10% old | **cov@10% new** | AUROC old | **AUROC new** | **abort** | **exc(te)** | **exc(all)** |
|---|---|---|---|---|---|---|---|---|---|
| Raw confidence | 0.0798 | **0.0774 ± 0.011** | 62.0% | **20.2 ± 27.6%** | 0.762 | **0.7683 ± 0.022** | **59.2%** | 11.7% | 2.5% |
| Learned, confidence only | 0.0701 | **0.0682 ± 0.009** | 63.6% | **41.6 ± 29.5%** | 0.782 | **0.7856 ± 0.020** | **32.5%** | 14.8% | 3.0% |
| + CLIP grounding | 0.0650 | **0.0637 ± 0.008** | 66.0% | **63.4 ± 7.4%** | 0.796 | **0.7987 ± 0.019** | 0.5% | 12.1% | 3.8% |
| **+ detection grounding** | 0.0566 | **0.0554 ± 0.008** | **73.1%** | **68.6 ± 9.0%** | 0.829 | **0.8320 ± 0.019** | 0.8% | 14.9% | 3.5% |
| + both | 0.0559 | **0.0547 ± 0.008** | 74.1% | **70.7 ± 5.6%** | 0.831 | **0.8343 ± 0.018** | 0.0% | 16.0% | 3.2% |

**Deltas:** raw confidence **−41.8 pp**; learned confidence **−22.0 pp**; +CLIP −2.6;
+detection **−4.5**; +both −3.4. AURCs and AUROCs are essentially unchanged (they never
depended on the threshold rule) — only the *certified coverage* column moves, which is the
column the rule governs.

Paired gains over `learned_conf` (identical splits, two-sided):

| gain | mean | sd/split | 95% CI | p(t) | p(Wilcoxon) |
|---|---|---|---|---|---|
| + CLIP grounding, cov (pp) | **+21.76** | 28.88 | [+18.93, +24.60] | 5.8e−41 | 5.7e−50 |
| + detection grounding, cov (pp) | **+27.01** | 28.66 | [+24.19, +29.83] | 3.8e−57 | 2.8e−66 |
| + both, cov (pp) | **+29.06** | 28.83 | [+26.23, +31.89] | 7.5e−63 | 3.6e−67 |
| + CLIP grounding, AURC (×1e−3, lower better) | **+4.43** | 2.33 | [+4.21, +4.66] | 3.8e−135 | 8.9e−66 |
| + detection grounding, AURC (×1e−3) | **+12.72** | 2.91 | [+12.43, +13.00] | 4.3e−262 | 2.7e−67 |
| + both, AURC (×1e−3) | **+13.49** | 3.28 | [+13.16, +13.81] | 3.6e−252 | 2.7e−67 |

➜ **This is C1.** Replace "+8.4 points against +2.3 for CLIP" with **"+27.0 pp against
+21.8 pp"**, and say why: the confidence-only baseline fails to certify anything on a third of
splits.

**The mechanism, verified exactly.** For n=1500 the calibration fold is 500 items, and
`lam0 = 1.5·k_min/n_cal = 1.5·22/500 = 0.0660`, so the **first hypothesis emits k ≈ 33**
(measured mean k = 33.7). Clopper–Pearson at k = 33 tolerates **zero** errors:

| e | `cp_upper(e, 33, 0.10)` | passes α=0.10? |
|---|---|---|
| 0 | **0.0674** | yes |
| 1 | **0.1128** | **no** |
| 2 | 0.1533 | no |

For raw confidence, the top 33 calibration items contain **at least one error on 59.2%** of
splits (mean e = 0.71) — and the **measured first-hypothesis failure rate is 59.2%, identical
to the measured abort rate**. So the abort is *entirely* decided by whether the single
strictest hypothesis passes. Because **feasibility is not a prefix in λ** for a weak score, the
fixed sequence must stop there (§ Remark 3's exact condition), whereas the old rule skipped
ahead to a large-k threshold where dozens of errors are tolerable — the multiplicity it never
paid for. This is the empirical face of the paper's own Remark 3, and it deserves a paragraph:
it is the most interesting consequence of the whole correction.

---

## 4. Table 4 `tab:selfcons` — self-consistency baseline (POPE-adv, n=450, α=0.10)

`local_selfconsistency.py`, 400 paired splits. LLaVA acc 0.827.

| Score | AURC old | **AURC new** | cov@10% old | **cov@10% new** | **abort** | **exc(te)** | **exc(all)** |
|---|---|---|---|---|---|---|---|
| Self-consistency only | 0.147 | **0.1366 ± 0.035** | 0.0% | **0.0 ± 0.0%** | **100.0%** | n/a | n/a |
| Confidence + self-consistency | 0.082 | **0.0770 ± 0.020** | 45.8% | **11.0 ± 21.1%** | **75.2%** | 14.1% | 3.0% |
| **+ detection grounding (ours)** | 0.063 | **0.0587 ± 0.016** | **57.7%** | **34.8 ± 27.5%** | **32.2%** | 10.3% | 1.8% |

Paired grounding gain: **+23.84 pp, sd 24.87, 95% CI [+21.39, +26.28], p(t) = 1.6e−58,
p(Wilcoxon) = 1.2e−42**. Old reported value: **+11.9 ± 21.1**.

**Deltas:** baseline **−34.8 pp**, ours **−22.9 pp**, gain **+11.9 pp larger** than reported.

➜ The manuscript's careful hedge on this cell ("a standard deviation that swamps it … we flag
this rather than report it as support") is now *too* pessimistic on the mean and *not
pessimistic enough* about the mechanism. The gain is large and its CI excludes zero
comfortably — but only because the baseline aborts on three-quarters of splits. **The right
statement is that the baseline cannot certify on this split at all, not that grounding adds
24 points of coverage to a working baseline.** n = 450 leaves a 150-item calibration fold
against k_min = 22; the abort column, not the sd, is the honest expression of that.

Note the label: this arm is **not** ConfLVLM. ConfLVLM's scorer is CLIP image–text
similarity (Table 3 row 3, Table 8 rung 1). Renamed "internal (conf+self-cons)" in the script.

---

## 5. Table 5 `tab:ccrc` — CCRC vs filtering (the flagship table)

`ccrc_gains_stats.py`, 400 paired splits, q = 0.10, δ = 0.10. **Same splits for both arms**,
so `gain` is a paired difference and the p-value is a paired test.

### α = 0.10

| Setting | n | μ | Filter old | **Filter new** | **ab%** | CCRC old | **CCRC new** | **ab%** | Gain old | **Gain new** | **sd** | **95% CI** | **p (2-sided)** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| POPE-1500 LLaVA | 1500 | 18.1% | 68.2% | **68.6 ± 9.0%** | 0.8% | 72.6% | **71.8 ± 6.0%** | 0.0% | +4.4 | **+3.18** | 6.23 | [+2.57, +3.79] | 7.5e−22 |
| POPE-adv LLaVA | 444 | 17.3% | 42.1% | **44.0 ± 24.8%** | **18.8%** | 46.8% | **47.2 ± 25.0%** | **17.5%** | +4.7 | **+3.19** | 15.61 | [+1.65, +4.72] | 5.4e−05 |
| POPE-adv Qwen2-VL | 591 | 12.9% | 77.5% | **77.9 ± 15.1%** | 0.2% | 79.4% | **79.5 ± 14.9%** | 0.8% | +1.9 | **+1.60** | 8.49 | [+0.76, +2.43] | 1.9e−04 |
| POPE-adv LLaVA+VCD | 591 | 19.6% | 45.0% | **48.8 ± 19.3%** | 5.8% | 47.5% | **51.0 ± 17.2%** | 4.0% | +2.5 | **+2.15** | 8.61 | [+1.31, +3.00] | 8.5e−07 |
| AMBER(d) LLaVA | 228 | 11.4% | 92.1% | **91.8 ± 5.1%** | 0.0% | 84.5% | **81.6 ± 31.7%** | **13.0%** | −7.5 | **−10.20** | 32.25 | [−13.37, −7.03] | 6.8e−10 |

### α = 0.15

| Setting | n | μ | Filter old | **Filter new** | **ab%** | CCRC old | **CCRC new** | **ab%** | Gain old | **Gain new** | **sd** | **95% CI** | **p** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| POPE-1500 LLaVA | 1500 | 18.1% | 86.8% | **86.4 ± 4.3%** | 0.0% | 88.4% | **88.4 ± 4.0%** | 0.0% | +1.7 | **+1.95** | 1.33 | [+1.82, +2.08] | 6.2e−102 |
| POPE-adv LLaVA | 444 | 17.3% | 69.2% | **69.1 ± 25.8%** | **9.5%** | 77.2% | **76.7 ± 15.4%** | 0.5% | +8.0 | **+7.59** | 21.31 | [+5.49, +9.68] | 5.0e−12 |
| POPE-adv Qwen2-VL | 591 | 12.9% | 94.4% | **94.8 ± 6.2%** | 0.0% | 94.9% | **95.2 ± 5.9%** | 0.0% | +0.5 | **+0.40** | 0.69 | [+0.33, +0.47] | 9.9e−27 |
| POPE-adv LLaVA+VCD | 591 | 19.6% | 73.4% | **75.8 ± 14.0%** | 0.8% | 75.4% | **77.2 ± 11.9%** | 0.2% | +2.0 | **+1.39** | 9.02 | [+0.50, +2.28] | 0.0022 |
| AMBER(d) LLaVA (new cell) | 228 | 11.4% | — | **95.9 ± 2.8%** | 0.0% | — | **84.0 ± 32.2%** | **12.8%** | — | **−11.90** | 32.41 | [−15.09, −8.72] | 1.2e−12 |

**Deltas.** Every positive gain shrinks: +4.4→+3.18, +4.7→+3.19, +1.9→+1.60, +2.5→+2.15,
+8.0→+7.59, +2.0→+1.39; only POPE-1500 at α=0.15 rises, +1.7→+1.95. The single loss deepens,
−7.5→**−10.20**. The manuscript's summary "gains are +0.5 to +8.0 points" becomes
**+0.40 to +7.59**, and the abstract's "+1.9 to +8.0" becomes **+0.40 to +7.59**.

**New and consequential: the abort column.** POPE-adv LLaVA at α=0.10 aborts on **18.8%
(filter) / 17.5% (CCRC)** of splits — nearly a fifth of deployments certify nothing. The
caption's "Realised test risk was ≤ α in every row" is *true but not the guarantee* and should
be replaced by the exceedance figures in §12.

**Caption fix.** The current caption cross-references a risk column in Table 14 that does not
contain one, and the sentence ends "and." mid-clause. §12 below is the risk reporting.

---

## 6. Table 6 `tab:datasets` — full sweep (α=0.10)

`master_comparison.py` (all rows) and `local_backbone_analysis.py` (Qwen detail),
400 paired splits.

| Dataset | Backbone | acc | grnd | cov old | **cov new (conf → +grounding)** | **abort conf** | **abort +g** | **exc(te)** | **exc(all)** |
|---|---|---|---|---|---|---|---|---|---|
| POPE (1500) | LLaVA-1.5 | 81.9% | yes | 63.6 → **73.1** | **41.6 ± 29.5 → 68.6 ± 9.0** | **32.5%** | 0.8% | 14.9% | 3.5% |
| POPE-adv (450) | LLaVA-1.5 | 82.7% | yes | 46.6 → **57.4** | **11.7 ± 22.4 → 36.5 ± 28.0** | **75.0%** | **29.8%** | 10.7% | 0.7% |
| POPE-adv (600) | Qwen2-VL-2B | 87.3% | yes | 83.5 → **85.5** | **75.9 ± 14.4 → 79.0 ± 13.2** | 0.0% | 0.0% | 9.2% | 1.0% |
| POPE-adv (600) | LLaVA+VCD | 80.3% | yes | 21.8 → **53.9** | **1.7 ± 8.7 → 47.3 ± 17.3** | **95.8%** | 5.0% | 9.5% | 0.5% |
| MME-existence (60) | LLaVA-1.5 | 95.0% | yes | "underpowered" | **0.0 → 0.0** | **100.0%** | **100.0%** | n/a | n/a |
| MME (full, 700) | LLaVA-1.5 | 68.9% | no | 5.7% | **2.6 ± 6.4%** | **85.0%** | — | 25.0% | 5.0% |
| GQA (yes/no, 700) | LLaVA-1.5 | 72.4% | no | 17.8% | **4.5 ± 11.1%** | **84.0%** | — | 23.4% | 7.8% |
| HallusionBench (447) | LLaVA-1.5 | 51.2% | no | 0.0% | **0.0 ± 0.0%** | **100.0%** | — | n/a | n/a |

Paired grounding gains (identical splits, two-sided):

| row | mean | sd/split | 95% CI | p(t) |
|---|---|---|---|---|
| POPE (1500) LLaVA-1.5 | **+27.01** | 28.66 | [+24.19, +29.83] | 3.8e−57 |
| POPE-adv (450) LLaVA-1.5 | **+24.87** | 27.26 | [+22.19, +27.55] | 1.6e−54 |
| POPE-adv (600) Qwen2-VL-2B | **+3.10** | 12.15 | [+1.91, +4.30] | 5.1e−07 |
| POPE-adv (600) LLaVA+VCD | **+45.63** | 18.17 | [+43.84, +47.41] | 1.5e−174 |
| MME-existence (60) | +0.00 | 0.00 | — | n/a (both arms abort on 100%) |

**Deltas:** the confidence-only column collapses everywhere (−22.0, −34.9, −7.6, −20.1, −3.1,
−13.3 pp) because it is the arm that aborts; the grounded column falls more modestly
(−4.5, −20.9, −6.5, −6.6). MME-existence is no longer "underpowered" as a hedge — it is
**100% abort, stated as such**, which is a cleaner disclosure and consistent with the
footnote's own arithmetic (U(3,60;δ)=0.108 > α).

**Cross-table reconciliation, and a warning.** POPE (1500)'s grounded arm here is **68.6%**,
identical to Table 3's "+ detection grounding" and Table 5's "Filter" — same four features,
same 1500 items, same rule, same splits. **That is C2.** But do **not** try to reconcile
Table 6's POPE-adv (450) row with Table 5's POPE-adv row: they are legitimately different
quantities. Table 6 uses **all 450 items** with ungrounded detector values imputed to the
median *and* includes `sc_yesfrac` as an extra feature; Table 5 uses the **444 grounded
items** and the four canonical features. Say so in the caption; this is one of the sources of
the cross-table drift.

### Qwen2-VL detail — `local_backbone_analysis.py` (n = 600, imputed)

| arm | AURC | cov@10% | abort | exc(te) | exc(all) | cov@15% | abort | exc(te) | exc(all) |
|---|---|---|---|---|---|---|---|---|---|
| Confidence only | 0.0347 | **75.9 ± 14.4%** | 0.0% | 9.8% | 1.8% | **95.9 ± 3.9%** | 0.0% | 8.0% | 0.0% |
| + grounding (ours) | 0.0310 | **79.0 ± 13.2%** | 0.0% | 9.2% | 1.0% | **95.4 ± 5.0%** | 0.0% | 8.8% | 0.0% |

| gain | mean | sd | 95% CI | p(t) |
|---|---|---|---|---|
| α=0.10 cov (pp) | **+3.10** | 12.15 | [+1.91, +4.30] | 5.1e−07 |
| α=0.15 cov (pp) | **−0.50** | 3.60 | **[−0.85, −0.14]** | **0.0061** |

➜ **This is C7.** At α=0.15 grounding *costs* Qwen2-VL half a point, significantly. The claim
"on the more accurate Qwen2-VL the same addition is worth 2.0 points" must be restricted to
α=0.10 and restated as **+3.10 pp [+1.91, +4.30]**, with the α=0.15 sign reversal reported.
This is the well-behaved end of the range — no aborts at all — so the negative result cannot
be explained away as an abort artefact.

---

## 7. Fig 5 `risk_coverage.png` — and the relabelled baseline

`risk_coverage_vs_conflvlm.py`, 400 paired splits, POPE n=1500, α=0.10.

| arm | calibrated cov old | **calibrated cov new** | oracle cov old | **oracle cov new** | **abort** | **exc(te)** | **exc(all)** |
|---|---|---|---|---|---|---|---|
| confidence only (model logits) | 63.6% | **41.6 ± 29.5%** | 71.2% | **71.6%** | **32.5%** | 14.8% | 3.0% |
| + structured grounding (ours) | 73.1% | **68.6 ± 9.0%** | 76.6% | **78.0%** | 0.8% | 14.9% | 3.5% |

| gain | mean | sd | 95% CI | p(t) |
|---|---|---|---|---|
| calibrated cov (pp) | **+27.01** | 28.66 | [+24.19, +29.83] | 3.8e−57 |
| oracle cov (pp) | **+6.41** | 2.73 | [+6.14, +6.68] | 9.6e−165 |
| AURC ×1e−3 (lower better) | **+12.85** | 2.95 | [+12.56, +13.14] | 6.8e−262 |

AURC baseline **0.0696**, ours **0.0567**. Ours ≤ baseline at **99%** of coverage levels.

**Presentation bug fixed (task 8.1).** The legend said *"ConfLVLM-style (internal
uncertainty)"*. That baseline uses **only LLaVA's own output logits** `[conf, p_yes]` — no
external evidence — so the label was wrong twice: ConfLVLM's scorer is CLIP image–text
similarity (external visual evidence), and this is not "internal uncertainty" in ConfLVLM's
sense. It is now **"confidence only (model logits)"** and ConfLVLM is not named anywhere in
the figure. The caption's own disclaimer ("Note that this baseline is not ConfLVLM") can now
be shortened, since the label no longer creates the confusion it was apologising for.

The legend also keeps **oracle** and **calibrated** coverage explicitly labelled and
side-by-side, since conflating them produced two of the nine POPE-1500 values (71.7/78.9 were
oracle numbers, never calibrated ones).

---

## 8. Fig 8 `comparison_figure.png` — panel A numbers and the header fix

`make_comparison_figure.py`, 400 paired splits.

| score | AURC | oracle cov | **calibrated cov** | **abort** | exc(te) | exc(all) |
|---|---|---|---|---|---|---|
| confidence only (model logits) | 0.0696 | 71.6% | **41.6 ± 29.5%** | **32.5%** | 14.8% | 3.0% |
| + CLIP grounding | 0.0651 | 72.8% | **63.4 ± 7.4%** | 0.5% | 12.1% | 3.8% |
| + OWLv2 grounding (ours) | 0.0567 | 78.0% | **68.6 ± 9.0%** | 0.8% | 14.9% | 3.5% |

**Presentation bug fixed (task 8.2).** Panel B's column headers collided illegibly: the
row-label gutter consumed 4 of the 9.2 x-units, leaving each of the 5 columns ≈ 11% of the
axis width — narrower than the header text itself. Fixed by narrowing the gutter to 2.9
units, wrapping headers one word per line at 8.0 pt, and adding explicit headroom
(`ylim` top `nR + 1.35`). Verified by rendering and inspecting the PNG: all five headers are
fully separated, no clipping, no overlap with the title or the footnote block. Panel A's
legend was also given `labelspacing=.9` so the three two-line entries do not run together.

---

## 9. Table 14 `tab:multialpha` — certified coverage vs α (POPE-1500)

`local_multi_alpha.py`, 400 paired splits. k_min per α: **45 / 22 / 15 / 11**.

| Score | α=0.05 old | **new** | α=0.10 old | **new** | α=0.15 old | **new** | α=0.20 old | **new** |
|---|---|---|---|---|---|---|---|---|
| Confidence only | 26.4% | **9.0 ± 13.7%** | 63.6% | **41.6 ± 29.5%** | 85.5% | **81.4 ± 6.5%** | 98.0% | **97.3 ± 3.0%** |
| + CLIP grounding | — | **13.9 ± 16.1%** | 66.0% | **63.4 ± 7.4%** | — | **82.5 ± 5.8%** | — | **97.2 ± 3.0%** |
| + detection grounding | 47.1% | **27.3 ± 20.4%** | 73.1% | **68.6 ± 9.0%** | 88.5% | **86.4 ± 4.3%** | 98.2% | **97.7 ± 2.4%** |

**Abort rate (%) — new, and the whole story of this table:**

| Score | α=0.05 | α=0.10 | α=0.15 | α=0.20 |
|---|---|---|---|---|
| Confidence only | **65.8%** | **32.5%** | 0.0% | 0.0% |
| + CLIP grounding | **49.8%** | 0.5% | 0.0% | 0.0% |
| + detection grounding | **24.0%** | 0.8% | 0.0% | 0.0% |

Paired gains over confidence-only:

| gain | mean | sd | 95% CI | p(t) |
|---|---|---|---|---|
| α=0.05 + CLIP | +4.88 | 14.07 | [+3.49, +6.26] | 1.7e−11 |
| α=0.10 + CLIP | +21.76 | 28.88 | [+18.93, +24.60] | 5.8e−41 |
| α=0.15 + CLIP | +1.10 | 2.70 | [+0.83, +1.36] | 5.2e−15 |
| α=0.20 + CLIP | **−0.06** | 1.12 | **[−0.17, +0.05]** | **0.272 (n.s.)** |
| α=0.05 + OWLv2 | +18.29 | 16.67 | [+16.65, +19.93] | 1.4e−70 |
| α=0.10 + OWLv2 | +27.01 | 28.66 | [+24.19, +29.83] | 3.8e−57 |
| α=0.15 + OWLv2 | +5.03 | 3.85 | [+4.65, +5.41] | 2.5e−88 |
| α=0.20 + OWLv2 | +0.48 | 1.67 | [+0.32, +0.65] | 1.7e−08 |

➜ The claim *"at α=0.05 it nearly doubles usable coverage (26.4% → 47.1%)"* becomes
**9.0% → 27.3%**, which is a **3.0× ratio, not a doubling** — a *stronger* statement, but the
honest mechanism is the abort rate: **65.8% → 24.0%**. The claim "at α=0.20 both scores
certify almost everything" survives (97.3 vs 97.7), and note the CLIP arm at α=0.20 is
**indistinguishable from the baseline** (−0.06, p=0.272), so "the grounded score dominates at
every level" (Fig 6 caption) is true for OWLv2 but **false for CLIP at α=0.20**.

---

## 10. Table 13 `tab:qsens` — repair-gate sensitivity

`ccrc_gains_stats.py`, gain over filtering, 400 paired splits, all **ten** cells.

| setting | α | q=0.10 | q=0.25 | q=0.50 |
|---|---|---|---|---|
| POPE-1500 LLaVA | 0.10 | **+3.18** [+2.57, +3.79] | +5.39 [+3.91, +6.88] | +8.84 [+7.67, +10.01] |
| POPE-1500 LLaVA | 0.15 | **+1.95** [+1.82, +2.08] | +5.38 [+5.19, +5.58] | +6.78 [+6.55, +7.01] |
| POPE-adv LLaVA | 0.10 | **+3.19** [+1.65, +4.72] | −0.05 [−2.94, +2.84] | +2.04 [−1.39, +5.48] |
| POPE-adv LLaVA | 0.15 | **+7.59** [+5.49, +9.68] | +10.14 [+7.69, +12.59] | +14.47 [+12.02, +16.92] |
| POPE-adv Qwen2-VL | 0.10 | **+1.60** [+0.76, +2.43] | −16.44 [−19.99, −12.89] | −14.86 [−18.33, −11.39] |
| POPE-adv Qwen2-VL | 0.15 | **+0.40** [+0.33, +0.47] | −1.22 [−2.60, +0.16] | +0.28 [−0.72, +1.29] |
| POPE-adv LLaVA+VCD | 0.10 | **+2.15** [+1.31, +3.00] | −6.35 [−9.22, −3.47] | −9.04 [−12.02, −6.06] |
| POPE-adv LLaVA+VCD | 0.15 | **+1.39** [+0.50, +2.28] | +0.48 [−1.00, +1.95] | +1.40 [+0.14, +2.67] |
| AMBER(d) LLaVA | 0.10 | **−10.20** [−13.37, −7.03] | −38.69 [−43.32, −34.06] | −25.91 [−30.23, −21.59] |
| AMBER(d) LLaVA | 0.15 | **−11.90** [−15.09, −8.72] | −11.55 [−14.69, −8.41] | −2.19 [−3.82, −0.56] |

| q | cells positive | range | worst cell | old summary |
|---|---|---|---|---|
| 0.10 | **8 / 10** | **−11.90 to +7.59** | **−11.90** | "positive in all eight (+0.5 to +8.0)", worst +0.5 |
| 0.25 | 4 / 10 | −38.69 to +10.14 | **−38.69** | "mixed", worst −16.3 |
| 0.50 | 6 / 10 | −25.91 to +14.47 | **−25.91** | "mixed", worst −15.5 |

➜ **This is C9.** q=0.10 is not positive in all cells once AMBER is included, and the worst
cells at q=0.25/0.50 are two to three times worse than reported. Also note the abort rates
this table hides: at q=0.25 the abort rate reaches **43.8%** (AMBER, α=0.10), **31.2%**
(POPE-adv LLaVA) and **26.2%** (VCD) — see §12. The conclusion "q must be strict" survives and
is in fact *strengthened*; the specific numbers must change.

---

## 11. Table 11 `tab:combiner` — combiner ablation

`combiner_ablation.py`, 400 paired reps, α=δ=0.10, 6 features. Both protocols share the same
held-out test fold, so they are directly comparable; protocol I fits *and* calibrates on the
same 2n/3 items.

### Protocol I — naive, fit == calibrate

| Combiner | cov old | **cov new** | **abort** | **exc(te)** | exc(all) | verdict old | **verdict new** |
|---|---|---|---|---|---|---|---|
| Logistic regression | 74.8% | **74.3 ± 2.9%** | 0.0% | **18.5%** | 0.8% | OK | borderline |
| Gradient boosting | 91.3% | **89.2 ± 1.8%** | 0.0% | **93.5%** | 16.2% | violated | **VIOLATED** |
| Random forest | 88.5% | **86.1 ± 2.2%** | 0.0% | **72.5%** | 6.8% | violated | **VIOLATED** |
| MLP (64,32) | ≈28% | **25.6 ± 25.2%** | **37.8%** | 14.9% | 0.0% | — | OK (but aborts) |

### Protocol II — three-way split, disjoint

| Combiner | cov old | **cov new** | **abort** | exc(te) | **exc(all)** | **verdict** |
|---|---|---|---|---|---|---|
| Logistic regression | 72.5% | **70.7 ± 5.6%** | **0.0%** | 16.0% | **3.2%** | OK |
| Gradient boosting | 72.5% | **58.8 ± 26.5%** | **14.5%** | 14.3% | **0.0%** | OK |
| Random forest | 75.7% | **55.3 ± 32.5%** | **24.8%** | 15.9% | **0.3%** | OK |
| MLP (64,32) | ≈28% | **14.2 ± 21.2%** | **60.8%** | 10.2% | **1.3%** | OK |

Paired differences vs logistic, within protocol (two-sided):

| comparison | mean | sd | 95% CI | p(t) |
|---|---|---|---|---|
| I: GradBoost − Logistic | **+14.88** | 2.62 | [+14.62, +15.14] | 8.9e−306 |
| I: RandForest − Logistic | **+11.75** | 2.45 | [+11.51, +11.99] | 8.0e−278 |
| I: MLP − Logistic | −48.71 | 25.34 | [−51.20, −46.22] | 3.1e−136 |
| **II: GradBoost − Logistic** | **−11.90** | 26.41 | **[−14.50, −9.31]** | **8.2e−18** |
| **II: RandForest − Logistic** | **−15.41** | 31.52 | **[−18.51, −12.31]** | **2.2e−20** |
| II: MLP − Logistic | −56.49 | 21.54 | [−58.61, −54.37] | 4.4e−181 |

Paired coverage price of the three-way split (II − I):

| combiner | mean | 95% CI | p(t) |
|---|---|---|---|
| Logistic | **−3.64** | [−4.12, −3.17] | 6.7e−41 |
| Gradient boosting | **−30.43** | [−33.03, −27.82] | 6.3e−75 |
| Random forest | **−30.80** | [−33.97, −27.63] | 3.8e−58 |
| MLP | −11.42 | [−14.68, −8.16] | 2.2e−11 |

> **⚠ C10 — a conclusion change.** The manuscript says *"Under a three-way split all combiners
> are statistically tied and all are valid."* **The first half is now false.** Logistic
> regression **significantly beats** gradient boosting by **11.90 pp** (p=8e−18) and random
> forest by **15.41 pp** (p=2e−20), because the high-capacity combiners **abort** — 14.5% and
> 24.8% of splits against logistic's **0.0%**. The paper's *recommendation* is unchanged and in
> fact strengthened; its stated *reason* must change from "equal efficiency" to **"logistic
> regression is the only combiner that reliably certifies anything at all: a high-capacity
> score is not merely no better, it fails the first hypothesis of the fixed sequence on a
> quarter of splits."** The sentence "a small multilayer perceptron collapses (≈28% coverage,
> high variance) from overfitting four features" should read **≈14% with a 60.8% abort rate**
> (and it is six features, not four).

**Also note the audit column swap.** Under protocol I, `exc(all)` is *not* a valid population
proxy — two thirds of the items were used to fit the score, so an overfit combiner looks
accurate on exactly those items. Random forest shows this starkly: `exc(all) = 6.8%` (passes)
against `exc(te) = 72.5%` (catastrophic). **For protocol I the honest column is `exc(te)`;**
`combiner_ablation.py` now sets its verdict accordingly and says so in the output. The
manuscript's "realised test risk is 13.3% and 12.1% against a 10% target" should be replaced by
the exceedance figures — **93.5% and 72.5% of splits exceed α** — which are far more damning
and are the actual guarantee being broken.

---

## 11b. Tables 7, 8, 12 — detector-only, attribution, risk bounds

*`missing_numbers.py` at REPS=400. See `RUN_STATUS` at the end of this file.*

---

## 12. Validity audit — `Pr[realised risk > α]` vs δ (task 4)

The guarantee is `Pr[Risk ≤ α] ≥ 1 − δ`. **Mean realised risk is not the guarantee**, neither
implies it nor is implied by it, and it is reported nowhere in the new output. The audit
statistic is a **tail fraction**, on two evaluation sets:

- `exc(te)` — held-out test fold. Unbiased, but noisy: n_te = n/3, so a single split's
  realised risk is a binomial proportion on a few hundred items and the fraction exceeding α
  is inflated by that noise even for a perfectly valid procedure.
- `exc(all)` — the selected policy re-scored on all n items. Low-noise **proxy for the
  population risk of the selected policy**; slightly optimistic because the fit and
  calibration folds are included. **This is the column that decides whether the guarantee
  holds.**

Aborted splits emit nothing, so have no risk, and are excluded from the denominators;
the abort rate is printed beside every exceedance figure so the reader can reconstruct.
(Counting aborts as risk-0 would only *lower* the exceedance.)

Full grid from `ccrc_gains_stats.py`, all settings × both α × all four arms — **every cell
verdict is OK, i.e. `exc(all) ≤ δ = 0.10`:**

| setting | α | arm | exc(te) | **exc(all)** | abort | verdict |
|---|---|---|---|---|---|---|
| POPE-1500 LLaVA | 0.10 | filter | 14.9% | **3.5%** | 0.8% | OK |
| POPE-1500 LLaVA | 0.10 | q=0.10 | 14.5% | **3.5%** | 0.0% | OK |
| POPE-1500 LLaVA | 0.10 | q=0.25 | 15.2% | **3.1%** | 3.2% | OK |
| POPE-1500 LLaVA | 0.10 | q=0.50 | 15.5% | **2.8%** | 1.5% | OK |
| POPE-1500 LLaVA | 0.15 | filter | 15.5% | **3.2%** | 0.0% | OK |
| POPE-1500 LLaVA | 0.15 | q=0.10 | 14.2% | **2.0%** | 0.0% | OK |
| POPE-1500 LLaVA | 0.15 | q=0.25 | 15.5% | **2.8%** | 0.0% | OK |
| POPE-1500 LLaVA | 0.15 | q=0.50 | 15.2% | **2.8%** | 0.0% | OK |
| POPE-adv LLaVA | 0.10 | filter | 10.8% | **2.5%** | 18.8% | OK |
| POPE-adv LLaVA | 0.10 | q=0.10 | 10.6% | **3.3%** | 17.5% | OK |
| POPE-adv LLaVA | 0.10 | q=0.25 | 15.3% | **4.0%** | 31.2% | OK |
| POPE-adv LLaVA | 0.10 | q=0.50 | 16.6% | **3.7%** | 32.2% | OK |
| POPE-adv LLaVA | 0.15 | filter | 14.1% | **0.8%** | 9.5% | OK |
| POPE-adv LLaVA | 0.15 | q=0.10 | 12.6% | **1.5%** | 0.5% | OK |
| POPE-adv LLaVA | 0.15 | q=0.25 | 12.5% | **1.3%** | 4.0% | OK |
| POPE-adv LLaVA | 0.15 | q=0.50 | 12.9% | **1.5%** | 1.5% | OK |
| POPE-adv Qwen2-VL | 0.10 | filter | 16.0% | **1.8%** | 0.2% | OK |
| POPE-adv Qwen2-VL | 0.10 | q=0.10 | 15.4% | **1.8%** | 0.8% | OK |
| POPE-adv Qwen2-VL | 0.10 | q=0.25 | 17.0% | **3.0%** | 23.8% | OK |
| POPE-adv Qwen2-VL | 0.10 | q=0.50 | 17.6% | **3.6%** | 23.5% | OK |
| POPE-adv Qwen2-VL | 0.15 | filter | 10.5% | **0.0%** | 0.0% | OK |
| POPE-adv Qwen2-VL | 0.15 | q=0.10 | 9.8% | **0.0%** | 0.0% | OK |
| POPE-adv Qwen2-VL | 0.15 | q=0.25 | 9.9% | **0.0%** | 2.0% | OK |
| POPE-adv Qwen2-VL | 0.15 | q=0.50 | 9.8% | **0.0%** | 1.0% | OK |
| POPE-adv LLaVA+VCD | 0.10 | filter | 13.0% | **1.9%** | 5.8% | OK |
| POPE-adv LLaVA+VCD | 0.10 | q=0.10 | 12.2% | **2.3%** | 4.0% | OK |
| POPE-adv LLaVA+VCD | 0.10 | q=0.25 | 16.3% | **3.1%** | 26.2% | OK |
| POPE-adv LLaVA+VCD | 0.10 | q=0.50 | 22.3% | **5.1%** | 36.0% | OK |
| POPE-adv LLaVA+VCD | 0.15 | filter | 12.6% | **2.0%** | 0.8% | OK |
| POPE-adv LLaVA+VCD | 0.15 | q=0.10 | 12.8% | **2.0%** | 0.2% | OK |
| POPE-adv LLaVA+VCD | 0.15 | q=0.25 | 14.1% | **1.5%** | 2.5% | OK |
| POPE-adv LLaVA+VCD | 0.15 | q=0.50 | 12.5% | **2.5%** | 1.8% | OK |
| AMBER(d) LLaVA | 0.10 | filter | 7.8% | **1.5%** | 0.0% | OK |
| AMBER(d) LLaVA | 0.10 | q=0.10 | 7.2% | **0.9%** | 13.0% | OK |
| AMBER(d) LLaVA | 0.10 | q=0.25 | 8.0% | **0.9%** | 43.8% | OK |
| AMBER(d) LLaVA | 0.10 | q=0.50 | 7.2% | **0.7%** | 30.2% | OK |
| AMBER(d) LLaVA | 0.15 | filter | 3.2% | **0.0%** | 0.0% | OK |
| AMBER(d) LLaVA | 0.15 | q=0.10 | 3.2% | **0.0%** | 12.8% | OK |
| AMBER(d) LLaVA | 0.15 | q=0.25 | 3.1% | **0.0%** | 12.5% | OK |
| AMBER(d) LLaVA | 0.15 | q=0.50 | 3.4% | **0.0%** | 3.0% | OK |

**Ranges to quote.** `exc(all)`: **0.0–5.1%** across all 40 cells (manuscript said 1–4%,
close enough that the claim survives with a widened range). `exc(te)`: **3.1–22.3%**
(manuscript said 11–15%; **the range must be widened**, and the VCD q=0.50 cell at 22.3% is
the worst). The guarantee holds everywhere on the population proxy.

---

## 13. NEW: dilution decomposition (review blocker 6)

**The question.** The manuscript claims a **1.9–3.4× leverage multiplier**: "0.9–1.8% of items
repaired at a strict evidence gate buys 1.9–3.4× that mass in certified coverage." Per-setting
ratios quoted: 3.36×, 2.44×, 2.78×, 1.90×.

**The decomposition.** A CCRC gain has two structurally different sources:

- **(i) the repaired items themselves.** These are emitted by construction. If λ̂ does not
  move, CCRC emits exactly filtering's accepted set *plus* the gated repairs, so the gain is
  mechanically equal to the repaired mass. This costs nothing, proves nothing, and is **not
  leverage**.
- **(ii) extra accepted items because λ̂ actually moved.** Repairs with `r_r < r_a` pull the
  mixture risk down, which can let the fixed sequence pass a *later* grid point, admitting
  more accepted mass. **This is the only part attributable to dilution.**

Computed per split as: (i) = repairs available at *filtering's own* λ̂; (ii) = total gain − (i),
which decomposes further into `d_acc` (extra accepted mass — the dilution payoff) and `d_rep`
(negative, because a more permissive λ absorbs items that were previously repairs).
Conditioned on both arms certifying; abort rates reported separately in §5.

### α = 0.10

| setting | splits | **λ̂ moves** | gain | **(i) repairs** | **(ii) dilution** | 95% CI (ii) | p(ii) | d_acc | d_rep | **multiplier** |
|---|---|---|---|---|---|---|---|---|---|---|
| POPE-1500 LLaVA | 397 | **28.0%** | +2.68 | +1.82 | **+0.86** | [+0.63, +1.09] | 2.3e−12 | +0.88 | −0.02 | **1.47×** |
| POPE-adv LLaVA | 313 | **29.4%** | +3.59 | +1.74 | **+1.84** | [+1.22, +2.46] | 1.1e−08 | +1.89 | −0.05 | **2.06×** |
| POPE-adv Qwen2-VL | 397 | **19.9%** | +2.03 | +1.02 | **+1.01** | [+0.43, +1.59] | 6.9e−04 | +1.04 | −0.03 | **1.99×** |
| POPE-adv LLaVA+VCD | 374 | **12.3%** | +2.05 | +0.97 | **+1.08** | [+0.62, +1.53] | 4.2e−06 | +1.25 | −0.17 | **2.12×** |
| AMBER(d) LLaVA | 348 | **15.2%** | +2.17 | +1.97 | **+0.20** | [+0.08, +0.31] | 9.0e−04 | +0.36 | −0.17 | **1.10×** |

### α = 0.15

| setting | splits | **λ̂ moves** | gain | **(i) repairs** | **(ii) dilution** | 95% CI (ii) | p(ii) | **multiplier** |
|---|---|---|---|---|---|---|---|---|
| POPE-1500 LLaVA | 400 | **23.5%** | +1.95 | +1.36 | **+0.60** | [+0.48, +0.71] | 7.9e−21 | **1.44×** |
| POPE-adv LLaVA | 362 | **18.8%** | +1.54 | +1.28 | **+0.26** | [−0.36, +0.89] | **0.41 (n.s.)** | **1.21×** |
| POPE-adv Qwen2-VL | 400 | **1.8%** | +0.40 | +0.35 | **+0.05** | [+0.00, +0.10] | 0.045 | **1.14×** |
| POPE-adv LLaVA+VCD | 396 | **12.6%** | +1.11 | +0.47 | **+0.65** | [+0.15, +1.15] | 0.012 | **2.38×** |
| AMBER(d) LLaVA | 349 | **1.7%** | +0.47 | +0.46 | **+0.00** | [−0.00, +0.01] | **0.32 (n.s.)** | **1.01×** |

**Findings — this is C5.**

1. **λ̂ is identical between filtering and CCRC on 70.6–87.7% of splits at α=0.10, and on
   76.5–98.3% at α=0.15.** On those splits there is no dilution effect at all, by definition.
   (The reviewer's "identical on 74–99%" is reproduced.)
2. **The genuine dilution term is +0.20 to +1.84 pp (α=0.10) and +0.00 to +0.65 pp
   (α=0.15).** The reviewer's "+0.01 to +1.76 pp" is reproduced. Everything else is term (i).
3. **Honest multipliers are 1.10–2.12× (α=0.10) and 1.01–2.38× (α=0.15), not 1.9–3.4×.**
   Note the identity: multiplier = 1 + (ii)/(i), so **any claim above 1× is a claim about
   term (ii) alone** — which is the right way to state it.
4. `d_rep` is small and negative everywhere (−0.00 to −0.17), confirming the nesting the
   method assumes: as λ opens, repairs are absorbed into the accepted set.
5. **Two cells have a dilution term indistinguishable from zero** (POPE-adv LLaVA α=0.15,
   p=0.41; AMBER α=0.15, p=0.32). On AMBER at α=0.15 the term is +0.00 pp and λ̂ moves on
   1.7% of splits: there is essentially no dilution there at all.

**How to restate the claim.** Proposition 2 (dilution lowers emitted risk pointwise in λ) is
correct algebra and survives. What does not survive is the leap from it to a "leverage
multiplier": Remark 3 already concedes that a larger feasible *set* yields a later stopping
index only if feasibility is a **prefix** in λ, which the paper declines to assume. The
measurement now shows how much that gap costs — **λ̂ fails to move on 71–98% of splits** — so
the mechanism is real but small, and the headline should be the repaired mass being emitted at
essentially no risk cost, with dilution as a **secondary, measured, sub-2-point effect**.

---

## 14. NEW: re-diagnosis of the AMBER negative result (review blocker 1)

`amber_diagnosis.py`, 400 paired splits, q=0.10.

### The two gain columns, read together

| quantity | **α = 0.10** | **α = 0.15** |
|---|---|---|
| filtering coverage | 91.76 ± 5.06% | 95.85 ± 2.77% |
| CCRC coverage | 81.57 ± 31.69% | 83.95 ± 32.22% |
| **UNCONDITIONAL gain** (what Table 5 reports) | **−10.20 pp** sd 32.25, CI [−13.37, −7.03], p=6.8e−10 | **−11.90 pp** sd 32.41, CI [−15.09, −8.72], p=1.2e−12 |
| **gain \| NEITHER arm aborts** | **+2.17 pp** sd 3.38, CI [+1.81, +2.53], p=6.0e−28, n=348 | **+0.47 pp** sd 0.70, CI [+0.39, +0.54], p=6.5e−30, n=349 |
| **abort rate, filtering** | **0.00%** | **0.00%** |
| **abort rate, CCRC** | **13.00%** | **12.75%** |
| **λ̂ moves** (both certify) | **15.23%** | **1.72%** |
| **r_a** accepted-region risk | **4.53%** | **7.45%** |
| **r_r** repair-region risk | **3.60%** | **0.00%** |
| dilution direction | **FAVOURABLE (r_r < r_a)** | **FAVOURABLE (r_r < r_a)** |
| repaired mass | 1.81% | 0.46% |

Reviewer's independent values: +2.06 / +0.49 conditional, ~12.8% abort, r_r=3.1% < r_a=4.4%.
**All four reproduced.**

### The arithmetic ledger — where the loss actually comes from

α = 0.10, contributions to the −10.20 pp unconditional gain, by split class:

| split class | splits | contribution |
|---|---|---|
| both arms certify | 348 | **+1.89 pp** |
| **CCRC aborts, filtering does not** | **52** | **−12.09 pp** ← the entire loss |
| filtering aborts, CCRC does not | 0 | +0.00 pp |
| neither certifies | 0 | +0.00 pp |

The CCRC-only-abort class alone accounts for **119%** of the reported loss at α=0.10 and
**103%** at α=0.15. Everything else is a *positive* contribution.

### Contrast — a setting where CCRC genuinely gains (POPE-1500)

| quantity | α=0.10 | α=0.15 |
|---|---|---|
| unconditional gain | +3.18 pp | +1.95 pp |
| gain \| neither aborts | +2.68 pp (n=397) | +1.95 pp (n=400) |
| abort filtering / CCRC | 0.75% / **0.00%** | 0.00% / **0.00%** |
| λ̂ moves | 27.96% | 23.50% |
| r_a / r_r | 7.96% / **0.24%** | 12.65% / **0.32%** |

Here CCRC **never** aborts and filtering occasionally does — the mirror image of AMBER.

### Sample-size probe

CCRC vs filtering at matched n, 400 paired splits per cell:

| setting | n | α | cov filt | cov CCRC | gain | abort filt | **abort CCRC** |
|---|---|---|---|---|---|---|---|
| AMBER(d) | 114 | 0.10 | 38.2% | 35.3% | −2.87 | 58.1% | **61.9%** |
| AMBER(d) | 152 | 0.10 | 71.1% | 84.1% | **+13.00** | 17.5% | **9.2%** |
| AMBER(d) | 190 | 0.10 | 87.5% | 82.7% | −4.79 | 0.0% | **10.2%** |
| AMBER(d) | 228 | 0.10 | 92.2% | 83.7% | −8.48 | 0.0% | **11.0%** |
| AMBER(d) | 114 | 0.15 | 92.9% | 93.1% | +0.21 | 0.3% | 1.5% |
| AMBER(d) | 152 | 0.15 | 95.4% | 94.7% | −0.75 | 0.2% | 2.0% |
| AMBER(d) | 190 | 0.15 | 96.4% | 87.3% | −9.15 | 0.0% | **9.8%** |
| AMBER(d) | 228 | 0.15 | 96.0% | 86.0% | −9.96 | 0.0% | **10.8%** |
| POPE-1500 | 114 | 0.10 | 0.2% | 0.2% | +0.00 | 99.8% | 99.8% |
| POPE-1500 | 152 | 0.10 | 6.7% | 6.1% | −0.62 | 90.2% | 91.5% |
| POPE-1500 | 190 | 0.10 | 18.4% | 15.3% | −3.07 | 71.5% | 78.0% |
| **POPE-1500** | **228** | **0.10** | 33.6% | 29.9% | **−3.65** | **43.5%** | **53.8%** |
| POPE-1500 | 456 | 0.10 | 40.5% | 55.4% | +14.99 | 31.2% | 12.5% |
| POPE-1500 | 912 | 0.10 | 64.1% | 70.0% | +5.88 | 4.0% | 0.0% |
| POPE-1500 | 1500 | 0.10 | 67.9% | 71.3% | +3.37 | 1.2% | 0.0% |
| POPE-1500 | 228 | 0.15 | 59.1% | 66.6% | +7.52 | 15.0% | 8.8% |
| POPE-1500 | 456 | 0.15 | 55.6% | 71.8% | +16.26 | 20.5% | 3.0% |
| POPE-1500 | 1500 | 0.15 | 86.0% | 88.0% | +1.96 | 0.0% | 0.0% |

➜ **This is C3 and C4.** Three separate facts contradict "the cause is the absence of
headroom":

1. **Conditional on both arms certifying, CCRC WINS on AMBER** at both α, decisively
   (+2.17, p=6e−28; +0.47, p=6.5e−30). A no-headroom story predicts a loss there too.
2. **r_r < r_a at both α** (3.60 < 4.53; 0.00 < 7.45), so by the paper's own Proposition 2
   dilution is in the **favourable** regime on AMBER. The "repairs consume slack" branch —
   which is what a headroom explanation needs — is *not* the branch AMBER is in.
3. **The loss is entirely the abort class**: filtering never aborts, CCRC aborts on ~13% of
   splits, and that class accounts for 103–119% of the loss.

And the manuscript's refutation of the sample-size explanation fails on both counts (C4):
the +3.2/+5.6/+10.3 figures measure the **grounding-score** gain from
`missing_numbers.py` BLOCK 6, not the CCRC gain; and CCRC-vs-filtering on POPE at n=228
**loses −3.65 pp** with a 53.8% abort rate.

**The mechanism to write.** At n=228 a three-way split leaves a 76-item calibration fold.
With k_min=22, `lam0 = 1.5·22/76 = 0.434`, so the very first hypothesis already emits ≈33
items — and Clopper–Pearson at k=33 tolerates **zero** errors. AMBER's filtering arm clears
this because the score is clean at the top; adding a fixed-mass repair block introduces the
occasional repair error into that first, smallest, most fragile hypothesis, and the fixed
sequence must stop at the first non-rejection. So the loss is a **small-calibration-fold
prefix-fragility effect specific to the fixed-sequence test**, not a property of AMBER's
headroom and not a weak repair channel (r_r is 3.6% / 0.0%). This is the same mechanism §4.4
already identifies for self-repair; AMBER is a mild case of it. Note it does **not** vanish
with n over AMBER's available range (11.0% at 228, 10.2% at 190, but 61.9% at 114): the
binding constraint is the interaction of fold size with the grid start, not n alone.

---

## 15. NEW: the self-repair experiment (task 7 — script now exists)

`self_repair_experiment.py`. Previously **no script in the repo produced these numbers**,
although they are the empirical basis for the paper's "a second channel is required" claim.

**Design, as the manuscript describes it.** The repair channel is the **negation of the
model's own answer** (binary yes/no, so the negated answer is right exactly when the model was
wrong: `ok_flip = 1 − ok_model`). The flip region is gated at a fixed **q = 0.02 on the
correctness score** — the bottom 2% of s. Certification is at the **mixture** level: one
canonical fixed sequence over accepted ∪ flipped at full δ, because a flip region whose own
risk exceeds α is not automatically inadmissible (Prop. 2's blend can absorb it).

### α = 0.10 — the authoritative numbers

| setting | filter cov | **self-repair cov** | **gain** | sd | 95% CI | p | abort filt | **abort self** | **acc in flip = r_r** |
|---|---|---|---|---|---|---|---|---|---|
| POPE-1500 LLaVA | 68.6% | **19.3%** | **−49.3** | 32.4 | [−52.5, −46.1] | 3.8e−106 | 0.8% | **73.0%** | 22.1% |
| POPE-adv LLaVA | 44.0% | **6.6%** | **−37.3** | 28.5 | [−40.1, −34.5] | 1.6e−88 | 18.8% | **88.8%** | 43.8% |
| POPE-adv Qwen2-VL | 77.9% | **1.1%** | **−76.9** | 17.7 | [−78.6, −75.1] | 1.4e−261 | 0.2% | **98.8%** | 62.4% |
| POPE-adv LLaVA+VCD | 48.8% | **42.2%** | **−6.6** | 23.7 | [−8.9, −4.3] | 4.4e−08 | 5.8% | **25.2%** | 4.9% |
| AMBER(d) LLaVA | 91.8% | **92.7%** | **+0.9** | 15.8 | **[−0.6, +2.5]** | **0.25 (n.s.)** | 0.0% | 2.8% | 0.9% |

### α = 0.15

| setting | filter cov | self-repair cov | gain | sd | 95% CI | p | abort self | acc in flip |
|---|---|---|---|---|---|---|---|---|
| POPE-1500 LLaVA | 86.4% | 53.8% | **−32.6** | 43.3 | [−36.9, −28.4] | 5.6e−41 | **39.2%** | 22.1% |
| POPE-adv LLaVA | 69.1% | 36.9% | **−32.2** | 41.4 | [−36.3, −28.1] | 4.8e−43 | **50.0%** | 43.8% |
| POPE-adv Qwen2-VL | 94.8% | 11.7% | **−83.1** | 32.4 | [−86.3, −79.9] | 7.1e−178 | **87.8%** | 62.4% |
| POPE-adv LLaVA+VCD | 75.8% | 77.7% | **+1.9** | 13.9 | [+0.6, +3.3] | 0.006 | 2.5% | 4.9% |
| AMBER(d) LLaVA | 95.9% | 98.7% | **+2.9** | 2.4 | [+2.6, +3.1] | 6.0e−78 | 0.0% | 0.9% |

Validity audit of the self-repair arm, and the flip block:

| setting | α=0.10 exc(te) | exc(all) | flip mass | r_flip | α=0.15 exc(te) | exc(all) | flip mass | r_flip |
|---|---|---|---|---|---|---|---|---|
| POPE-1500 LLaVA | 23.1% | 8.3% | 0.56% | 27.2% | 16.5% | 4.1% | 1.29% | 24.4% |
| POPE-adv LLaVA | 31.1% | 6.7% | 0.41% | 52.3% | 16.5% | 1.0% | 1.59% | 48.6% |
| POPE-adv Qwen2-VL | 40.0% | **20.0%** | 0.06% | 71.3% | 20.4% | 0.0% | 0.33% | 71.4% |
| POPE-adv LLaVA+VCD | 14.0% | 2.7% | 1.53% | 4.3% | 13.8% | 1.8% | 2.28% | 4.8% |
| AMBER(d) LLaVA | 7.5% | 1.5% | 3.15% | 0.9% | 3.0% | 0.0% | 2.85% | 0.6% |

(The Qwen `exc(all) = 20%` is on the ~5 splits of 400 that certified anything — n=5
denominators, not a validity finding. Where the abort rate is ~99% the exceedance columns are
uninterpretable and should not be quoted.)

### Comparison of all three measurements — this is C6

| setting | manuscript | reviewer's reimpl. | **this script (authoritative)** |
|---|---|---|---|
| POPE-1500 LLaVA | −43.9 | −49.3 | **−49.3** |
| POPE-adv LLaVA | −34.7 | −38.3 | **−37.3** |
| POPE-adv Qwen2-VL | −77.5 | −77.1 | **−76.9** |
| POPE-adv LLaVA+VCD | −3.9 | −6.1 | **−6.6** |
| abort rates | 65 / 88 / 100 / 25% | 73 / 88 / 99 / 22.5% | **73.0 / 88.8 / 98.8 / 25.2%** |
| AMBER gain | +1.3 | — | **+0.9 (p=0.25, n.s.)** at α=0.10; **+2.9 (p=6e−78)** at α=0.15 |

**My values agree with the reviewer's reimplementation, not with the manuscript.** Use
**−49.3 / −37.3 / −76.9 / −6.6** and **73.0 / 88.8 / 98.8 / 25.2%**.

Two further corrections to §4.4:

- **Qwen aborts on 98.8%, not 100%.** Do not write "100%" — a handful of splits certify.
- **The AMBER self-repair gain is not significant at α=0.10** (+0.9, CI [−0.6, +2.5],
  p=0.25). The sentence "on AMBER … self-repair does gain (+1.3 points at α=0.10)" must
  become **"+0.9 pp, not distinguishable from zero (p=0.25); at α=0.15 it gains +2.9 pp
  (p<1e−70)"**. The *qualitative* point the paper needs — that the budget in
  Prop. 4 admits self-repair when the flip region is clean — is supported by the α=0.15 cell
  and by the VCD α=0.15 cell (+1.9, p=0.006), so the argument survives with the right cells
  cited.

**Model accuracy in the bottom 2% of s (= r_r, the flip region's own risk):**

| setting | manuscript | **measured** |
|---|---|---|
| POPE-1500 LLaVA | 20.9% | **22.1%** |
| POPE-adv LLaVA | 44.6% | **43.8%** |
| POPE-adv Qwen2-VL | 64.3% | **62.4%** |
| POPE-adv LLaVA+VCD | 2.1% | **4.9%** |
| AMBER(d) LLaVA | 0.0% | **0.9%** |

All five confirm the paper's structural point: the two settings with a genuinely flippable
region (VCD, AMBER) are exactly the two with the least to gain, and on Qwen the bottom 2% is
**above chance**, so flipping there is actively harmful. The mechanism paragraph stands as
written; only the magnitudes change.

---

## 16. Where I could NOT reproduce the reviewer's numbers — read before citing

Honesty requires flagging these rather than quietly adopting the more convenient figure.

**16.1 The exceedance gap between the two protocols is much smaller than reported.**
The brief states the leaky rule has "population-risk exceedance of 7.0–9.6% against δ=0.10,
versus 2.0–5.9% for proper FST." I measure (§1.5, `exc(all)`, filtering arm, α=0.10):
**leaky 1.0–5.8%, fixed sequence 1.5–3.5%**; at α=0.15, **leaky 0.0–4.8%, FST 0.0–3.2%**. The
*ordering* holds on 9 of 10 cells and the leaky rule is clearly the more anti-conservative of
the two, but **neither rule empirically violates δ on any cell I ran.** I could not find an
evaluation set, α, or arm that produces 7.0–9.6%. Possibilities I cannot exclude: a different
feature set, δ, or a definition of "population risk" other than re-scoring the selected policy
on all n items. **Do not write that the old protocol broke the guarantee empirically.** Write
that it has no valid theory, contradicts §7.1's own protocol statement, is measurably more
anti-conservative, and concealed the abort behaviour — all of which is established here.

**16.2 The reviewer's "73.1% certified coverage where FST gives 68.2%" is close but not
exact.** Canonically FST gives **68.6%** and the leaky rule at 400 reps gives **70.8%**. The
73.1 in the manuscript is the leaky rule at **20** reps — i.e. part of that gap is Monte-Carlo
luck, not protocol. The manuscript's 68.2 is `ccrc_v3.py` at 60 reps; at 400 reps the same
code gives 68.6. So the single-source value is **68.6%**, and the "four different values"
(68.2 / 71.6 / 72.5 / 73.1) collapse to it.

**16.3 λ̂-identical fraction.** Reviewer: "identical on 74–99% of splits." I measure identical
on **70.6–87.7%** at α=0.10 and **76.5–98.3%** at α=0.15 — same conclusion, slightly wider at
the bottom end. Quote the two α ranges separately rather than a single 74–99%.

**16.4 Abort rate "up to 21.8%".** My maximum on the settings/arms corresponding to Table 5 is
**18.8%** (POPE-adv LLaVA, filtering, α=0.10). Higher values do occur at looser repair gates:
**43.8%** (AMBER q=0.25, α=0.10), **36.0%** (VCD q=0.50), **32.2%** (POPE-adv LLaVA q=0.50) —
and far higher on weak scores (Table 3's raw-confidence row, **59.2%**; Table 4's baseline,
**75.2%**). So "up to 21.8%" understates the worst cases rather than overstating them.

**16.5 p-values are conditional on the dataset.** Restated from §1.4 because it will otherwise
be read as a population claim: 400 splits of the same n items shrink the CI like 1/√400 while
dataset-level uncertainty does not shrink at all. Every p-value here is two-sided and tests
the split distribution, not replication on new data.

---

## 17. Presentation bugs fixed (task 8) — all three verified by rendering

| bug | fix | verified |
|---|---|---|
| `risk_coverage.png` legend said "ConfLVLM-style (internal uncertainty)" for a baseline that uses only LLaVA's logits | relabelled **"confidence only (model logits)"**; ConfLVLM no longer named in the figure; oracle vs calibrated coverage both labelled | ✔ read the PNG; legend correct, nothing clipped |
| `comparison_figure.png` panel B column headers collided illegibly | gutter 4.0 → **2.9** x-units, headers wrapped one word per line at 8.0 pt, `ylim` headroom to `nR+1.35`, figure widened to 16.0 in, panel A legend `labelspacing=.9` | ✔ read the PNG; all five headers separated, no overlap or clipping |
| `make_paper_figures.py` `fig_monotonicity` panel A and B titles overlapped | both titles wrapped to two lines at 10.5 pt, figure widened 9.6 → **10.4** in with explicit `wspace=.30`, y-limits raised so the value labels and annotation do not collide with the frame | ✔ read the PNG; titles fully separated |

Also regenerated `coverage_vs_alpha.png` under the corrected protocol (✔ read; clean, and the
error bars are now honestly enormous where the abort rate is high — that bimodality is real
and should not be smoothed away).

Two further figure items, not requested but necessary for consistency:

- **`fig_precondition.png` panel B rewritten.** Its title claimed "gain is 1.9–3.4× the
  repaired mass" and its hardcoded per-setting values came from the leaky rule at 60 reps.
  Panel B is now the **decomposition** of §13 — bars for term (i) and term (ii) with the
  multiplier annotated — and the title reads the measured range. Hardcoded values updated to
  the canonical α=0.10 numbers. Panel A retitled "a diagnostic, not a prediction", matching
  Remark 6.
- **`fig_monotonicity` p-value no longer hardcoded.** It was `p=0.025`; computed from the data
  it is **paired t p = 0.0247** (so the manuscript's 0.025 was right) and **Wilcoxon
  p = 0.0347**. Both are now printed on the panel, and the mean is **+0.2066**.

---

## 18. Miscellaneous numbers checked in passing

| claim | manuscript | **measured** | note |
|---|---|---|---|
| k_min at α=δ=0.10 | 22 | **22** | ✔ `⌈ln 0.10 / ln 0.90⌉` |
| k_min at α=0.10, δ=0.05 | "29, not 22" | **29** | ✔ the aside is correct |
| k_min at α=0.05/0.15/0.20, δ=0.10 | — | **45 / 15 / 11** | needed for Table 14 |
| k_min for loose bounds (α=δ=0.10) | 116 / 71 / 44 / 22 | **116 / 71 / 44 / 22** | ✔ Table 12's column reproduces |
| monotonicity mean paired difference | +0.207 | **+0.2066** | ✔ n=121 |
| monotonicity p | 0.025 | **0.0247** (t), **0.0347** (Wilcoxon) | ✔ |
| downstream hallucinated mentions | 1.207 → 1.413 | **1.207 → 1.413** | ✔ |
| VCD composition, certified coverage | "0.4% → 45.0%" (§5.4) vs "21.8 → 53.9" (Table 6) | **1.7% → 47.3%**, gain +45.63 [+43.84, +47.41] | the two manuscript versions of this now collapse to one |
| AMBER repaired mass | 1.45% (was hardcoded 0.0) | **1.81%** (α=0.10), **0.46%** (α=0.15) | α-dependent; quote both |
| AMBER detector accuracy | 95.6% | see §11 / `missing_numbers.py` BLOCK 2 | |
| HallusionBench certified coverage | 0.0% | **0.0%**, 100% abort | ✔ guarantee working as intended |

---

## RUN_STATUS

Everything above is final. The two long-running scripts feeding §11 are noted there;
re-run `python3 missing_numbers.py` and `python3 combiner_ablation.py` to fill Tables 7, 8,
11 and 12 if that section is still marked pending in your copy.

**Reproduce everything:**

```bash
cd conformal_followup
python3 protocol_audit.py          # §1.5   why the port was needed
python3 ccrc_gains_stats.py        # §5 §10 §12 §13  Tables 5, 13, audit, dilution
python3 amber_diagnosis.py         # §14    blocker 1
python3 self_repair_experiment.py  # §15    task 7
python3 local_analysis_owlv2.py    # §3     Table 3
python3 local_selfconsistency.py   # §4     Table 4
python3 master_comparison.py       # §6     Table 6
python3 local_backbone_analysis.py # §6     Qwen detail
python3 local_multi_alpha.py       # §9     Table 14 + coverage_vs_alpha.png
python3 risk_coverage_vs_conflvlm.py  # §7  Fig 5
python3 make_comparison_figure.py  # §8     Fig 8
python3 combiner_ablation.py       # §11    Table 11
python3 missing_numbers.py         # §11    Tables 7, 8, 12
python3 make_paper_figures.py      # §17    Figs 2, 4, 7
```

Set `CANON_REPS=6` to smoke-test any of them in seconds; leave it unset for reportable numbers.
