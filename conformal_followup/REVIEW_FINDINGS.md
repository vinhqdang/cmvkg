# Independent review of the TMLR manuscript — findings and triage

Four independent reviewers audited `manuscript_tmlr/main.tex` (31 pp) with separate
remits: numerical consistency, bibliography, mathematics/notation, and a cold TMLR read.
Three have reported (references pending). Everything below that is marked VERIFIED I
re-checked myself against the data; findings I could not confirm are marked as such.

## P0 — blocks submission

| # | finding | status |
|---|---|---|
| 1 | **Compiled as camera-ready, not anonymous.** `\usepackage[accepted]{tmlr}`, running head "Published in TMLR (07/2026)", author names/emails on p1, `\openreview id=XXXXXXXXXX`, acknowledgments "Placeholder for the camera-ready version". Paper also self-identifies ("our own earlier work \citep{cmvkgguard}", authored by us) and Fig. 7B labels the method "Ours (CMVKG-conformal)". | VERIFIED — desk reject as-is |
| 2 | **Theorem 2 does not prove the paper's main structural claim.** It states a *region-wise* certifiability condition (repair region certifiable iff model accuracy on it ≤ α), but CCRC certifies the **mixture**, and §5.5 says explicitly that repairs above α can be admitted if the blend complies. The words "independent channel" appear nowhere in the theorem or proof, yet abstract, Contribution 2, §2.7, §4.4 and the Conclusion all claim it is proved. Limitations even concedes independence is not needed for validity. | VERIFIED — claim unproven |
| 3 | **"3–6× leverage" is wrong.** Actual per-setting ratios: 3.36×, 2.44×, 2.78×, 1.90× → range **1.9–3.4×**. Three of four plotted points sit *below* Fig. 3B's own 3× reference line, while the panel title reads "gain is 3–6×". In the abstract, §1.4, §4.6 and the figure title. | VERIFIED by recomputation |
| 4 | **The repair gate is structurally one-directional.** `m = |owl − 0.15|`, so a maximally confident detector *"no"* attains m ≤ 0.15, while the q=0.10 gate sits at 0.357 / 0.366 / 0.366 / 0.209. Measured: **100% of gated repairs in all four settings are "detector says yes"**. CCRC as evaluated can only repair *missed objects* (false negatives) and can never repair the canonical false-positive existence hallucination that POPE-adversarial is built to elicit. Never disclosed. | VERIFIED — 45/45, 60/60, 60/60, 23/23 |

## P1 — substantive science

| # | finding | status |
|---|---|---|
| 5 | **The flagship negative result needs a material qualification.** Table 10's own unmentioned row: faithful mentions rise +0.174 alongside hallucinated +0.207, in slightly shorter continuations. Recomputed paired tests: per-**word** hallucination rises significantly (+0.0075, p=0.011), but hallucination **share of mentions** does not move at all (−0.004, p=0.884). So "repair makes the continuation worse" is true in absolute and per-word terms and **false** per-mention: repair makes the continuation more object-dense. The monotonicity conclusion still stands (absolute hallucinated claims is the right quantity for CHAIR-style risk), but the share result must be reported. | VERIFIED |
| 6 | **The empirical validity audit measures the wrong thing.** Code reports *mean* realised risk (`rsk = np.mean(R)`); the guarantee is `Pr[Risk ≤ α] ≥ 1−δ`. Mean ≤ α neither implies nor is implied by the quantile claim. **The guarantee itself is fine** — I audited it properly: population-risk exceedance is 1–4% against δ=0.10. (Raw test-fold exceedance is 11–15%, but that is finite-test-fold binomial noise, not a violation.) Fix the reported statistic, not the method. | VERIFIED — guarantee holds |
| 7 | **Self-certified repair: right conclusion, wrong mechanism.** Ran it at the mixture level. Coverage collapses: −43.9 / −34.7 / −77.5 / −3.9 points, driven by **total FST abort on 65% / 88% / 100% / 25% of splits**. The cause is not region-wise uncertifiability but that a fixed-mass, high-error repair block makes the early grid points (small k) fail, and FST must stop at the *first* failure → returns nothing → 0% coverage. Theorem 2 should be restated in mixture terms with this prefix/abort mechanism as the actual argument. | VERIFIED |
| 8 | **Tables 3, 4, 13, Fig. 6, Fig. 8, Fig. 9A use a two-way split with the combiner fitted on the calibration fold** (`local_analysis_owlv2.py`, `local_selfconsistency.py`, `local_multi_alpha.py`, `local_backbone_analysis.py`, `risk_coverage_vs_conflvlm.py`, `make_comparison_figure.py`) — the exact protocol §7.1 shows "silently breaks the guarantee" — while §7 claims "Every reported number uses a three-way disjoint split." Explains much of the cross-table drift below. | reviewer-VERIFIED, code-level |
| 9 | **The precondition does not "predict" the AMBER loss.** Prop. 3 is an upper bound on the gain; an upper bound of 7.9 pp cannot predict a loss of 7.5 pp. Worse, Table 5's Qwen α=0.15 row has μ=12.9% < α=15% (floor zero, precondition violated more starkly than AMBER) and yet **gains** +0.5; POPE-adv/LLaVA at α=0.15 has 2.3 pp headroom and gives the paper's **largest** gain (+8.0). Fig. 3A silently plots only the five α=0.10 points while the text says "across all settings". Also: ε in the proposition is *defined* as the excess over the floor, so the second inequality is an identity, and the μ→α consequence needs ε→0, which AMBER falsifies (floor 1.6%, 1−Cov_filt 7.9%). | VERIFIED (arithmetic + table) |
| 10 | **Theorem 3 assumes what §6 refutes.** Its first proof step is "on Rᶜ the output is unchanged" — exactly the non-interference the matched-prefix experiment falsifies. Its corollary also requires certifying the *uncorrected* output, the infeasible object the paper exists to avoid. Needs an explicit no-interference hypothesis plus an interference term (the measured +0.207). | reviewer-VERIFIED |
| 11 | **Prop. 2 (dilution) over-concludes.** Algebra is right, but it opens "Fix λ" and concludes about the whole feasible set; and "the certified λ̂ is at least as permissive" needs feasibility to be a *prefix* in λ, i.e. risk monotonicity — which §5 explicitly says is unavailable. | reviewer-VERIFIED |

## P2 — numerical drift (17 confirmed, 11 possible)

Worst offenders: POPE-1500/LLaVA certified coverage at α=0.10 appears as **63.6, 65.2, 68.2, 72.0, 72.5, 73.1, 73.6, 78.9, 79%** across Tables 3/5/6/7/8/11 and Figs. 5/7 with no reconciliation. VCD composition appears as 21.8→53.9 (Table 6) and 47.6→64.2 (§5.4 prose, sourced only to `RESULTS.md`, no script). AMBER LLaVA accuracy 78.6% (all 500 items) vs 88.6% (228 grounded) — both correct, both labelled identically. Abstract says "three backbones" (roadmap: two + a decoder); "up to 10.2 points" contradicted by Table 6's +32.1; gains "+1.9 to +8.0" vs Table 4/5's +0.5 to +8.0; "1.5% repaired × 3–6×" ⇒ 4.5–9.0 pp, above every measured value. Intro promises "95% confidence" (δ=0.05) while every experiment uses δ=0.10 — and at δ=0.05, k_min = 29, not 22. Table 2 caption says accuracy "degrades monotonically"; the printed deciles are non-monotonic in three places and the 7th decile (where it is worst) is omitted. Table 5 caption cites a "risk column" that does not exist; §7 claims risk is reported "in every table" (only 2 of 13 have it). Fig. 5 legend (71.7 / 78.9%) matches no table and calls ConfLVLM's scorer "internal uncertainty" while Table 8 calls it CLIP similarity and certifies 5.6%. `make_paper_figures.py:82` hardcodes AMBER repaired mass as 0.0 (measured 1.45%).

**Unsourced in the repo** (no script produces them): Table 7 detector-only column, Table 8 rows 1–2, Table 12 risk-bound ablation (no Hoeffding/Bernstein code exists), §5.4 VCD prose, the 13.3%/41.3% bottom-2%/decile accuracies (canonical v3 score gives 20.0%/42.7%), the subsample gains +3.2/+5.6/+10.3.

## P3 — notation and presentation

`s` is called a *nonconformity* score throughout but is a *correctness* score (accept when large). Four meanings of `p` (binomial parameter, p-value, model's P(yes), reported test p-values); `K`/`k` = emitted-set size and also self-consistency samples and top-k continuations; `Y` = correctness indicator while `Y*` = gold answer; `R` = calibration index subset and repair region and Risk; `E_λ` vs `𝓔_λ` differ only by font in the same display; τ orphaned. "Monotonicity" names two unrelated properties (risk-in-λ vs correction-map) with no qualifier in the abstract — inviting the reader to think §6 undermines Theorem 1, which it does not. "Fourth action" (§1.1) vs "third action" (§2.5). Roadmap and §2.5 both promise a re-derivation of Prop. 1 that the paper does not contain. Fig. 7B omits BCEA entirely, gives ConfLVLM "no grounding", and has clipped headers; Fig. 7A duplicates Fig. 5. The core positioning claim is restated five times; the endogenous-item-set point is duplicated near-verbatim 140 lines apart; contributions are enumerated three times with three different counts (four / five / four). Intro is 5 pp and related work 6 pp against 9 pp for method+experiments.

## What survives

The scaffolding checks out and I verified most of it independently: the abstention floor and every numeric instance of it; the one-sided Clopper–Pearson direction and `U(0,k;δ)=1−δ^{1/k}`; k_min=22 at α=δ=0.10; the quantile convention and the direction of λ; the decoupled-gate design in code matching the paper; the 3-way split in the canonical path; **the guarantee itself (population exceedance 1–4% vs δ=0.10)**; every value in Table 10 recomputed from the JSON; Table 5's nine rows, Table 3's fifteen cells, Tables 6/9/13 all reproduced by running the scripts; the AMBER negative result consistently negative in all five places; percentages vs points used correctly throughout; the qualitative figure exact on all six items. Both headline negative results are real measurements honestly reported, and neither the abstract nor the conclusion overclaims a sequence-level guarantee.

## Triage

The empirical contribution is intact. What is broken is the **claim–evidence chain**: three of seven formal statements need restatement, two headline numbers are wrong, one undisclosed scope restriction (one-directional repair), and the abstract overclaims against its own body in six places. None of this requires new GPU work; it requires a rewrite of the theory section, a single-source-of-truth pass over every number, and honest scoping.

## Bibliography review (4th reviewer, now reported)

84 of 87 entries independently confirmed against arXiv/DBLP/ACL/PMLR/CVF/Crossref.
`crcimposs` and `bcea` verified as correctly formatted (real per the abstracts supplied).
**No hallucinated references** and **no dangling citations** (85 keys cited, all 85 present).

**One entry not findable in any index:** `cmvkgguard` — our own `In press` self-citation
("Real-Time Hallucination Correction in VLMs Using Dynamic Knowledge Graph Verification",
Discover Artificial Intelligence). The journal is real; the article is unindexed, which is
expected for in-press. Add the DOI once assigned. It is load-bearing for the framing, so it
must not stay unresolvable at submission.

**One duplicate pair:** `mmhal` and `sun2024aligning` are the *same paper* (Sun et al.,
"Aligning Large Multimodal Models with Factually Augmented RLHF", Findings of ACL 2024,
13088–13110), cited separately as the MMHalBench benchmark and as Fact-RLHF. Renders twice.

**Three misdescriptions in the related-work prose:**
- `opera` — Retrospection-Allocation is a rollback that re-allocates **token selection**, not
  attention; and OPERA attributes over-trust to *summary tokens*, not "the language prior".
- `mcallava` — **not** a contrastive-decoding or layer-intervention method at all. It is a
  *positional encoding* method (2-D Manhattan spatial decay replacing RoPE's 1-D decay).
  Both mentions (§2.1 and the intro) are wrong.
- `chen2024halc` — HALC's focal contrast is over field-of-view **crops**; and
  instruction-contrastive decoding is a different, uncited paper.

**Metadata fixes (15):** `degf` author list truncated 10→4; `marine` uses the superseded
arXiv title (ICML 2025 is "…via Image-Grounded Guidance"); `mkgrag` pages should be
10767–10782; `amber` missing the "AMBER:" prefix and 2 of 11 authors, and is an `@inproceedings`
with an arXiv `booktitle`; `mme` missing 2 of 14 authors; `tecp` missing "TECP:" prefix;
`kumar2023` wrong workshop; `woodpecker`/`chen2023frugalgpt`/`tibshirani2019`/`vovk2003mondrian`
wrong entry types; and `only` (ICCV 2025), `maskcd` (EMNLP 2025 Findings), `mcallava` (ACM MM
2025), `causalgating` (ICML 2026), `anydist` (ICML 2024) are cited as arXiv preprints although
peer-reviewed versions exist.

**Every quantitative claim about a cited paper checked out**, including ConfLVLM's
87.8%→10.0% (and its CLIP/BiomedCLIP/LayoutLMv3 scorer breakdown), VCD's diffusion-noised
contrast, RLHF-V's correctional feedback, conformal factuality as back-off, KnowNo, and
Farquhar et al. in Nature 630:625–630.


---

# Resolution status (all findings addressed)

| finding | resolution |
|---|---|
| P0.1 anonymity | Fixed. `\usepackage{tmlr}`, camera-ready author block retained but inert, acknowledgments placeholder removed, OpenReview id neutralised, self-identifying prose rewritten third-person, figure label renamed CCRC. PDF reads "Anonymous authors". Verified: no institutional strings in the text layer. |
| P0.2 Theorem 2 | Replaced by Proposition 4, a mixture-level budget `c_r <= c_a(alpha-r_a)/(r_r-alpha)`, plus the two measurements (no cheap flip region where correction pays; FST aborts on 65-100% of splits). Claim downgraded to an empirical regularity with a mechanism; all 8 downstream overclaims rewritten. |
| P0.3 leverage 3-6x | Corrected to 1.9-3.4x in abstract, intro, method and figure title; figure reference lines 1x/2x/3x; AMBER repaired mass 0.0 -> 1.45%. |
| P0.4 one-directional gate | Disclosed in a new Remark 2, before the experiments, with the arithmetic and the 45/45, 60/60, 60/60, 23/23 counts. |
| P1.5 monotonicity | Competing explanation reported with both normalisations (share p=0.884, per-word p=0.011); conclusion retained on the grounds that the certified quantity is a count, and said so explicitly. Abstract's "any sequential procedure" -> "context-blind". |
| P1.6 validity audit | New "How validity is audited" paragraph: mean risk is not the guarantee; reports test-fold exceedance (11-15%) and full-set exceedance (1-4%) vs delta=0.10. `ccrc_v3.py` now flags on exceedance, not mean. Guarantee confirmed sound. |
| P1.8 two-way splits | All six scripts converted to three-way splits and re-run; tables 3, 4, 13 and three figures updated. This resolved the nine-coverage-values problem: the 71.7/78.9 pair were *oracle* coverages, never calibrated ones. Everything now reconciles on 63.6 -> 73.1. |
| P1.9 precondition | Prop 3 stripped of the epsilon identity; "predicts" downgraded to "diagnostic" everywhere; Remark 6 gives the counterexample from our own data. |
| P1.10 Theorem 3 | Non-interference now an explicit hypothesis; interference term added for the general case; two remarks report the measured +0.207 and the base-policy limitation. |
| P1.11 Prop 2 | Restated as pointwise/population; the step to a more permissive lambda-hat moved to Remark 3 naming both gaps. |
| Theorem 1 proof | "Exchangeable => binomial" replaced by the Poisson-binomial identification plus Hoeffding (1956) domination, so Clopper-Pearson conservatism is argued rather than assumed. Nesting no longer reads as a validity hypothesis. |
| Prop 1 | Derivation supplied, with the premise-dependent step marked as the one CCRC exits. |
| P2 numbers | Backbone count, the 10.2 ceiling, the +1.9 floor, Table 2's monotonicity caption, the nonexistent risk column, "every table", the 95%-confidence promise, the MME/Lemma-1 footnote, AMBER/Qwen accuracy labels, item 518's percentile and the frame description. Table 4's gain restated as +11.9+-21.1 with an explicit note that the interval does not support a significant effect. |
| P3 notation | Notation table added; s renamed a conformity/correctness score with orientation stated; error count E -> V; gold answer a*; "fourth action" -> third; the two monotonicity properties named separately; Algorithm 1 tests K >= k_min and defines the empty-lambda case. |
| Figures | Capability matrix gains BCEA and online abstention, corrects ConfLVLM's grounding row, renames columns to what they mean, widens for clipped headers. Risk-coverage caption distinguishes oracle from calibrated coverage and states the baseline is *not* ConfLVLM. |
| Bibliography | 15 metadata fixes applied and verified against arXiv/ACL/PMLR/CVF/DBLP; five understated venues corrected (ICCV/EMNLP-Findings/ACM-MM/ICML); four entry types fixed; `degf` author list 4 -> 10; the mmhal/sun2024aligning duplicate collapsed onto one key with prose naming MMHal-Bench; opera, mcallava and chen2024halc descriptions corrected (MCA-LLaVA is a positional-encoding method, not a layer intervention). |

Still open, and flagged rather than fixed: Table 12's risk-bound ablation has no
Hoeffding/empirical-Bernstein implementation in the repo, and the detector-only column of
Table 7 plus rows 1-2 of Table 8 are sourced only to `ALGORITHM.md`. These need either code
or removal before submission. `cmvkgguard` still needs a DOI.
