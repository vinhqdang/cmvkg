# Literature verification for the CCRC revision (Reviewer 1)

Verification date: 2026-10-04. Every field below was read from a page or file fetched on that date
(ACL Anthology page, BibTeX export and full PDF; arXiv abstract pages, HTML and PDF for each version;
Project Euclid; NeurIPS pages and proceedings; Crossref; Springer). Where a field could not be read it is
marked UNVERIFIED. Manuscript bib keys used: `angelopoulos2021ltt` (not `ltt`), `mme`, `crcimposs`, `bcea`,
`cmvkgguard` (all exist in `manuscript_neurocomputing/refs.bib`). New entries are in `new_refs.bib`.

---

## 1. CEBC (Mishra et al., ACL 2026)

### 1.1 Bibliographic data (ACL Anthology page and .bib export)

| Field | Value |
|---|---|
| Title | CEBC: Conformal Evidence-Bounded Control for Low-Hallucination Vision-Language Generation (en dash in the Anthology title) |
| Authors (order) | Ashish Mishra, Tarun Kumar, Arpit Shah, Suparna Bhattacharya, Martin Foltin |
| Affiliation (from PDF) | Hewlett Packard Labs (Bangalore for the first four authors, USA for Foltin); contact ashish.mishra@hpe.com |
| Venue | Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), San Diego, California, USA |
| Dates | Conference 2-7 July 2026; Anthology month = July 2026 (reviewer's "July 2026" is correct) |
| Pages | 46193-46206 (14 pages including appendix) |
| DOI | 10.18653/v1/2026.acl-long.2142 |
| URL | https://aclanthology.org/2026.acl-long.2142/ |
| Anthology id | 2026.acl-long.2142 |
| Publisher / ISBN | Association for Computational Linguistics; 979-8-89176-390-6 |
| arXiv id | None listed on the Anthology page; a web search for the title returned no arXiv version. Treat as "no arXiv id found". |

### 1.2 Technical summary (read from the full PDF, all 14 pages)

**What the method is.** A training-free, post-hoc pipeline around a base VLM caption or VQA answer.
Steps: (a) build a candidate pool (greedy, beam, K=6 sampled captions); (b) score each object mention with an
external detector; (c) select a candidate by lexicographic "risk first, then quality" (Eq. 7); (d) if the chosen
caption still has a mention below threshold tau, run a constrained rewrite by the VLM and then a deterministic
filter that drops any remaining object name with s_o < tau.

**Action space on an unsupported object mention.** Three things, none of them a certified action:
- Keep: if the whole selected caption has zero mentions below tau ("evidence-safe") it is returned unchanged.
- Revise or suppress: the VLM is prompted to (i) remove unverified object names and (ii) not introduce object names
  outside the allowed set A(x) = {o : s_o >= tau_desc}, with tau_desc = max(tau_min, tau - Delta), Delta = 0.05.
  The abstract says the step "minimally revises or suppresses unsupported object mentions". Qualitative examples
  show substitution ("scissors" becomes "knives"; "eating a car" becomes "eating a piece of cake") and insertion of
  up to m=2 "forced" verified objects (MAX_FORCE_OBJECTS=2). So the rewriter can introduce new content.
- Hard drop: final deterministic removal of any remaining mention with s_o < tau (the ablation shows this is
  necessary: without it CHAIR_I goes 1.62 to 2.60).
- VQA (POPE): CEBC_RISK flips Yes to No when detector evidence is below tau_lo; CEBC_BAL additionally flips
  No to Yes when evidence is above tau_hi. Flips% is 4.5-9.8 (COCO) and under 0.6 (GQA) for CEBC_BAL.

**Quantity whose risk is "controlled".** Evidence risk R_tau(x,y) = number of mentioned objects with s_o(x) < tau
(Eq. 2); a caption is "evidence-safe" if R_tau = 0. The calibration target, however, is different. Eq. 3-4: let H
be the multiset of detector scores of hallucinated mentions (mentions not in COCO ground truth) produced by the
base captioner on a calibration set; tau = clip(Quantile_{1-delta}(H), tau_min, tau_max). The paper's own gloss:
"only a small fraction (about delta) of hallucinated mentions from the base captioner would pass the evidence
test". That is a per-mention false-accept rate P(s >= tau | mention is hallucinated) of about delta. It is not
precision, not a false-discovery rate, not a selective risk over emitted claims, and not a caption-level
hallucination rate.

**Statistical guarantee: there is none stated.** There is no theorem, proposition or proof anywhere in the paper.
Specifically absent: a coverage or risk bound, an exchangeability or i.i.d. assumption, a confidence parameter
(probability 1-delta' over calibration draws), a finite-sample (n+1) correction, and any empirical check of the
realized false-accept rate or violation frequency. The text only describes the rule as a "conformal-style
calibration rule" (Sec. 3.3) and says in Sec. 2.3 that conformal prediction in general has finite-sample guarantees
(citing Tibshirani et al. 2019) and that "our approach provides a principled mechanism to bound hallucination risk
across datasets and base models". Delta is a "strictness" knob in the ablation (0.01, 0.10, 0.20 vs default
0.05), and the paper reports CHAIR changing monotonically with delta, not a validity check. The only conformal
reference in the bibliography is Tibshirani et al. 2019; CEBC does not cite conformal risk control, Learn-then-Test,
conformal factuality, ConfLVLM or BCEA.

**Do revised claims fall inside the guarantee?** No.
- Calibration set H is built from base (greedy) captions only. At test time the deployed text is a selected
  candidate (selection is an argmin over a pool using the same detector scores, so it depends on the very scores
  being thresholded) and then, in about 31-37% of images, a VLM rewrite. Rewrites and selected candidates are not
  exchangeable with the calibration mentions.
- The rewrite is allowed to use objects with s_o >= tau - 0.05, i.e. below tau; only the final deterministic filter
  re-imposes tau, and only on names in the fixed 80-class COCO vocabulary O (Mentions(y) is a lexical+POS pipeline).
- Attributes, relations and counts are not covered (stated in Limitations).
So the output is "detector-consistent" by construction, but no statement is made about the hallucination rate of
revised text. This is the main contrast with CCRC, whose guarantee is stated to cover the mixture of kept and
substituted answers.

**Role of the external detector.** It is the only evidence source and the sole arbiter of support:
DETR-ResNet50 (OWL-ViT v2 as a fallback) for CHAIR; for POPE a max-ensemble of DETR, OWLv2 and optionally CLIP with
weights 1.0, 1.0, 0.8; OWLv2 with phrase queries for OpenCHAIR. Ground truth is used only to label hallucinated
mentions when calibrating tau (COCO annotations); the Limitations section concedes this requirement and that
missed detections cause over-deletion. Inference (not stated in the paper): CHAIR is computed against COCO
annotations and DETR is trained on COCO, so detector and metric share a label space.

**Calibration or threshold-selection procedure.** A single scalar quantile; no grid, no fixed-sequence test, no
Bonferroni or other multiplicity correction, no search over alpha. Details: delta = 0.05 default; calibration on
the "first 120 images" (CALIB_N=120) for CHAIR with clamping to [0.25, 0.95] and a fallback tau = 0.80; for POPE,
thresholds calibrated separately per dataset on 250 held-out questions. The Table 1 footnote says "TAU=0.80;
Delta=0.05; calib_n=120", so it is unclear whether the headline CHAIR runs use the calibrated tau or the 0.80
fallback. The paper does not say that the 120 calibration images are disjoint from the 500 evaluation images
(both are described as the Karpathy test split). Inference: with CHAIR_S about 10% for the base captioner, 120
images contain roughly a dozen hallucinated mentions, so the 95% quantile rests on one or two points and the clamp
or fallback is probably active; the CALIB_N ablation (30/60/120/240) does not examine this directly.
"Default (Optimum) hyperparameters" in the Table 7 caption indicates settings were chosen from results on the
evaluation images.

**Benchmarks and models.** Models: LLaVA-1.5-7B, InstructBLIP, Qwen2-VL-7B-Instruct, Idefics2-8B. Data: MS-COCO
Karpathy test (500 images) for CHAIR and caption quality; POPE on MS-COCO and on GQA (1,000 questions each);
OpenCHAIR (random 500-image subset); GPT-4V-judged accuracy and detail. Baselines compared on LLaVA-1.5 only:
VADE, LURE, PMI, M3ID, OPERA, PAI, VisTexAttnAgg, greedy, beam. Overhead 1.58x runtime (0.526 to 0.830 s/image).

**Headline numbers.**
- CHAIR_S base greedy to CEBC: LLaVA 10.03 to 1.96; InstructBLIP 10.22 to 1.57; Qwen2-VL 8.31 to 1.80;
  Idefics2 9.03 to 2.02. CHAIR_I (LLaVA) 8.20 to 1.62. Recall rises (LLaVA 28.62 to 30.18). Rev.% 31.2-37.2.
- POPE-COCO (CEBC_BAL, Acc/F1): LLaVA 89.8/89.2 to 96.2/96.1; InstructBLIP 90.9/90.4 to 95.8/95.7;
  Idefics2 92.2/91.8 to 95.3/95.3; Qwen2-VL 92.4/91.9 to 96.1/96.1. GQA: changes of at most about +0.4 (LLaVA 83.1 to 83.4).
  An evidence-only baseline matches CEBC on COCO POPE (96.0 vs 96.2 for LLaVA) and degrades on GQA (83.1 to 80.8).
- Table 3 (LLaVA-1.5, vs. baselines): CHAIR_S 11.96, CHAIR_I 4.20, recall 45.80. These differ from Table 1
  (1.96/1.62) for the same model, and Table 3's greedy baseline (CHAIR_S 44.0) is not Table 1's (10.03); the paper
  does not explain the two protocols.
- OpenCHAIR OCH 0.4173 to 0.3789 (about 9.2% relative).

**Reliability observations to keep in mind when citing it** (all read directly from the PDF):
InstructBLIP shows Rev.% = 0.0 in Table 1 while its CHAIR_S falls from 10.22 to 1.57; Table 5 states that values
for non-evaluated VLMs "are set to 0 (placeholder)"; Table 7 row labels contain drafting residue ("measured; from
your tables", "requires small code toggles", "stricter; higher TAU expected") and round-number entries, so the
ablation rows other than the default may not be independent measurements. We would cite CEBC for its action space
and its calibration idea, not for its numbers.

### 1.3 Suggested contrast sentences for CCRC (factual, from the above)
1. Same pipeline shape (external detector, calibration set, edit unsupported object mentions); different
   objective: CEBC calibrates a false-accept quantile on hallucinated-mention detector scores, CCRC controls a
   risk with a finite-sample test.
2. CEBC states no theorem and does not measure violation frequency; CCRC states its bound and measures violations.
3. CEBC's revised text (selected candidate plus VLM rewrite) lies outside whatever its calibration covers;
   CCRC's guarantee covers the post-revision mixture.
4. CEBC uses one quantile with no multiplicity correction and an unclear calibration/evaluation split; CCRC uses a
   fixed-sequence test and reports a small-calibration failure mode that CEBC's 120-image calibration would be exposed to.

---

## 2. BCEA (arXiv 2606.16667)

### 2.1 Authors and version history (arXiv abstract page)

- Title: Look Again Before You Abstain: Budgeted Conformal Evidence Acquisition for Reliable Vision-Language Models
  (the arXiv listing title omits the final "s": "...Reliable Vision-Language Model"; the paper PDF has the plural).
- Authors: Jian Xu, Yanning Wu, Delu Zeng, John Paisley, Qibin Zhao. Subject: cs.CV. No journal-ref or comments field.
- v1: Mon 15 Jun 2026 13:02:50 UTC; v2: Wed 15 Jul 2026 07:10:17; v3: Mon 20 Jul 2026 13:54:25; v4: Tue 21 Jul 2026 10:42:21.
  The reviewer's "v4 on 21 July 2026" is correct.
- I downloaded and diffed all four versions (HTML text, and PDF text for v3 vs v4). v3 and v4 are textually
  identical except for the arXiv date stamp (the PDF diff is exactly two lines: the version/date banner). So the
  substantive changes happened in v2 (15 Jul) and v3 (20 Jul); "v4" content equals v3 content.
- Published venue: none listed.

### 2.2 What changed across versions

| Version | Claim count / models | What the headline result is |
|---|---|---|
| v1 | 1,440 existence claims, CLIP-guided; LLaVA-1.5, Qwen2.5-VL, LLaVA-NeXT, InternVL2 | Table 1: "Clopper-Pearson fixed-sequence procedure controls the selective risk with probability 1-delta"; reports coverage / 90th-percentile realized risk. BCEA coverage .22/.37/.47 at alpha .05/.10/.20 vs No-Acq .19/.28/.39, described as guaranteed. The Thm-1 proof scans the grid "in a fixed order and stop[s] at the first acceptance". |
| v2 | 3,000 one-per-image claims; Qwen2.5-VL replaced by Qwen3-VL | Exact variant (Bonferroni-corrected Clopper-Pearson over a fixed label-free grid Lambda) introduced and distinguished from a "practical" scan; practical variant shown to be anti-conservative (violation .21/.11/.16). Gated exact coverage .03/.31/.42 vs No-Acq .14/.23/.27. |
| v3 = v4 | same | Grid and gating band must be "specified independently of the calibration sample" ("calibration-independent", "not read from the calibration scores"); grid = 17 thresholds evenly spaced on [-4,4]; gated rule searches 7 bands x 17 thresholds under one Bonferroni correction; adds a "matched correction" No-Acq row; gated exact coverage .06/.32/.42. |

### 2.3 Exact certified method (v4 text, Theorem 1, Algorithm 1, App. B)

Fix the acquisition policy before seeing calibration data (post-acquisition score s_A depends only on (x,c) and the
model). With i.i.d. calibration claims, take a deterministic grid Lambda specified independently of the calibration
sample, compute a Clopper-Pearson upper bound at level delta/|Lambda| (Bonferroni) for the risk of the accepted set
at each t in Lambda, and take tau = the smallest t in Lambda whose bound is <= alpha. Guarantee:
Pr[R(tau_hat) > alpha] <= delta (probability over the calibration sample; delta = 0.1), selective risk on test claims.
The "Practical" branch (Algorithm 1, line 7) instead scans the calibration scores with the Clopper-Pearson bound
at the full level delta and no correction; the paper calls it "uncertified", "anti-conservative", and says "we do
not call it guaranteed". The reviewer's description is accurate.

### 2.4 Numbers (LLaVA-1.5-7B; COCO val2017 existence claims, 3,000 one-per-image claims, 300 random 50/50 splits, delta = 0.1)

Certified (exact) coverage, with empirical test-set exceedance 0.00 throughout (Table 1, v4):

| alpha | No-Acq exact | BCEA full-budget exact | BCEA gated exact | BCEA practical (uncertified) |
|---|---|---|---|---|
| 0.05 | .14 (matched-correction .04) | .07 | .06 | .32 (violation .21) |
| 0.10 | .23 | .22 | .32 | .43 (violation .11) |
| 0.20 | .28 | .41 | .42 | .52 (violation .16) |

- CLIP-guided full-budget under the exact certificate (App. C): .06/.23/.41 at alpha = .05/.10/.20, vs No-Acq .14/.23/.28.
  So at alpha = 0.10 the certified CLIP-guided coverage (0.23) equals No-Acq; it wins only at alpha = 0.20.
- The 28% to 37% at alpha = 0.10 is Table 2 of v4 (Table 4 in v2, Table 1 in v1): LLaVA, existence claims,
  **practical variant, "risk near alpha"**: coverage 0.28 (no acquisition) to 0.33 (uniform crop grid) to 0.37 (CLIP-guided);
  AUROC .824, .862, .882. At alpha = 0.05: .19/.18/.22; at alpha = 0.20: .39/.43/.47. The caption says "practical
  variant"; the text says this coverage "is measured under the practical variant". The reviewer is right: it is
  not the exact certified number. The certified analogues at alpha = 0.10 are 0.23 to 0.32 (gated) or 0.23 to 0.23 (CLIP-guided full budget).
- alpha = 0.15: **UNVERIFIED / not in the paper.** The alpha values evaluated are 0.05, 0.10, 0.20 only
  (plus delta sweeps in Table C.4). The only "0.15" in the paper is a Qwen3-VL practical coverage entry (0.15 at
  alpha = 0.05, Table C.5). I cannot supply certified-vs-practical numbers at alpha = 0.15.
- Qwen3-VL-7B (Table C.5, practical, existence): No-Acq/BCEA coverage .01/.15 (alpha .05), .13/.22 (.10), .22/.30 (.20).
- POPE (Table 3, practical, alpha = 0.10, 4 VLMs x 3 splits, ~11.5k claims) is explicitly "an empirical evaluation, not
  a finite-sample certificate" (claim-level splits). Example LLaVA-1.5 random: coverage .28 to .36. The certified,
  image-level version is Table C.1 (zero violations on all four backbones).
- Free-form captions (1,563 images, one random claim per image, 20.6% hallucinated): exact certificate at alpha = 0.10
  asserts 34% of self-generated claims vs 24% for abstention; practical variant at alpha = 0.05 is .30 with violation .16.
- Benchmarks and models: COCO val2017-constructed existence (3,000) and spatial claims (996), POPE
  random/popular/adversarial; LLaVA-1.5-7B, Qwen3-VL-7B, LLaVA-NeXT-7B, InternVL2-8B.

### 2.5 Which version our earlier text quotes

`main.tex` (Related work, "Budgeted conformal evidence acquisition") says BCEA uses "Clopper-Pearson p-values with
fixed-sequence testing" and "certified coverage rising from roughly 28% to 37% at alpha = 0.10", on "LLaVA and
Qwen backbones". That wording matches **v1 (15 Jun 2026)**: only v1 presents 0.28 to 0.37 as guaranteed coverage from a
fixed-sequence Clopper-Pearson procedure. The same numbers survive in v2-v4 but are relabelled "practical variant";
the certified (exact) figure at alpha = 0.10 in v4 is 0.23 (No-Acq) to 0.32 (gated). **Required fix:** replace
"certified ... 28% to 37%" with the exact numbers (0.23 to 0.32 gated, 3,000 one-per-image claims) and, if the
practical figure is quoted, label it "uncertified practical variant". Also in RESULTS.md/ALGORITHM.md the comparison
sentence "their published +9 pp at alpha = 0.10" corresponds to 0.28 to 0.37 (practical); the certified gain is +9 pp
only for gated at alpha = 0.10 (0.23 to 0.32), so state which.

---

## 3. "When Can Conformal Risk Control Certify LLM Outputs?" (arXiv 2606.29054, key `crcimposs`)

- Title: When Can Conformal Risk Control Certify LLM Outputs? Bounds, Impossibility, and Adaptation for Structured Generation
- Author: **Varun Kotte** (single author; "Independent Researcher", per the paper's first page). Subject cs.LG.
- Versions: v1 27 Jun 2026; v2 7 Sep 2026 (current as of today). DOI (arXiv-assigned): 10.48550/arXiv.2606.29054.
- Published venue: none. No journal-ref or comments on the arXiv page, and a web search found no venue.
  Cite as an arXiv preprint (v2).
- **Important: v2 changed content that our manuscript quotes (we quote v1).** I diffed v1 and v2 text.
  - Floor: v1 states abstain >= (mu - alpha)/(1 - alpha). v2 states the sharpened floor (mu - alpha)/(M - alpha)
    with M = ess sup R, which equals the (1 - alpha) version when M = 1 (true for the F1/exact-match losses in the paper).
    Our manuscript's formula remains valid, but cite it as the M = 1 instance.
  - Hierarchy gain: v1 "+37% certified configurations" across 656 configurations; v2 "+41%" across 716
    configurations (51/72/80 certified at strict targets). Our manuscript (main.tex ~line 651) says +37%.
  - Shift: v1 "ACI cuts risk-target violations from 71% to 21%" (our manuscript says this); v2 drops it and instead
    reports that under cross-dataset shift (mu > alpha) static CRC and every tested ACI step size violate the target on
    14/16 transfers, with an anytime-valid full-feedback monitor certifying 0/16.
  - Relaxed targets: v1 says relaxing alpha to 0.30-0.40 unlocks certification (47% NER, 40% QA, 60% CLS); v2: 28% NER,
    13% QA, 19% CLS at alpha = 0.40.
  - The sentence in our manuscript that Kotte "state[s] it plainly: the predictor may only emit the model's own answer
    or abstain" is a paraphrase; the paper's selective predictor "emits y_hat when s >= lambda and abstains otherwise"
    (v1 problem setup). Quote that rather than a quotation.
  Action: update the +37% to +41% and 656 to 716, and remove or re-source the 71%-to-21% ACI claim.

---

## 4. Final-publication metadata

### (a) Learn then Test (key `angelopoulos2021ltt`)
Verified at Project Euclid (issue table of contents and article page), Crossref (10.1214/24-aoas1998) and NSF PAR:
- Authors: Anastasios N. Angelopoulos, Stephen Bates, Emmanuel J. Candes, Michael I. Jordan, Lihua Lei.
- Title (journal casing): Learn then test: Calibrating predictive algorithms to achieve risk control.
- The Annals of Applied Statistics, vol. 19, no. 2, pp. 1641-1662, June 2025. Publisher: Institute of Mathematical Statistics.
- DOI 10.1214/24-AOAS1998. ISSN 1932-6157.
- Page range read from the Project Euclid issue listing; Crossref returned no page field.
- Our current entry (arXiv:2110.01052, 2021) should be replaced. The arXiv page itself still shows no journal-ref (last version v5, 29 Sep 2022).

### (b) MME (key `mme`)
Verified at the NeurIPS proceedings page, its BibTeX export, neurips.cc virtual site, and arXiv abstract page:
- The reviewer is right. Published in Advances in Neural Information Processing Systems 38 (NeurIPS 2025), **Datasets and
  Benchmarks Track**; neurips.cc lists it as a Spotlight poster (San Diego). arXiv v5 (24 Oct 2025) carries the comment
  "NeurIPS DB 2025 Spotlight".
- Authors unchanged (14): Fu, Chen, Shen, Qin, Zhang, Lin, Yang, Zheng, Li, Sun, Wu, Ji, Shan, He.
- DOI 10.52202/085713-4899; publisher Curran Associates, Inc. (per the proceedings BibTeX); proceedings page date 2026-04-23.
- Page numbers / article number: **UNVERIFIED (none given;** the proceedings BibTeX has an empty `pages` field).
- Caveat: the proceedings BibTeX writes volume as "38, Main Conference" and the page header says "Main Conference", but
  the URL, file name and neurips.cc entry all say Datasets and Benchmarks Track; I use D&B.
- Our current entry (arXiv:2306.13394, 2023) should be replaced.

### (c) Reference [22]
`main.bbl` numbering (the 22nd `\bibitem`) = **`cmvkgguard`**: Q.-V. Dang, N.-S.-A. Nguyen, T.-B.-D. Vo, M. N. Dinh,
A. D. Le, H. V. Vu, P. L. Nguyen, "Real-time hallucination correction in vision-language models using dynamic knowledge
graph verification", rendered as "Discover Artificial IntelligenceIn press (2026)." (This is the authors' own paper.)

- The formatting error: the journal name and the `note = {In press}` field are fused with no separator
  ("...IntelligenceIn press"). Cause: in `refs.bib` the entry is `@article` with `journal` and `note` but no `volume`;
  in `elsarticle-num.bst` the `article` function resets the output state after the journal when `format.vol.num.pages`
  is empty, so the next field (the note) is concatenated with no punctuation. A second problem is that the entry is
  stale: it is no longer "in press".
- A DOI exists. Crossref and Springer: 10.1007/s44163-026-01778-z, Discover Artificial Intelligence (ISSN 2731-0809),
  Springer, first online 7 August 2026 (received 12 Feb 2026, accepted 8 Jul 2026), CC BY-NC-ND 4.0. Authors as in the bib.
  Volume, issue and article number: **UNVERIFIED** (Crossref lists none and the Springer cite block shows none; it is
  cited as "Discov Artif Intell (2026)").
- Corrected entry (see `new_refs.bib`): drop `note`, add `doi` and `url`, keep `year = 2026`. Without a volume the
  `.bst` should then print "... Discover Artificial Intelligence (2026)" followed by the DOI link. I could not run
  BibTeX here (not installed), so recompile and check the entry in `main.bbl`; if a volume or article number appears on
  the Springer page later, add it.
- Also check the second citation of the same paper in `main.tex` (lines ~205 and ~534): the text only cites the key.

---

## 5. Other closely related 2025-2026 work not mentioned by the reviewer (6 verified)

All six were checked on their arXiv abstract page (and, for the first two, the paper PDF). None repairs VLM
hallucinations with an external detector under a certificate that covers the repaired output.

1. **Conformal Linguistic Calibration** (Jiang, Liu, Van Durme; NeurIPS 2025 main-conference poster; arXiv 2502.19110).
   Post-processor T(y) rewrites claims into less specific ones and guarantees P(T(y) is correct) >= 1 - alpha, so the
   rewritten output is inside the guarantee. This is the closest conceptual precedent for certifying a revised answer.
   Differences: text-only QA (SimpleQA, Natural Questions), generalization of a claim rather than substitution
   with an externally verified alternative, trained claim rewriter, no VLM or external detector.
2. **Conformal Language Model Reasoning with Coherent Factuality** (Rubin-Toles, Gambhir, Ramji, Roth, Goel; ICLR 2025).
   Split conformal over subgraphs of a deducibility graph so that retained reasoning steps are jointly supported.
   Differences: removal-only (retains >= 80% of claims at 90% factuality), math reasoning (MATH, FELM), no repair, no VLM.
3. **IntroConformal** (Atabuzzaman, Alexander, Thomas; arXiv 2609.01375, 1 Sep 2026; arXiv comment "EMNLP 2026 main
   conference", not yet checked against proceedings). CRC with conformity scores from the LVLM's own hidden states and
   self-verification probability, explicitly avoiding external verifiers. Differences: filtering/abstention only, internal
   signal vs our external detector, no substitution.
4. **Adaptive Conformal Prediction for Improving Factuality of Generations by LLMs** (Rubashevskii, Piatrashyn, Nakov,
   Panov; arXiv 2604.13991, Apr 2026, preprint). Prompt-adaptive conformal score transformation giving better conditional
   coverage for claim filtering. Differences: marginal and conditional coverage of a filter, text only, no repair action.
5. **Is Conformal Factuality for RAG-based LLMs Robust?** (Chen, Chen, Chikodikar, Yin, Vinayak; arXiv 2603.16817, Mar 2026,
   preprint). Shows conformal claim filtering gives vacuous outputs at high factuality levels and is not robust to
   distribution shift or distractors. Differences: empirical audit rather than a method; relevant as external support for
   CCRC's premise (filtering pays an informativeness price) and its shift caveats.
6. **Inference-Time Conformal Reasoning (ITCR)** (Wang, Shi, Yan, Zhang; arXiv 2606.08831, v2 28 Jun 2026; arXiv comment
   "Accepted at ICML 2026", not checked against proceedings). Moves calibration inside generation (a calibrated stopping
   rule over a reasoning graph with a nested-validity argument). Differences: intervenes by truncating generation,
   not by replacing content; text reasoning, no detector.

Also relevant but older (2024, so outside the window; BCEA's baseline): Srinivasan et al., "Selective 'Selective
Prediction': Reducing Unnecessary Abstention in Vision-Language Reasoning" (ReCoVERR), Findings of ACL 2024, pp. 12935-12948,
DOI 10.18653/v1/2024.findings-acl.767 (BibTeX included in `new_refs.bib` as optional).

Not found: any 2025-2026 conformal method that substitutes or edits unsupported VLM claims with a certified guarantee on
the edited output other than CCRC (CEBC edits but certifies nothing). This is a negative search result, not proof of absence.
