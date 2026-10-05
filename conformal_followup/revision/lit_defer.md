# Positioning against learning-to-defer and certified cascades/routing

Only fields actually read on a primary or publisher source are reported. Items not verified are marked UNVERIFIED. Checked 2026-10-05.

## 1. Learning to defer

1. **Madras, Pitassi, Zemel (2018).** "Predict Responsibly: Improving Fairness and Accuracy by Learning to Defer." Advances in Neural Information Processing Systems 31 (NeurIPS 2018).
   - Read: NeurIPS proceedings abstract page and its generated BibTeX (title, authors "Madras, David and Pitassi, Toni and Zemel, Richard", volume 31, year 2018), and the arXiv record 1711.06664 (v3, 7 Sep 2018, which lists the first name as "Toniann"; the proceedings lists "Toni", used in the .bib).
   - URL: https://proceedings.neurips.cc/paper_files/paper/2018/file/09d37c08f7b129e96277388757530c72-Paper.pdf
   - Page numbers: UNVERIFIED. The proceedings BibTeX has an empty `pages` field. No pages in the .bib.
   - DOI: none found.
   - Key: `madras2018defer`.

2. **Mozannar and Sontag (2020).** "Consistent Estimators for Learning to Defer to an Expert." Proceedings of the 37th International Conference on Machine Learning (ICML), PMLR 119:7076-7087.
   - Read: the PMLR page (citation metadata gives first page 7076, last page 7087, volume 119, authors Hussein Mozannar and David Sontag) and arXiv 2006.01862 (v3, 25 Jan 2021, "ICML 2020").
   - URL: https://proceedings.mlr.press/v119/mozannar20b.html. No DOI on the PMLR page.
   - Key: `mozannar2020defer`.

3. **Cortes, DeSalvo, Mohri (2016).** "Learning with Rejection." Algorithmic Learning Theory (ALT 2016), pp. 67-82.
   - Already in the manuscript: `refs.bib` has key `cortes2016` (title, authors, ALT, pages 67--82, year 2016), and `sec_related.tex` line 37 already cites it in the sentence on selective prediction with abstention. It is a common reference in the paper. The key is reused and is NOT redefined in `new_refs_defer.bib`.
   - Verified on Crossref (https://api.crossref.org/works/10.1007/978-3-319-46379-7_5): title, three authors, "Lecture Notes in Computer Science" / "Algorithmic Learning Theory", pages 67-82, year 2016, DOI 10.1007/978-3-319-46379-7_5. This matches the existing entry. The existing entry has no DOI, so adding `doi = {10.1007/978-3-319-46379-7_5}` to it is optional (I did not edit `refs.bib`).
   - The Springer landing page redirected to a login and was not read. Crossref was the source.

Positioning note: the existing text cites Cortes et al. only as abstention prior art. The new sentence should add that learning to defer (Madras et al.; Mozannar and Sontag) replaces the constant rejection cost with the loss of a downstream decision-maker and learns a deferral policy. It should state that the present method has no learned policy and substitutes the detector answer.

## 2. Certified or risk-controlled cascades and routing (6 entries, most relevant first)

1. **Jung, Brahman, Choi (2025).** "Trust or Escalate: LLM Judges with Provable Guarantees for Human Agreement." ICLR 2025 (proceedings page read; arXiv 2407.18370). Key `jung2025trust`.
   - Cascaded Selective Evaluation tries cheaper judges first and escalates to a stronger one when the confidence is below a calibrated threshold. Thresholds are set by fixed-sequence testing on a calibration set, and the guarantee is P(agree with human | LLM evaluates x) >= 1 - alpha, with abstention when no judge is confident (read in the full text, Sec. 1, 2.3, Algorithm 1).
   - Difference: this is the closest precedent for a finite-sample certificate over the emitted mixture of several predictors, and fixed-sequence testing is shared. But every tier is an LLM judge with the same input and output space, escalation is driven by each tier's own confidence, and the output may be abstention. In ours, the second predictor is a different-modality detector whose answer is substituted and the certificate covers the emitted mixture without a confidence-based routing of tiers.
   - ICLR page numbers: none on the proceedings page. A search-result title lists the paper as an ICLR 2025 oral; this was not checked on iclr.cc and is not in the .bib.

2. **Dou, Lian, Li (2026).** "Conformal Cascade: Distribution-Free Accuracy Guarantees for Multi-Tier LLM Inference." arXiv:2607.25018 (v1 27 Jul 2026; v3 31 Jul 2026). No venue listed; preprint. Key `dou2026conformalcascade`.
   - Per-tier conformal prediction sets; the cascade accepts when the set is a singleton and defers otherwise. The guarantee is coverage of at least 1-K*alpha at the committing tier.
   - Difference: deferral goes to a larger LLM of the same kind. The certificate is a union bound over tiers on set coverage, not a risk bound on a mixture of two heterogeneous predictors, and deferral is a conformal-set-size rule.

3. **Huang, Park, Paoletti, Simeone (2025).** "Reliable Inference in Edge-Cloud Model Cascades via Conformal Alignment." arXiv:2510.17543 (v1 20 Oct 2025; v3 12 Aug 2026; the arXiv page says "Under Review"). Key `huang2025edgecloud`.
   - Escalation from edge to cloud is cast as multiple testing via conformal alignment, controlling the FDR of violations of cloud-referenced conditional coverage among edge-handled inputs.
   - Difference: the reference is the cloud model's predictive distribution, not the ground truth, and the output is a prediction set. The cloud is the stronger fallback, whereas ours substitutes a different-modality detector answer and certifies true-label risk.

4. **Overman and Bayati (2025).** "Conformal Arbitrage: Risk-Controlled Balancing of Competing Objectives in Language Models." arXiv:2506.00911 (1 Jun 2025). No venue listed. Key `overman2025arbitrage`.
   - Conformal risk control calibrates a threshold that decides when a Primary model acts and when a Guardian (a model or a human) is consulted. The risk loss is defined through the Guardian's scores.
   - Difference: both models share the same action set and scores, the guarantee bounds a residual risk of the Primary's candidate set relative to the Guardian, and the Guardian is a stronger or safer model or a human, not a detector of a different modality.

5. **Chen, Zaharia, Zou (2024).** "FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance." Transactions on Machine Learning Research (TMLR), 2024 (arXiv 2305.05176, 9 May 2023). Key `chen2024frugalgpt`.
   - Verified on OpenReview (note id cSimKw5p6R, "Accepted by TMLR", authors Lingjiao Chen, Matei Zaharia, James Zou). Volume and pages: UNVERIFIED, not in the .bib.
   - LLM cascade that queries LLM APIs in order of cost until an answer is accepted by a learned scorer.
   - Difference: cost-driven with a learned acceptance scorer and no finite-sample risk certificate. It is the cascade baseline without a guarantee.

6. **Bary, Macq, Petit (2025).** "No Need for Learning to Defer? A Training Free Deferral Framework to Multiple Experts through Conformal Prediction." arXiv:2509.12573 (v1 16 Sep 2025; v3 28 Mar 2026; 11 pages). No venue listed. Key `bary2025noneed`.
   - Uses conformal prediction sets to decide when to route to human experts and which expert to pick (segregativity criterion). It is a training-free alternative to learning to defer.
   - Difference: the deferral target is a human expert and the certificate concerns the conformal set, not the risk of an emitted mixture of model and detector answers.

Not included (6-entry cap): C3PO (Valkanas et al., NeurIPS 2025). It was only seen as a citation inside Dou et al. ("C3PO: Optimized large language model cascades with probabilistic cost constraints for reasoning", NeurIPS 2025, bounds Pr[cost > budget]). It is a cost guarantee and was not verified on a primary source.

## 3. Structural difference "defer to a stronger model or human" vs "substitute a detector answer under a risk certificate"

No source states this exact contrast. The sentences below are the supported statements, one per item, with the wording taken from the sources.

- **Madras et al.:** rejection learning is "inherently nonadaptive: both the model and the decision-maker act independently of one another," whereas learning to defer makes the pass decision depend on "the decision-maker's expertise and weaknesses," so the deferral policy is learned jointly with the downstream agent's loss (arXiv v3, Sec. 1-2).
- **Mozannar and Sontag:** the system loss L(h,r) = E[l(x,y,h(x)) 1{r(x)=0} + l_exp(x,y,m) 1{r(x)=1}] contains a classifier h and a learned rejector r, and it reduces to learning with rejection when l_exp is a constant c, so a deferred example is charged the expert's loss and no guarantee is placed on the expert's answer (arXiv v3, Sec. 1-2; "no guarantee on the expert" is our reading of the loss, not a quoted statement).
- **Cortes et al.:** learning with rejection charges a constant cost c for abstaining and returns no answer on the rejected region; this is the special case stated by Mozannar and Sontag above (the Cortes paper itself was not read in full text).
- **Jung et al.:** the guarantee is conditional on the instances that are evaluated, P(agree with human | LLM evaluates x) >= 1 - alpha, and the cascade abstains when no judge is confident, so the certificate covers the answers that are emitted (Sec. 1 and Algorithm 1 of arXiv v1).
- **Dou et al.:** "Deferral cannot itself cause an error: only the acceptance event A commits a wrong answer," so in a cascade the guarantee binds at the tier that commits (arXiv v3, main text and proof sketch).
- **Huang et al.:** conventional cascades "route easy inputs to a lightweight model, while deferring difficult inputs to a powerful model" using heuristic confidence, and they argue that such deferral lacks formal guarantees, so the guarantee is stated relative to the cloud model (Sec. I-B).
- **Overman and Bayati, Chen et al., Bary et al.:** no sentence on this contrast was found that I could quote.
