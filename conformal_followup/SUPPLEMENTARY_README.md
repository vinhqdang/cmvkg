# Certified-Correction Risk Control (CCRC) — supplementary material

Self-contained code, extracted per-item signals, and figure sources for the
submission. Anonymous: contains no author, institution, or repository
identifiers.

## Layout

    code/     analysis and experiment scripts (CPU unless noted)
    data/     extracted per-item signals (JSON) + 9 real POPE images
    figures/  the figures as they appear in the paper

## Reproducing the paper

Requires Python >= 3.10 with `numpy`, `scipy`, `scikit-learn`, `matplotlib`,
`Pillow`. No GPU is needed for any table or figure: all VLM and detector
inference was run once, and the per-item outputs are shipped in `data/`.

    cd code
    python3 ccrc_v3.py            # main results: CCRC vs filtering, all settings
    python3 missing_numbers.py    # risk-bound ablation, detector-only baseline,
                                  #   attribution ladder, bottom-q accuracies,
                                  #   subsample sweep
    python3 sym_gate.py           # polarity-symmetric repair gate
    python3 power_seq.py          # power analysis for the sequential experiment
    python3 qsel_transfer.py      # q-selection transfer (leave-one-dataset-out)
    python3 rank_vs_value_sim.py  # rank- vs value-indexed family, validity sim
    python3 make_paper_figures.py # regenerate figures

`ccrc_v3.py` is the canonical implementation and the reference for the
protocol: a three-way disjoint split (fit the score / calibrate the threshold /
test), an ascending fixed sequence stopping at the first non-rejection, and
exact Clopper-Pearson upper bounds at delta = 0.10.

Scripts named `colab_exp*.py` are the GPU extraction jobs that produced the
JSON signals (LLaVA-1.5-7B in 4-bit, Qwen2-VL-2B, OWLv2, CLIP). They are
included for completeness and are the only files that need a GPU; nothing in
the paper requires re-running them.

## Data

| file | contents |
|---|---|
| `raw_scores.json`, `owlv2_scores.json` | POPE-1500, LLaVA-1.5-7B: token probabilities, answers, gold labels, CLIP and OWLv2 grounding scores |
| `exp7_pope.json` | POPE-adversarial, LLaVA-1.5-7B |
| `exp8_qwen_pope.json` | POPE-adversarial, Qwen2-VL-2B |
| `exp9_vcd_pope.json` | POPE-adversarial, LLaVA with visual contrastive decoding |
| `exp10_mme.json`, `exp11_*.json` | MME, GQA, HallusionBench, MME-existence |
| `exp12_amber_*.json` | AMBER discriminative |
| `exp13_bcea.json` | budgeted-evidence-acquisition comparison |
| `exp14_monotonicity.json` | paired matched-prefix sequential experiment (n = 121) |
| `qual_picks.json`, `qual_imgs/` | the six qualitative examples and their source images |

Benchmark images in `qual_imgs/` originate from MS-COCO via POPE and are
included only so the qualitative figure can be regenerated.

## Notes on reading the code

Monte-Carlo counts are >= 400 paired splits for every reported contrast, with
one exception noted in the relevant table caption. Validity is audited as the
fraction of splits whose realised risk exceeds alpha, compared against delta —
not as mean realised risk, which is not the guarantee.
