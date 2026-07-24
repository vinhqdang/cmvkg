# Datasets used by baseline / related methods (for comparability)

Extracted from the papers themselves (arXiv/ACL), 2026.

## OPERA — Huang et al., CVPR 2024 (arXiv 2311.17911)
Models: InstructBLIP, MiniGPT-4, LLaVA-1.5, Shikra (all 7B).
- MSCOCO 2014 — CHAIR captioning (500 images, "describe in detail")
- Visual Genome — GPT-4-assisted attribute/location/relation hallucination eval
- POPE — object existence yes/no (random, popular, adversarial)
- MME — MLLM capability benchmark
- MMBench — general capability

## Attention Lens — "Devils in Middle Layers", CVPR 2025 (arXiv 2411.16724)
Models: LLaVA-1.5 (7B/13B), Shikra-7B, MiniGPT-4-7B.
- COCO 2014 val — CHAIR captioning (500 main; 2000 for mechanism case studies)
- AMBER — supplementary comparison
- Detection metrics: AUROC, mAP (hallucination detection), F1

## REVERSE — "Generate, but Verify", NeurIPS 2025 (reverse-vlm.github.io)
Base: LLaVA-v1.5. Training: 1.3M-sample augmented instruction set with confidence tokens.
- CHAIR-MSCOCO — object hallucination in captions
- AMBER(g) — grounded caption generation
- HaloQuest — open-ended QA hallucination
- MMHal — multimodal QA hallucination
- AMBER(d) — discriminative yes/no
- POPE — object hallucination detection
- MME-Hall — visual reasoning hallucination

## ConfLVLM — "Towards Statistical Factuality Guarantee for LVLMs", EMNLP 2025 (arXiv 2502.20560)
The closest peer (conformal factuality). Three domains:
- General scene understanding: MSCOCO (500 imgs, >3 objects, same pool as POPE);
  models LLaVA-1.5, Phi-3.5-vision-instruct, Llama-3.2-11B-Vision, GPT-4o-mini;
  scorer CLIP-ViT-B/32.
- Medical radiology report generation: MIMIC-CXR (500 chest X-rays); model
  LLaVA-Med (Mistral-7B v1.5); scorer BiomedCLIP.
- Document understanding: SROIE (receipt scans); scorer LayoutLMv3.
- References the THRONE object-hallucination benchmark.

## Union & frequency (standard hallucination benchmarks)
| Dataset | OPERA | AttnLens | REVERSE | ConfLVLM | ours |
|---|:-:|:-:|:-:|:-:|:-:|
| POPE | Y | | Y | (COCO) | done |
| CHAIR-MSCOCO (captioning) | Y | Y | Y | Y | to add |
| AMBER (g/d) | | Y | Y | | to add |
| MME / MME-Hall | Y | | Y | | done |
| MMHal | | | Y | | (accepted paper) |
| Visual Genome | Y | | | | - |
| MMBench | Y | | | | - |
| HaloQuest | | | Y | | - |
| MIMIC-CXR (medical) | | | | Y | - |
| SROIE (document) | | | | Y | - |

Must-have comparability set (>=2 papers): **POPE, CHAIR-MSCOCO, AMBER, MMHal, MME**.
We cover POPE + MME; adding CHAIR-MSCOCO (captioning, all 4 papers) and AMBER
(AttnLens + REVERSE) gives direct comparability. ConfLVLM's MIMIC-CXR + SROIE
support a cross-domain breadth claim.
