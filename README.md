# CMVKG-Guard: Trustworthy Vision-Language Models

## Project Summary

CMVKG-Guard is a research project focused on developing a novel framework for detecting and correcting hallucinations in Vision-Language Models (VLMs). The project is documented in `research_proposal.md` and `research_summary.md` in this repository.

## Documented Ideas

**Core Innovation**: A training-free, real-time hallucination detection and mitigation system with three key components:

### 1. Dynamic Hierarchical Multimodal Knowledge Graph (DH-MMKG)
- Self-constructing knowledge graphs that build during inference
- Visual scene graph construction from images
- External knowledge enrichment via WikiData/ConceptNet
- Three-level hierarchy: Objects → Scene Relations → Reasoning Chains

### 2. Cross-Modal Verification Engine (CMVE)
- Three-layer verification system that validates each token:
    - **Layer 1**: Visual-Textual Alignment (30% weight)
    - **Layer 2**: Knowledge Graph Grounding (40% weight)
    - **Layer 3**: Reasoning Chain Verification (30% weight)
- Computes Unified Verification Score (UVS) for each token

### 3. Adaptive Confidence Calibration with Real-Time Correction (ACC-RTC)
- Dynamic threshold adjustment based on query complexity, domain, etc.
- Real-time token replacement when hallucinations are detected
- Knowledge-grounded alternatives with explanation traces

## Implementation Status

### ✅ Well Implemented (85-90% complete)
- Core pipeline architecture in `cmvkg_guard/pipeline.py`
- Knowledge graph builder in `cmvkg_guard/graph/builder.py`
- Three-layer verification engine in `cmvkg_guard/verification/engine.py`
- Real-time correction mechanism in `cmvkg_guard/correction/corrector.py`
- Configuration management system

### 🔶 Partially Implemented
- Basic knowledge enrichment (needs deeper external KB integration)
- Reasoning verification (has placeholder mechanisms)
- Adaptive threshold computation (basic implementation)

### ❌ Missing/Needs Enhancement
- Advanced external knowledge source integration
- Sophisticated reasoning chain validation
- Complex multi-hop reasoning verification
- Comprehensive evaluation benchmarks

## Key Findings

1.  **High Fidelity to Research Proposal**: The code structure closely follows the architectural design outlined in the research documents, with proper modular separation and clear implementation of the three core components.
2.  **Solid Foundation**: The current implementation provides a robust prototype that demonstrates the core concepts, with type hints, documentation, and extensible design patterns.
3.  **Research-Grade Quality**: This project demonstrates a systematic approach to addressing VLM hallucinations with comprehensive documentation.
4.  **Implementation Gap**: While the core architecture is implemented, advanced algorithms (like complex reasoning verification) require further development.

## Papers

1. **Paper 1 — CMVKG-Guard.** Framework manuscript in `manuscript/`.
2. **Paper 2 — "Certified Correction of Vision–Language Hallucinations: Gains, Preconditions, and a Sequential Failure Mode" (CCRC).** Follow-up study in `conformal_followup/`, manuscript in `conformal_followup/manuscript_neurocomputing/`. **Status: under review at Neurocomputing.**

## Usage

To run the demonstration script:

```bash
python run_demo.py
```
