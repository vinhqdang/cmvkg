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

## Usage

### 1. Running the Demonstration

To run a simple illustrative demonstration of the guarded generation process:

```bash
python run_demo.py
```

### 2. Running Experiments

To run full validation experiments (e.g., across samples, ablation studies, or evaluating backward compatibility):

```bash
# Evaluate CMVKG-Guard using a specific core VLM model
python do_experiment.py --model llava

# Run an ablation study by disabling external knowledge graphs
python do_experiment.py --model qwen --disable_external_kg

# Evaluate a baseline model (without Guardian) instead
python do_experiment.py --baseline OPERA
```

Supported VLM drivers include: `llava`, `qwen`, and `instructblip`.

### 3. Latency Analysis

To run latency profiling across the different generation and reasoning layers:

```bash
python run_latency_analysis.py
```

### 4. Manuscript Generation

The repository also includes utility scripts for generating manuscript assets:
- `python generate_method_diagrams.py` - Generates architectural diagrams for the paper.
- `python generate_result_charts.py` - Synthesizes test outputs into quantitative performance charts.
- `python generate_manuscript_examples.py` - Extracts qualitative examples for analysis.
