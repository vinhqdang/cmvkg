# CMVKG-Guard: Cross-Modal Verified Knowledge Graph Guard
## A Novel Training-Free Framework for Real-Time Hallucination Detection and Mitigation in Vision-Language Models

**Research Area:** Trustworthy Vision-Language Models with Knowledge Grounding  
**Keywords:** Hallucination Mitigation, Multimodal Knowledge Graphs, Cross-Modal Verification, Real-Time Correction, Adaptive Confidence Calibration

---

## 1. Executive Summary

We propose **CMVKG-Guard** (Cross-Modal Verified Knowledge Graph Guard), a novel training-free framework that addresses the critical hallucination problem in Vision-Language Models (VLMs) through three key innovations:

1. **Dynamic Hierarchical Multimodal Knowledge Graph (DH-MMKG)** - Self-constructing knowledge graphs that evolve during inference
2. **Cross-Modal Verification Engine (CMVE)** - Bidirectional verification mechanism that validates consistency across visual, textual, and structured knowledge modalities
3. **Adaptive Confidence Calibration with Real-Time Correction (ACC-RTC)** - Dynamic threshold adjustment with immediate hallucination rectification

**Key Innovation:** Unlike existing methods that separate detection from mitigation or use static knowledge bases, CMVKG-Guard operates as an end-to-end system that simultaneously detects, verifies, and corrects hallucinations in real-time during token generation, using dynamically constructed multimodal knowledge graphs as the grounding source.

---

## 2. Research Motivation and Problem Analysis

### 2.1 Critical Gaps in Current Approaches

Based on comprehensive analysis of 2025-2026 literature, we identify five critical limitations:

| **Gap** | **Current State** | **Impact** | **Our Solution** |
|---------|-------------------|------------|------------------|
| **Separation of Detection & Mitigation** | Methods like HSA-DPO detect first, then retrain. Training-free methods (VCD, IBD) only adjust probabilities post-generation | 36-76% hallucination reduction but computationally expensive or incomplete | Unified real-time detection-correction pipeline |
| **Static Knowledge Integration** | Existing KG-RAG methods (mKG-RAG, MMGraphRAG) use pre-built static knowledge graphs | Cannot adapt to novel entities or evolving contexts | Dynamic self-constructing knowledge graphs |
| **Unidirectional Verification** | Methods verify text→image or use uncertainty estimation alone | Miss cross-modal inconsistencies (e.g., correct objects but wrong relationships) | Bidirectional cross-modal verification |
| **Fixed Confidence Thresholds** | Semantic entropy and DST-based methods use static thresholds | Suboptimal across diverse domains and query complexities | Adaptive calibration based on contextual factors |
| **Perception-Reasoning Gap** | 64-72% of residual hallucinations stem from reasoning failures, not perception errors | Current methods focus on object-level hallucinations, miss relation/temporal reasoning errors | Hierarchical verification at object, scene, and reasoning levels |

### 2.2 Benchmark Performance Analysis

Current SOTA methods show these limitations:

- **HSA-DPO**: 36.1% reduction on AMBER, 76.3% on Object HalBench (requires training)
- **VASE (Vision-Amplified Semantic Entropy)**: Best training-free medical VQA method but only 7% AUROC improvement
- **DST-based detection**: 4-10% AUROC improvement but no correction mechanism
- **mKG-RAG**: 76.29% accuracy vs 67.43% baseline but static KG limits adaptability
- **DeGF (Generative Feedback)**: Self-correction but lacks external knowledge grounding

**Target Performance:** CMVKG-Guard aims for:
- >85% hallucination reduction on general benchmarks (POPE, CHAIR, MMHalBench)
- >90% accuracy on knowledge-intensive VQA (KVQA, InfoSeek)
- <15ms additional latency per token (real-time inference compatible)
- Domain-agnostic performance without retraining

---

## 3. Novel Algorithm Architecture: CMVKG-Guard

### 3.1 System Overview

CMVKG-Guard operates as a plug-and-play middleware between the VLM's generation layer and output, consisting of three tightly integrated modules:

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT LAYER                               │
│              [Image] + [Text Query] → VLM Encoder                │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│                 MODULE 1: DH-MMKG Builder                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ Visual Scene │→ │ Entity/Rel   │→ │ Knowledge    │         │
│  │ Graph        │  │ Extraction   │  │ Enrichment   │         │
│  │ Construction │  │              │  │ (WikiData)   │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
│         ↓                   ↓                   ↓               │
│  ┌──────────────────────────────────────────────────┐          │
│  │    Hierarchical MMKG: {Objects, Relations,       │          │
│  │    Attributes, Scene Context, External Facts}     │          │
│  └──────────────────────────────────────────────────┘          │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│          MODULE 2: Cross-Modal Verification Engine               │
│                                                                  │
│  For each generated token candidate t:                          │
│                                                                  │
│  ┌────────────────────────────────────────────────────┐        │
│  │ Layer 1: Visual-Textual Alignment Verification     │        │
│  │  • VAR Score (Visual Attention Ratio)              │        │
│  │  • Semantic Similarity (CLIP-based)                │        │
│  │  • Spatial Consistency Check                       │        │
│  └────────────┬───────────────────────────────────────┘        │
│               ↓                                                 │
│  ┌────────────────────────────────────────────────────┐        │
│  │ Layer 2: Knowledge Graph Grounding Verification    │        │
│  │  • Entity existence in DH-MMKG                     │        │
│  │  • Relation path validation                        │        │
│  │  • Attribute consistency check                     │        │
│  └────────────┬───────────────────────────────────────┘        │
│               ↓                                                 │
│  ┌────────────────────────────────────────────────────┐        │
│  │ Layer 3: Reasoning Chain Verification              │        │
│  │  • Multi-hop reasoning path validation             │        │
│  │  • Temporal consistency (for video/sequences)      │        │
│  │  • Logical entailment checking                     │        │
│  └────────────┬───────────────────────────────────────┘        │
│               ↓                                                 │
│  ┌────────────────────────────────────────────────────┐        │
│  │  Compute Unified Verification Score (UVS):         │        │
│  │  UVS(t) = α·V_visual(t) + β·V_kg(t) + γ·V_reason(t) │        │
│  └────────────────────────────────────────────────────┘        │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│     MODULE 3: Adaptive Confidence Calibration + RTC             │
│                                                                  │
│  ┌────────────────────────────────────────────────────┐        │
│  │ 3.1 Adaptive Threshold Computation                 │        │
│  │  θ_adaptive = f(query_complexity, domain,          │        │
│  │                 historical_accuracy, UVS_dist)      │        │
│  └────────────┬───────────────────────────────────────┘        │
│               ↓                                                 │
│  ┌────────────────────────────────────────────────────┐        │
│  │ 3.2 Hallucination Detection Decision               │        │
│  │  IF UVS(t) < θ_adaptive:                           │        │
│  │    → Hallucination detected                        │        │
│  │  ELSE:                                              │        │
│  │    → Accept token                                  │        │
│  └────────────┬───────────────────────────────────────┘        │
│               ↓                                                 │
│  ┌────────────────────────────────────────────────────┐        │
│  │ 3.3 Real-Time Correction Mechanism                 │        │
│  │  IF hallucination detected:                        │        │
│  │    1. Query DH-MMKG for grounded alternatives      │        │
│  │    2. Retrieve K-best candidates from KG           │        │
│  │    3. Re-rank using contrastive scoring:           │        │
│  │       score(c) = P_vlm(c) + λ·UVS(c)               │        │
│  │    4. Replace hallucinated token with top-ranked   │        │
│  └────────────────────────────────────────────────────┘        │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│                      OUTPUT LAYER                                │
│               [Verified & Corrected Response]                    │
│         + [Explanation Trace] + [Confidence Scores]              │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Module 1: Dynamic Hierarchical Multimodal Knowledge Graph (DH-MMKG)

**Innovation:** Self-constructing knowledge graph that builds during inference rather than relying on pre-built static graphs.

#### 3.2.1 Visual Scene Graph Construction

```python
def construct_visual_scene_graph(image, vlm_features):
    """
    Constructs scene graph from image with three levels of granularity
    """
    # Level 1: Object Detection & Attributes
    objects = detect_objects(image)  # {obj_id: {class, bbox, confidence}}
    attributes = extract_attributes(objects, image)  # {obj_id: {color, size, shape, texture}}
    
    # Level 2: Spatial Relationships
    spatial_rels = compute_spatial_relations(objects)  
    # {(obj_i, obj_j): {relation: ['left_of', 'above'], confidence}}
    
    # Level 3: Scene Context
    scene_type = classify_scene(image)  # indoor/outdoor, setting type
    scene_attributes = extract_scene_features(image)  # lighting, weather, time
    
    # Construct hierarchical graph
    scene_graph = {
        'nodes': {obj_id: {**obj_data, **attrs} 
                  for obj_id, (obj_data, attrs) in enumerate(zip(objects, attributes))},
        'edges': spatial_rels,
        'scene_context': {'type': scene_type, 'attributes': scene_attributes}
    }
    
    return scene_graph
```

#### 3.2.2 Knowledge Enrichment via External KBs

```python
def enrich_with_external_knowledge(scene_graph, query_text):
    """
    Enriches scene graph with factual knowledge from Wikidata/ConceptNet
    """
    enriched_graph = scene_graph.copy()
    
    # Entity linking to Wikidata
    for node_id, node_data in scene_graph['nodes'].items():
        entity_class = node_data['class']
        
        # Query Wikidata for entity facts
        wikidata_facts = query_wikidata_api(entity_class)
        # Returns: {properties, typical_attributes, relations_with_other_entities}
        
        # Integrate factual constraints
        enriched_graph['nodes'][node_id]['factual_constraints'] = {
            'valid_attributes': wikidata_facts['properties'],
            'typical_relations': wikidata_facts['relations'],
            'physical_constraints': wikidata_facts.get('physical_properties', {})
        }
    
    # Query-specific knowledge retrieval
    query_entities = extract_entities(query_text)
    for entity in query_entities:
        if entity not in [n['class'] for n in enriched_graph['nodes'].values()]:
            # Add external knowledge node
            external_facts = retrieve_entity_knowledge(entity)
            enriched_graph['nodes'][f'ext_{entity}'] = {
                'class': entity,
                'source': 'external_kb',
                'facts': external_facts
            }
    
    return enriched_graph
```

#### 3.2.3 Hierarchical Structure

The DH-MMKG has three levels:

1. **Object Level (L1)**: Individual entities with attributes
2. **Scene Level (L2)**: Relations, spatial configurations, scene semantics
3. **Reasoning Level (L3)**: Multi-hop reasoning paths, causal relations, temporal sequences

### 3.3 Module 2: Cross-Modal Verification Engine (CMVE)

**Innovation:** Three-layer hierarchical verification that validates each token candidate against visual input, knowledge graph, and logical reasoning.

#### 3.3.1 Layer 1: Visual-Textual Alignment Verification

```python
def visual_textual_verification(token_candidate, image_features, attention_maps, scene_graph):
    """
    Verifies alignment between generated token and visual content
    """
    # Component 1: Visual Attention Ratio (VAR)
    # Measures how much attention is paid to visual tokens vs text tokens
    var_score = compute_var_score(token_candidate, attention_maps)
    # High VAR = model is grounding in visual content
    # Low VAR = potential hallucination (over-reliance on language prior)
    
    # Component 2: Semantic Similarity
    token_embedding = encode_text(token_candidate)  # CLIP text encoder
    relevant_regions = extract_relevant_image_regions(
        token_candidate, image_features, attention_maps
    )
    region_embeddings = encode_images(relevant_regions)  # CLIP image encoder
    
    semantic_sim = cosine_similarity(token_embedding, region_embeddings.mean(dim=0))
    
    # Component 3: Spatial Consistency
    # If token describes spatial relation (e.g., "left of"), verify against scene graph
    if is_spatial_token(token_candidate):
        spatial_score = verify_spatial_consistency(
            token_candidate, scene_graph['edges']
        )
    else:
        spatial_score = 1.0  # Not applicable
    
    # Unified visual verification score
    V_visual = 0.3 * var_score + 0.5 * semantic_sim + 0.2 * spatial_score
    
    return V_visual, {
        'var': var_score,
        'semantic_sim': semantic_sim,
        'spatial': spatial_score
    }
```

#### 3.3.2 Layer 2: Knowledge Graph Grounding Verification

```python
def knowledge_graph_verification(token_candidate, dh_mmkg, generated_context):
    """
    Verifies token against structured knowledge in DH-MMKG
    """
    # Component 1: Entity Existence Check
    is_entity, entity_type = check_if_entity(token_candidate)
    
    if is_entity:
        # Verify entity exists in visual scene graph
        entity_in_visual = entity_exists_in_graph(
            token_candidate, dh_mmkg['nodes']
        )
        
        # Verify entity is consistent with external knowledge
        if entity_type in ['person', 'organization', 'location', 'product']:
            entity_facts = retrieve_entity_facts(token_candidate, 'wikidata')
            factual_consistency = verify_factual_consistency(
                token_candidate, entity_facts, generated_context
            )
        else:
            factual_consistency = 1.0
    else:
        entity_in_visual = 1.0
        factual_consistency = 1.0
    
    # Component 2: Relation Path Validation
    # If token describes relation (e.g., "wearing", "holding"), validate path exists
    if is_relation_token(token_candidate):
        subject, relation, object_entity = parse_relation_triple(
            generated_context, token_candidate
        )
        
        path_exists = check_relation_path_in_kg(
            dh_mmkg, subject, relation, object_entity
        )
    else:
        path_exists = 1.0
    
    # Component 3: Attribute Consistency
    # If token is attribute (color, size, etc.), verify against visual features
    if is_attribute_token(token_candidate):
        target_entity = get_attribute_target(generated_context, token_candidate)
        
        if target_entity in dh_mmkg['nodes']:
            node_data = dh_mmkg['nodes'][target_entity]
            attribute_consistent = verify_attribute_consistency(
                token_candidate, node_data['attributes'],
                node_data.get('factual_constraints', {})
            )
        else:
            attribute_consistent = 0.5  # Uncertain
    else:
        attribute_consistent = 1.0
    
    V_kg = 0.4 * entity_in_visual + 0.3 * path_exists + 0.3 * attribute_consistent
    
    return V_kg, {
        'entity_check': entity_in_visual,
        'relation_path': path_exists,
        'attribute_consistency': attribute_consistent,
        'factual_consistency': factual_consistency
    }
```

#### 3.3.3 Layer 3: Reasoning Chain Verification

```python
def reasoning_chain_verification(token_candidate, dh_mmkg, generated_context, query):
    """
    Verifies token is consistent with logical reasoning chains
    """
    # Component 1: Multi-hop Reasoning Path Validation
    # Extract reasoning chain from context
    reasoning_chain = extract_reasoning_chain(generated_context)
    
    if len(reasoning_chain) > 1:  # Multi-hop reasoning
        # Verify each hop is supported by KG
        valid_hops = []
        for i in range(len(reasoning_chain) - 1):
            source_entity = reasoning_chain[i]
            target_entity = reasoning_chain[i + 1]
            
            path_exists = find_path_in_kg(
                dh_mmkg, source_entity, target_entity, max_hops=3
            )
            valid_hops.append(1.0 if path_exists else 0.0)
        
        multi_hop_score = np.mean(valid_hops) if valid_hops else 1.0
    else:
        multi_hop_score = 1.0
    
    # Component 2: Temporal Consistency (for video or sequential data)
    if has_temporal_context(query) or is_temporal_token(token_candidate):
        temporal_score = verify_temporal_consistency(
            token_candidate, generated_context, dh_mmkg
        )
    else:
        temporal_score = 1.0
    
    # Component 3: Logical Entailment
    # Check if current token is logically entailed by previous statements
    premises = extract_premises(generated_context)
    conclusion = token_candidate
    
    entailment_score = compute_logical_entailment(
        premises, conclusion, dh_mmkg
    )
    
    V_reason = 0.4 * multi_hop_score + 0.3 * temporal_score + 0.3 * entailment_score
    
    return V_reason, {
        'multi_hop': multi_hop_score,
        'temporal': temporal_score,
        'entailment': entailment_score
    }
```

#### 3.3.4 Unified Verification Score (UVS)

```python
def compute_unified_verification_score(token_candidate, image, dh_mmkg, 
                                      generated_context, query):
    """
    Combines all three layers into unified verification score
    """
    # Layer 1: Visual-Textual Alignment
    V_visual, visual_breakdown = visual_textual_verification(
        token_candidate, image, attention_maps, dh_mmkg
    )
    
    # Layer 2: Knowledge Graph Grounding
    V_kg, kg_breakdown = knowledge_graph_verification(
        token_candidate, dh_mmkg, generated_context
    )
    
    # Layer 3: Reasoning Chain
    V_reason, reason_breakdown = reasoning_chain_verification(
        token_candidate, dh_mmkg, generated_context, query
    )
    
    # Adaptive weighting based on query type
    weights = compute_adaptive_weights(query, generated_context)
    # e.g., visual-heavy queries → higher α, reasoning queries → higher γ
    
    α, β, γ = weights['visual'], weights['kg'], weights['reasoning']
    
    UVS = α * V_visual + β * V_kg + γ * V_reason
    
    return UVS, {
        'visual': visual_breakdown,
        'kg': kg_breakdown,
        'reasoning': reason_breakdown,
        'weights': weights
    }
```

### 3.4 Module 3: Adaptive Confidence Calibration with Real-Time Correction

**Innovation:** Dynamic threshold adjustment based on contextual factors, with immediate correction using knowledge-grounded alternatives.

#### 3.4.1 Adaptive Threshold Computation

```python
def compute_adaptive_threshold(query, generated_context, dh_mmkg, 
                               historical_performance):
    """
    Dynamically computes verification threshold based on multiple factors
    """
    # Factor 1: Query Complexity
    complexity_score = assess_query_complexity(query)
    # High complexity (multi-hop reasoning) → lower threshold (more strict)
    # Low complexity (simple object identification) → higher threshold (more lenient)
    
    # Factor 2: Domain Specificity
    domain = classify_domain(query, dh_mmkg)  # medical, general, technical, etc.
    domain_risk = DOMAIN_RISK_MAP[domain]
    # High-risk domains (medical, legal) → lower threshold
    
    # Factor 3: Knowledge Graph Density
    kg_density = compute_kg_density(dh_mmkg)
    # Rich KG (many entities/relations) → lower threshold (can be more strict)
    # Sparse KG → higher threshold (less grounding available)
    
    # Factor 4: Historical Accuracy
    if historical_performance is not None:
        recent_accuracy = historical_performance['last_10_tokens']
        # High recent accuracy → maintain current threshold
        # Low recent accuracy → lower threshold (be more conservative)
        history_adjustment = 1.0 - (1.0 - recent_accuracy) * 0.3
    else:
        history_adjustment = 1.0
    
    # Factor 5: Uncertainty Distribution
    # Analyze distribution of UVS scores for recent tokens
    if len(generated_context) > 5:
        recent_uvs_scores = get_recent_uvs_scores(generated_context)
        uvs_std = np.std(recent_uvs_scores)
        # High variance → lower threshold (uncertain generation)
    else:
        uvs_std = 0.0
    
    # Compute adaptive threshold
    base_threshold = 0.7  # Default
    
    θ_adaptive = base_threshold * (
        1.0 - 0.2 * complexity_score +      # ↓ threshold for complex queries
        0.1 * domain_risk +                  # ↑ threshold for risky domains
        0.1 * kg_density +                   # ↓ threshold with rich KG
        0.2 * history_adjustment +           # Adjust based on recent performance
        0.15 * (1.0 - min(uvs_std, 0.5))    # ↓ threshold for uncertain generation
    )
    
    # Clamp to reasonable range
    θ_adaptive = np.clip(θ_adaptive, 0.4, 0.9)
    
    return θ_adaptive, {
        'complexity': complexity_score,
        'domain_risk': domain_risk,
        'kg_density': kg_density,
        'history_adj': history_adjustment,
        'uvs_std': uvs_std
    }
```

#### 3.4.2 Real-Time Correction Mechanism

```python
def real_time_correction(token_candidate, uvs_score, dh_mmkg, 
                        vlm_logits, generated_context):
    """
    Performs real-time correction when hallucination is detected
    """
    # Step 1: Query DH-MMKG for grounded alternatives
    context_entities = extract_entities(generated_context)
    expected_token_type = predict_expected_token_type(generated_context)
    
    # Retrieve candidate replacements from KG
    kg_candidates = query_kg_for_candidates(
        dh_mmkg, 
        context_entities, 
        expected_token_type,
        k=10  # Top-10 candidates
    )
    
    # Step 2: Re-rank candidates using combined scoring
    candidate_scores = []
    
    for candidate in kg_candidates:
        # VLM's original probability
        P_vlm = get_token_probability(candidate, vlm_logits)
        
        # Recompute UVS for this candidate
        uvs_candidate, _ = compute_unified_verification_score(
            candidate, image, dh_mmkg, generated_context, query
        )
        
        # Combined score
        λ = 0.6  # Weight for UVS (prioritize grounding)
        combined_score = (1 - λ) * P_vlm + λ * uvs_candidate
        
        candidate_scores.append((candidate, combined_score, uvs_candidate))
    
    # Sort by combined score
    candidate_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Step 3: Select best candidate
    corrected_token = candidate_scores[0][0]
    corrected_uvs = candidate_scores[0][2]
    
    # Step 4: Generate explanation
    explanation = generate_correction_explanation(
        original_token=token_candidate,
        corrected_token=corrected_token,
        reasoning=f"Original UVS: {uvs_score:.3f}, Corrected UVS: {corrected_uvs:.3f}",
        evidence=kg_candidates[0]['evidence']  # KG path supporting correction
    )
    
    return corrected_token, explanation, candidate_scores
```

#### 3.4.3 Complete Token Generation Process

```python
def generate_with_cmvkg_guard(vlm_model, image, query, dh_mmkg_config):
    """
    Complete generation process with CMVKG-Guard
    """
    # Initialize
    generated_tokens = []
    generated_text = ""
    explanations = []
    historical_performance = {'last_10_tokens': []}
    
    # Build DH-MMKG
    scene_graph = construct_visual_scene_graph(image, vlm_model.encode_image(image))
    dh_mmkg = enrich_with_external_knowledge(scene_graph, query)
    
    # Prepare initial prompt
    prompt_tokens = tokenize(query)
    
    # Autoregressive generation
    for step in range(MAX_TOKENS):
        # Get VLM's next token prediction
        logits = vlm_model.forward(image, prompt_tokens + generated_tokens)
        top_k_candidates = get_top_k_tokens(logits, k=5)
        
        # For each candidate, compute UVS
        candidate_results = []
        for candidate_token in top_k_candidates:
            uvs_score, breakdown = compute_unified_verification_score(
                candidate_token, image, dh_mmkg, 
                generated_text, query
            )
            candidate_results.append((candidate_token, uvs_score, breakdown))
        
        # Get best candidate by VLM
        best_vlm_candidate = candidate_results[0]  # Highest VLM probability
        token_candidate, uvs_score, breakdown = best_vlm_candidate
        
        # Compute adaptive threshold
        θ_adaptive, threshold_factors = compute_adaptive_threshold(
            query, generated_text, dh_mmkg, historical_performance
        )
        
        # Detection decision
        if uvs_score < θ_adaptive:
            # Hallucination detected → Correct
            corrected_token, explanation, alternatives = real_time_correction(
                token_candidate, uvs_score, dh_mmkg,
                logits, generated_text
            )
            
            final_token = corrected_token
            explanations.append({
                'position': step,
                'original': token_candidate,
                'corrected': corrected_token,
                'explanation': explanation,
                'uvs_original': uvs_score,
                'uvs_corrected': alternatives[0][2],
                'threshold': θ_adaptive
            })
            
            # Update historical performance (correction occurred)
            historical_performance['last_10_tokens'].append(0)
        else:
            # Accept token
            final_token = token_candidate
            historical_performance['last_10_tokens'].append(1)
        
        # Keep only last 10 for history
        if len(historical_performance['last_10_tokens']) > 10:
            historical_performance['last_10_tokens'].pop(0)
        
        # Update generation
        generated_tokens.append(final_token)
        generated_text += decode_token(final_token) + " "
        
        # Check for EOS
        if final_token == EOS_TOKEN:
            break
    
    return {
        'generated_text': generated_text.strip(),
        'corrections': explanations,
        'dh_mmkg': dh_mmkg,  # For inspection
        'historical_performance': historical_performance
    }
```

---

## 4. Experimental Design

### 4.1 Benchmark Datasets

We will evaluate CMVKG-Guard on six diverse benchmarks covering different hallucination types:

| **Benchmark** | **Type** | **Size** | **Metrics** | **Purpose** |
|---------------|----------|----------|-------------|-------------|
| **POPE** | Object Hallucination | 3,000 Yes/No questions | Accuracy, Precision, Recall, F1 | Adversarial object existence |
| **CHAIR** | Image Captioning | 500 images (MSCOCO) | CHAIRs, CHAIRi | Object hallucination in captions |
| **MMHalBench** | Multi-type Hallucination | 96 images, 8 question types | Accuracy, Hallucination Score | Comprehensive evaluation |
| **MedHallBench** | Medical Hallucination | 1,200 medical images | MediHall Score, Accuracy | High-stakes domain |
| **InfoSeek** | Knowledge-intensive VQA | 1.3M questions | Accuracy, F1 | External knowledge requirement |
| **KVQA** | Knowledge-based VQA | 183K image-question pairs | Accuracy | Structured knowledge |

### 4.2 Baseline Comparisons

We will compare against SOTA methods in three categories:

**Training-Free Methods:**
- VCD (Visual Contrastive Decoding)
- IBD (Instruction-Based Decoding)
- VASE (Vision-Amplified Semantic Entropy)
- RCD (Retrieval Contrastive Decoding)

**Knowledge-Enhanced Methods:**
- mKG-RAG (Multimodal KG-RAG)
- MMGraphRAG
- Wiki-LLaVA

**Training-Based Methods (for reference):**
- HSA-DPO (Hallucination Severity-Aware DPO)
- GAVIE (Grounded Attribute Value Identification)

### 4.3 Ablation Studies

| **Ablation** | **Variant** | **Purpose** |
|--------------|-------------|-------------|
| w/o DH-MMKG | Use static KG (ConceptNet) | Test dynamic KG value |
| w/o CMVE Layer 1 | Remove visual-textual verification | Test visual grounding importance |
| w/o CMVE Layer 2 | Remove KG verification | Test structured knowledge impact |
| w/o CMVE Layer 3 | Remove reasoning verification | Test reasoning chain validation |
| w/o Adaptive Threshold | Use fixed threshold (0.7) | Test adaptive calibration value |
| w/o Real-Time Correction | Detection only (no correction) | Test correction mechanism |
| Simplified UVS | Equal weights (α=β=γ=1/3) | Test adaptive weighting |

### 4.4 Evaluation Metrics

**Hallucination Detection Metrics:**
- AUROC (Area Under ROC Curve)
- AUPRC (Area Under Precision-Recall Curve)
- F1 Score at optimal threshold

**Hallucination Mitigation Metrics:**
- Hallucination Rate Reduction (%)
- CHAIRs / CHAIRi (image captioning)
- Accuracy improvement

**Quality Metrics:**
- BLEU, ROUGE, METEOR (for generative tasks)
- Human evaluation (fluency, faithfulness, informativeness)

**Efficiency Metrics:**
- Latency per token (ms)
- Throughput (tokens/second)
- Memory overhead (GB)

---

## 5. Expected Results and Contributions

### 5.1 Quantitative Targets

Based on current SOTA performance, we target:

| **Metric** | **Current SOTA** | **CMVKG-Guard Target** |
|------------|------------------|------------------------|
| POPE Accuracy | 82.5% (best baseline) | **>90%** |
| CHAIR Reduction | 76.3% (HSA-DPO) | **>85%** |
| MedHallBench Accuracy | 53.96% (GPT-4o) | **>75%** |
| InfoSeek Accuracy | 67.43% (baseline RAG) | **>80%** |
| AUROC (Detection) | 0.78 (VASE) | **>0.88** |
| Latency Overhead | 50-100ms (RAG methods) | **<15ms** |

### 5.2 Key Technical Contributions

1. **First training-free end-to-end hallucination detection + correction system** that operates in real-time during generation

2. **Dynamic self-constructing multimodal knowledge graphs** that adapt to input content rather than using static pre-built KGs

3. **Three-layer cross-modal verification** addressing perception, knowledge grounding, and reasoning hallucinations simultaneously

4. **Adaptive confidence calibration** that adjusts detection threshold based on query complexity, domain risk, and historical performance

5. **Explainable AI component** providing correction traces and evidence chains for each detected hallucination

### 5.3 Practical Impact

**For Healthcare Applications:**
- Reduce medical hallucinations by >85% (critical for clinical decision support)
- Provide verifiable evidence chains for each claim
- Domain-agnostic: works without medical-specific training

**For Autonomous Systems:**
- Real-time verification suitable for robotics and autonomous driving
- Multi-modal grounding prevents action errors
- Temporal consistency checking for sequential decisions

**For Education:**
- Reliable knowledge retrieval and explanation
- Explainable corrections help students learn
- Multilingual support through knowledge graphs

---

## 6. Implementation Plan

### 6.1 Technical Stack

- **VLM Backbone:** LLaVA-1.5 (7B), InstructBLIP, Qwen2-VL
- **Knowledge Graph:** WikiData API, ConceptNet, Domain-specific KGs
- **Vision Components:** CLIP (ViT-L/14), DINO v2, SAM for segmentation
- **Inference Framework:** PyTorch, vLLM for efficient serving
- **Evaluation:** LMMs-Eval framework

### 6.2 Development Timeline (6 months)

**Month 1-2: Core Implementation**
- Implement DH-MMKG construction pipeline
- Develop visual scene graph extraction
- Integrate external knowledge APIs

**Month 3-4: Verification Engine**
- Build three-layer verification system
- Implement UVS computation
- Develop adaptive threshold mechanism

**Month 5: Real-Time Correction**
- Implement correction pipeline
- Optimize for latency
- Add explanation generation

**Month 6: Evaluation & Refinement**
- Run comprehensive benchmark evaluation
- Ablation studies
- Human evaluation study
- Performance optimization

### 6.3 Resource Requirements

- **Compute:** 4× A100 GPUs (40GB) for development and evaluation
- **Storage:** 500GB for datasets and knowledge bases
- **APIs:** WikiData, ConceptNet access
- **Team:** 1 PhD researcher + 1 MS student + advisor supervision

---

## 7. Broader Impact and Alignment with CFP

### 7.1 SDG 9 Alignment (Industry, Innovation and Infrastructure)

CMVKG-Guard directly contributes to SDG 9 through:
- **Innovation:** Novel training-free architecture advancing AI safety
- **Infrastructure:** Plug-and-play system deployable across industries
- **Inclusive Innovation:** Open-source release enables global access

### 7.2 CFP Topic Coverage

| **CFP Topic** | **CMVKG-Guard Contribution** |
|---------------|------------------------------|
| Multimodal representation learning | Three-layer hierarchical verification across modalities |
| Cross-modal alignment and reasoning | Bidirectional visual-textual-knowledge alignment |
| Knowledge-enhanced multimodal AI | Dynamic KG construction + external KB integration |
| Explainable and trustworthy AI | Verification breakdowns + correction explanations |
| Applications in healthcare | Medical hallucination mitigation (MedHallBench) |
| Multimodal large models | Compatible with any VLM architecture |

### 7.3 Societal Impact

**Positive Impacts:**
- Safer deployment of VLMs in high-stakes domains
- Reduced misinformation from AI systems
- Transparent and explainable AI decisions

**Potential Risks:**
- Over-reliance on automated verification
- Privacy concerns with external knowledge retrieval
- Computational cost for resource-constrained settings

**Mitigation:**
- Clear documentation of system limitations
- Privacy-preserving KG construction options
- Lightweight variant for edge deployment

---

## 8. Conclusion

CMVKG-Guard represents a paradigm shift in addressing VLM hallucinations by unifying detection and correction in a real-time, training-free, explainable framework. The key innovation is the dynamic hierarchical multimodal knowledge graph that evolves during inference, coupled with three-layer cross-modal verification and adaptive confidence calibration.

Unlike existing methods that treat hallucination as a post-hoc problem or require expensive retraining, CMVKG-Guard intervenes during token generation, using structured knowledge grounding to prevent hallucinations before they propagate. The system is designed for practical deployment: training-free, plug-and-play compatible with any VLM, and real-time efficient.

By addressing the critical gap between perception and reasoning, and by providing explainable verification traces, CMVKG-Guard advances both the theoretical understanding and practical deployment of trustworthy multimodal AI systems—directly aligned with the CFP's mission to foster reliable, knowledge-enhanced, and explainable multimodal AI innovations.

---

## References

[Comprehensive references to be added from literature review above, including 40+ papers from 2025-2026 on hallucination detection, KG-RAG, multimodal verification, and related topics]

---

**Contact Information:**
[To be filled with actual researcher details]

**Code Release:** Upon acceptance, full implementation will be open-sourced on GitHub

**Dataset:** Evaluation datasets and DH-MMKG construction code will be publicly released