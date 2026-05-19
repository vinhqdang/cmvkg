"""
Independent Layer 3 Validation (Revision 2, Reviewer 2 Concern #1).

Validates the ReasoningVerifier in isolation using synthetic prompts
with known logical properties:
  - Temporal contradictions ("X happened after Y, then before Y again")
  - Negation consistency ("there is no X" → corrector should not insert X)
  - Multi-hop entailment ("Paris is in France, France is in Europe → Paris is in Europe")

Evaluates precision and recall of the verifier at catching logical errors.
"""

import sys, os, json, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cmvkg_guard.verification.reasoning import ReasoningVerifier
from cmvkg_guard.graph.schema import DHMMKG, Node as GraphNode, Edge as GraphEdge


def make_consistent_graph() -> DHMMKG:
    """Graph with temporally and logically consistent edges."""
    g = DHMMKG()
    g.add_node(GraphNode(id="paris",  label="Paris",  type="entity", confidence=0.9, attributes={}))
    g.add_node(GraphNode(id="france", label="France", type="entity", confidence=0.9, attributes={}))
    g.add_node(GraphNode(id="europe", label="Europe", type="entity", confidence=0.9, attributes={}))
    g.add_edge(GraphEdge(source_id="paris",  target_id="france", relation="located_in", confidence=0.95))
    g.add_edge(GraphEdge(source_id="france", target_id="europe", relation="part_of",    confidence=0.95))
    return g


def make_inconsistent_graph() -> DHMMKG:
    """Graph with a temporal contradiction."""
    g = DHMMKG()
    g.add_node(GraphNode(id="event_a", label="Event A", type="event", confidence=0.8, attributes={}))
    g.add_node(GraphNode(id="event_b", label="Event B", type="event", confidence=0.8, attributes={}))
    g.add_edge(GraphEdge(source_id="event_a", target_id="event_b", relation="before", confidence=0.8))
    g.add_edge(GraphEdge(source_id="event_b", target_id="event_a", relation="before", confidence=0.8))  # Contradiction!
    return g


# ---------------------------------------------------------------------------
# Test probes: (token, context, graph, is_logically_inconsistent)
# ---------------------------------------------------------------------------
PROBES = [
    # Consistent: grounded token in consistent graph → should score HIGH
    ("europe",  "Paris is in France, so it is in Europe.",       make_consistent_graph(), False),
    ("france",  "Paris is the capital of France.",                make_consistent_graph(), False),
    ("located", "Paris is located in France.",                    make_consistent_graph(), False),

    # Inconsistent: temporal keywords in an inconsistent graph → score LOW
    ("before",  "Event A happened before Event B then after it.", make_inconsistent_graph(), True),
    ("after",   "Event B occurred after A but also before A.",    make_inconsistent_graph(), True),

    # Borderline: negation context — should be treated carefully
    ("dog",     "There is no dog in the image.",                  make_consistent_graph(), False),
    ("cat",     "There is no cat here, only a bird.",             make_consistent_graph(), False),
]


def run_layer3_validation():
    verifier = ReasoningVerifier()
    THRESHOLD = 0.55  # Layer-3 standalone threshold (lower than full UVS threshold)

    print("=" * 70)
    print("CMVKG-Guard — Layer 3 (Reasoning Verifier) Independent Validation")
    print("=" * 70)
    print(f"{'Token':<10} {'Inconsistent':>13} {'L3 Score':>10} {'Flagged':>8} {'Correct':>8}")
    print("-" * 60)

    tp = tn = fp = fn = 0
    rows = []

    for token, context, graph, is_inconsistent in PROBES:
        scores = verifier.verify(token, context, graph)
        l3_score = (0.33 * scores["temporal"] + 0.33 * scores["logic"] + 0.34 * scores["multi_hop"])
        flagged = l3_score < THRESHOLD

        correct = (flagged == is_inconsistent)
        if is_inconsistent and flagged:     tp += 1
        elif not is_inconsistent and not flagged: tn += 1
        elif not is_inconsistent and flagged:     fp += 1
        else:                                     fn += 1

        tag = "✓" if correct else "✗"
        print(f"{token:<10} {str(is_inconsistent):>13} {l3_score:>10.4f} {str(flagged):>8} {tag:>8}")

        rows.append({
            "token": token, "context": context,
            "is_inconsistent": is_inconsistent,
            "l3_score": round(l3_score, 4),
            "flagged": flagged, "correct": correct,
            "scores": {k: round(v, 4) for k, v in scores.items()},
        })

    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-9)
    accuracy  = (tp + tn) / len(PROBES)

    print("-" * 60)
    print(f"  Precision : {precision:.3f}")
    print(f"  Recall    : {recall:.3f}")
    print(f"  F1        : {f1:.3f}")
    print(f"  Accuracy  : {accuracy:.3f}")

    output = {
        "threshold": THRESHOLD,
        "probes": rows,
        "metrics": {
            "precision": round(precision, 3),
            "recall":    round(recall, 3),
            "f1":        round(f1, 3),
            "accuracy":  round(accuracy, 3),
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        }
    }

    os.makedirs("experiments/results", exist_ok=True)
    out_path = "experiments/results/layer3_standalone_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")
    return output


if __name__ == "__main__":
    run_layer3_validation()
