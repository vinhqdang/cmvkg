"""
Failure Case Analysis for CMVKG-Guard (Revision 2, Reviewer 2 Concern #3).

Taxonomy of failure modes:
  1. Over-correction  – valid token flagged and incorrectly replaced
  2. Under-correction – hallucinated token slips through undetected
  3. Wrong-correction – hallucination caught but replacement is wrong

Each case is simulated by constructing carefully crafted probe scenarios and
running them through the full verification engine with a synthetic graph,
so the experiment is self-contained and reproducible.
"""

import sys, os, json, random, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cmvkg_guard.config import CMVKGConfig
from cmvkg_guard.verification.engine import UnifiedVerificationEngine
from cmvkg_guard.graph.schema import DHMMKG, Node as GraphNode, Edge as GraphEdge

random.seed(42)


def make_graph_with_objects(objects: list) -> DHMMKG:
    """Build a minimal synthetic DHMMKG containing the given objects."""
    g = DHMMKG()
    for obj in objects:
        g.add_node(GraphNode(id=obj, label=obj, type="object",
                             confidence=0.9, attributes={}))
    if len(objects) >= 2:
        g.add_edge(GraphEdge(source_id=objects[0], target_id=objects[1],
                             relation="near", confidence=0.8))
    return g


def build_engine() -> UnifiedVerificationEngine:
    config = CMVKGConfig()
    return UnifiedVerificationEngine(config)


# ---------------------------------------------------------------------------
# Failure Case Definitions
# ---------------------------------------------------------------------------
FAILURE_CASES = [
    # --- OVER-CORRECTION: valid, well-grounded tokens that score low due to
    #     heuristic misfire (e.g., graph lacks the object but it really exists)
    {
        "type": "over_correction",
        "token": "umbrella",
        "context": "The woman is holding an umbrella.",
        "graph_objects": ["woman", "bag"],   # umbrella deliberately absent from graph
        "expected_uvs_range": "low",
        "description": "Correct token 'umbrella' absent from graph → may be falsely flagged.",
    },
    {
        "type": "over_correction",
        "token": "sitting",
        "context": "The cat is sitting on the sofa.",
        "graph_objects": ["cat", "sofa"],
        "expected_uvs_range": "borderline",
        "description": "Action verb 'sitting' has no direct KG entity → kg score may penalise unfairly.",
    },
    # --- UNDER-CORRECTION: hallucinated tokens that score high because the
    #     graph accidentally supports them (object in graph but not in image)
    {
        "type": "under_correction",
        "token": "dog",
        "context": "There is a dog near the fence.",
        "graph_objects": ["dog", "fence"],   # graph built from wrong detections
        "expected_uvs_range": "high",
        "description": "Hallucinated 'dog' is grounded in graph (bad detection) → slips through.",
    },
    {
        "type": "under_correction",
        "token": "bicycle",
        "context": "A bicycle is parked beside the car.",
        "graph_objects": ["bicycle", "car"],
        "expected_uvs_range": "high",
        "description": "Hallucinated 'bicycle' spuriously detected and added to graph.",
    },
    # --- WRONG CORRECTION: hallucination caught correctly, but replacement
    #     token is the highest-confidence node in the graph regardless of context
    {
        "type": "wrong_correction",
        "token": "airplane",
        "context": "The man is riding an airplane.",
        "graph_objects": ["man", "bicycle"],  # bicycle is grounded, but nonsensical replacement
        "expected_uvs_range": "low",
        "description": "Hallucinated 'airplane' caught, but corrector picks 'bicycle' which is wrong in context.",
    },
]


def run_failure_analysis():
    engine = build_engine()
    results = []

    print("=" * 65)
    print("CMVKG-Guard Failure Case Analysis")
    print("=" * 65)

    counts = {"over_correction": 0, "under_correction": 0, "wrong_correction": 0}
    triggered = {"over_correction": 0, "under_correction": 0, "wrong_correction": 0}

    for case in FAILURE_CASES:
        graph = make_graph_with_objects(case["graph_objects"])
        from PIL import Image
        dummy_image = Image.new("RGB", (224, 224), color=(128, 128, 128))

        uvs, breakdown = engine.verify_token(
            token=case["token"],
            image=dummy_image,
            graph=graph,
            context=case["context"]
        )

        threshold = CMVKGConfig().base_threshold
        flagged = uvs < threshold

        # Determine if failure mode actually triggered
        is_failure = False
        if case["type"] == "over_correction":
            # Failure = valid token was flagged (false positive)
            is_failure = flagged
        elif case["type"] == "under_correction":
            # Failure = hallucination not flagged (false negative)
            is_failure = not flagged
        elif case["type"] == "wrong_correction":
            # Failure = token flagged (first condition for wrong replacement)
            is_failure = flagged

        counts[case["type"]] += 1
        if is_failure:
            triggered[case["type"]] += 1

        result = {
            "type": case["type"],
            "token": case["token"],
            "uvs": round(uvs, 4),
            "threshold": threshold,
            "flagged": flagged,
            "failure_triggered": is_failure,
            "description": case["description"],
            "breakdown": {k: {sk: round(sv, 3) for sk, sv in v.items()}
                          if isinstance(v, dict) else round(v, 3)
                          for k, v in breakdown.items()},
        }
        results.append(result)

        status = "⚠ FAILURE" if is_failure else "✓ OK"
        print(f"\n[{case['type'].upper()}] {status}")
        print(f"  Token   : '{case['token']}'")
        print(f"  Context : {case['context']}")
        print(f"  UVS     : {uvs:.4f}  (threshold={threshold})")
        print(f"  Flagged : {flagged}")
        print(f"  Note    : {case['description']}")

    # Summary
    print("\n" + "=" * 65)
    print("FAILURE SUMMARY")
    print("=" * 65)
    total_cases = len(FAILURE_CASES)
    total_failures = sum(triggered.values())
    for ftype, total in counts.items():
        t = triggered[ftype]
        print(f"  {ftype:<20}: {t}/{total} triggered ({100*t//max(total,1)}%)")
    print(f"\n  Total failures : {total_failures}/{total_cases} ({100*total_failures//total_cases}%)")

    taxonomy = {
        ftype: {"triggered": triggered[ftype], "total": counts[ftype],
                "rate_pct": round(100 * triggered[ftype] / max(counts[ftype], 1), 1)}
        for ftype in counts
    }

    output = {
        "cases": results,
        "taxonomy": taxonomy,
        "total_cases": total_cases,
        "total_failures": total_failures,
        "failure_rate_pct": round(100 * total_failures / total_cases, 1),
    }

    os.makedirs("experiments/results", exist_ok=True)
    out_path = "experiments/results/failure_cases.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")
    return output


if __name__ == "__main__":
    run_failure_analysis()
