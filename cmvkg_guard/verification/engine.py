from typing import Dict, Any, Tuple
from PIL import Image
from ..config import CMVKGConfig
from ..graph.schema import DHMMKG
from .visual_textual import VisualTextualVerifier
from .knowledge_grounding import KnowledgeGroundingVerifier
from .reasoning import ReasoningVerifier

class UnifiedVerificationEngine:
    """Orchestrates all verification layers and computes UVS."""
    
    def __init__(self, config: CMVKGConfig):
        self.config = config
        self.visual_verifier = VisualTextualVerifier(device=config.device)
        self.kg_verifier = KnowledgeGroundingVerifier()
        self.reasoning_verifier = ReasoningVerifier()
        
    def verify_token(self, 
                     token: str, 
                     image: Image.Image, 
                     graph: DHMMKG,
                     context: str) -> Tuple[float, Dict[str, Any]]:
        """
        Runs all verification layers and returns UVS and breakdown.
        """
        # 0. Fast-track Stopwords
        # Common words often lack knowledge grounding but are valid.
        stopwords = {"the", "is", "are", "on", "in", "at", "of", "a", "an", "and", "to", "for", "with"}
        if token.lower().strip() in stopwords:
            return 1.0, {
                "layer1": {"semantic_sim": 1.0, "var_score": 1.0, "spatial_score": 1.0},
                "layer2": {"entity_check": 1.0, "attribute_check": 1.0, "relation_path": 1.0},
                "layer3": {"temporal": 1.0, "logic": 1.0, "multi_hop": 1.0},
                "totals": {"v1": 1.0, "v2": 1.0, "v3": 1.0}
            }

        # Layer 1
        v1_scores = self.visual_verifier.verify(token, image, context)
        v1_total = (0.3 * v1_scores["var_score"] + 
                    0.5 * v1_scores["semantic_sim"] + 
                    0.2 * v1_scores["spatial_score"])
        
        # Layer 2
        v2_scores = self.kg_verifier.verify(token, graph)
        v2_total = (0.4 * v2_scores["entity_check"] + 
                    0.3 * v2_scores["attribute_check"] + 
                    0.3 * v2_scores["relation_path"])
        
        # Layer 3
        v3_scores = self.reasoning_verifier.verify(token, context, graph)
        v3_total = (0.33 * v3_scores["temporal"] + 
                    0.33 * v3_scores["logic"] + 
                    0.33 * v3_scores["multi_hop"])
        
        # Compute UVS
        # Adaptive weighting logic can go here (based on query type)
        # Using config defaults for now
        uvs = (self.config.weight_visual * v1_total + 
               self.config.weight_kg * v2_total + 
               self.config.weight_reasoning * v3_total)
               
        breakdown = {
            "layer1": v1_scores,
            "layer2": v2_scores,
            "layer3": v3_scores,
            "totals": {"v1": v1_total, "v2": v2_total, "v3": v3_total}
        }
        
        return uvs, breakdown
