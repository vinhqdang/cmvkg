from typing import Dict, Any
from ..graph.schema import DHMMKG

class KnowledgeGroundingVerifier:
    """Layer 2: Verifies tokens against the DH-MMKG."""
    
    def verify(self, token: str, graph: DHMMKG) -> Dict[str, float]:
        """
        Checks if the token exists as a node or attribute in the graph.
        """
        token_lower = token.lower().strip()
        
        # 1. Entity Existence Check
        entity_exists = False
        for node in graph.nodes.values():
            if token_lower in node.label.lower():
                entity_exists = True
                break
        
        # 2. Attribute Consistency (Simplified)
        # Check if token is an attribute of a recently mentioned entity?
        # For now, just check if it appears in any attribute values
        attribute_exists = False
        if not entity_exists:
            for node in graph.nodes.values():
                for attr_val in node.attributes.values():
                    if isinstance(attr_val, str) and token_lower in attr_val.lower():
                        attribute_exists = True
                        break
                    elif isinstance(attr_val, list):
                        for item in attr_val:
                            if str(item) == token_lower: # Exact match for bbox numbers? unlikely for single token
                                pass
        
        return {
            "entity_check": 1.0 if entity_exists else 0.0,
            "attribute_check": 1.0 if attribute_exists else 0.0,
            "relation_path": 0.5 # Placeholder
        }
