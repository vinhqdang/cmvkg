import math
from typing import Dict, List, Any
from ..graph.schema import DHMMKG

class ReasoningVerifier:
    """Layer 3: Verifies logical, temporal, and multi-hop reasoning (Eq. 7)."""

    def verify(self, token: str, context: str, graph: DHMMKG) -> Dict[str, float]:
        """
        Verify reasoning consistency using structural entailment Eq. 7:
        S_{reason}(t_i) = \sigma( \sum_{p \in T} \phi(p) * I(t_i |= p) )
        """
        temporal_score = 1.0
        temporal_keywords = ["before", "after", "then", "later"]
        if token.lower() in temporal_keywords:
             temporal_score = 0.5 
             
        # Generate logical predicates T_ti from context
        # Simplified abstraction: We treat graph edges as the logical predicates
        T_ti = graph.edges
        
        logic_score = 0.0
        if not T_ti:
            logic_score = 0.8 # Default baseline if no complex edges exist to refute
        else:
            # Implement the summation from Eq. 7
            entailment_sum = 0.0
            
            for edge in T_ti:
                # phi(p) - heuristic path weight. We use edge confidence or default to 1.0
                phi_p = getattr(edge, 'confidence', 1.0)
                
                # I(t_i |= p) - Entailment indicator function
                # Naive text overlap as proxy for lightweight NLI entailment
                if token.lower() in edge.relation.lower() or token.lower() in edge.target_id.lower():
                    indicator = 1.0
                else:
                    indicator = 0.1 # Partial/No entailment
                    
                entailment_sum += phi_p * indicator
                
            # Sigma (sigmoid) activation to normalize the sum to [0,1]
            logic_score = 1 / (1 + math.exp(-entailment_sum))
        
        return {
            "temporal": temporal_score,
            "logic": logic_score,
            "multi_hop": logic_score * 0.9 # Approximation for multi-hop cascading structure
        }
