import math
import difflib
from typing import Dict, List, Any, Set
from ..graph.schema import DHMMKG

class ReasoningVerifier:
    """Layer 3: Verifies logical, temporal, and multi-hop reasoning (Eq. 7)."""

    def _string_similarity(self, a: str, b: str) -> float:
        return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()

    def _bfs_multi_hop(self, token: str, graph: DHMMKG, max_depth: int = 3) -> float:
        """Finds path confidence score traversing up to max_depth."""
        # Find start nodes matching token
        start_nodes = []
        for n_id, node in graph.nodes.items():
            if self._string_similarity(token, node.label) > 0.6:
                start_nodes.append(n_id)
        
        if not start_nodes:
            # Token doesn't directly map to a node, check if it maps to edges
            for edge in graph.edges:
                if self._string_similarity(token, edge.relation) > 0.6:
                    start_nodes.extend([edge.source_id, edge.target_id])
            if not start_nodes:
                return 0.5 # Default uncertainty
            
        best_path_score = 0.0
        visited: Set[str] = set()
        queue = [(n, 1.0, 0) for n in start_nodes] # (node_id, current_confidence, depth)

        while queue:
            current_id, conf, depth = queue.pop(0)
            if depth > max_depth:
                continue
                
            best_path_score = max(best_path_score, conf)
            visited.add(current_id)
            
            # Get neighbors (both outgoing and incoming for undirected connectivity heuristic)
            edges_out = graph.get_edges_from(current_id)
            for edge in edges_out:
                if edge.target_id not in visited:
                    next_conf = conf * getattr(edge, 'confidence', 1.0) * 0.9 # decay factor
                    queue.append((edge.target_id, next_conf, depth + 1))
                    
            edges_in = graph.get_edges_to(current_id)
            for edge in edges_in:
                if edge.source_id not in visited:
                    next_conf = conf * getattr(edge, 'confidence', 1.0) * 0.9
                    queue.append((edge.source_id, next_conf, depth + 1))
                    
        return best_path_score

    def verify(self, token: str, context: str, graph: DHMMKG) -> Dict[str, float]:
        """
        Verify reasoning consistency using structural entailment Eq. 7:
        S_{reason}(t_i) = \sigma( \sum_{p \in T} \phi(p) * I(t_i |= p) )
        """
        temporal_score = 1.0
        temporal_keywords = ["before", "after", "then", "later", "subsequently", "previously"]
        if any(keyword in token.lower() for keyword in temporal_keywords):
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
                
                # I(t_i |= p) - Entailment indicator function using SequenceMatcher
                rel_sim = self._string_similarity(token, edge.relation)
                tgt_sim = self._string_similarity(token, edge.target_id)
                src_sim = self._string_similarity(token, edge.source_id)
                
                max_sim = max(rel_sim, tgt_sim, src_sim)
                indicator = max_sim if max_sim > 0.4 else 0.1 # Threshold for partial entailment
                    
                entailment_sum += phi_p * indicator
                
            # Sigma (sigmoid) activation to normalize the sum to [0,1]
            logic_score = 1 / (1 + math.exp(-entailment_sum))
            
        multi_hop_score = self._bfs_multi_hop(token, graph, max_depth=3)
        if multi_hop_score == 0.0:
            multi_hop_score = logic_score * 0.9 # Approximation for multi-hop cascading structure
        
        return {
            "temporal": temporal_score,
            "logic": logic_score,
            "multi_hop": multi_hop_score
        }
