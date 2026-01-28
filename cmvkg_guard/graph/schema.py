from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set, Any

@dataclass
class Node:
    """Base class for a node in the DH-MMKG."""
    id: str
    label: str
    type: str  # 'object', 'attribute', 'concept', 'external'
    confidence: float = 1.0
    attributes: Dict[str, Any] = field(default_factory=dict)
    source: str = "visual"  # 'visual', 'external_kb', 'reasoning'
    
@dataclass
class Edge:
    """Directed edge in the DH-MMKG."""
    source_id: str
    target_id: str
    relation: str
    confidence: float = 1.0
    source: str = "visual"

@dataclass
class SceneContext:
    """Scene-level context information."""
    scene_type: str = "unknown"
    scene_attributes: Dict[str, Any] = field(default_factory=dict)
    
@dataclass
class DHMMKG:
    """Dynamic Hierarchical Multimodal Knowledge Graph."""
    nodes: Dict[str, Node] = field(default_factory=dict)
    edges: List[Edge] = field(default_factory=list)
    scene_context: SceneContext = field(default_factory=SceneContext)
    
    def add_node(self, node: Node):
        self.nodes[node.id] = node
        
    def add_edge(self, edge: Edge):
        self.edges.append(edge)
        
    def get_node(self, node_id: str) -> Optional[Node]:
        return self.nodes.get(node_id)
        
    def get_edges_from(self, node_id: str) -> List[Edge]:
        return [e for e in self.edges if e.source_id == node_id]

    def get_edges_to(self, node_id: str) -> List[Edge]:
        return [e for e in self.edges if e.target_id == node_id]

    def to_dict(self):
        return {
            "nodes": {k: v.__dict__ for k, v in self.nodes.items()},
            "edges": [e.__dict__ for e in self.edges],
            "scene_context": self.scene_context.__dict__
        }
