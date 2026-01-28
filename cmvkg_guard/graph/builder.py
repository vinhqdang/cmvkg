from PIL import Image
from typing import Optional, List
from .schema import DHMMKG, Node
from .visual import VisualSceneGraphExtractor
from .knowledge import KnowledgeIntegrator

class DHMMKGBuilder:
    """Builder for the Dynamic Hierarchical Multimodal Knowledge Graph."""
    
    def __init__(self, device: str = "cpu"):
        self.visual_extractor = VisualSceneGraphExtractor(device=device)
        self.knowledge_integrator = KnowledgeIntegrator()
        
    def build(self, image: Image.Image, query: str) -> DHMMKG:
        """
        Builds the initial DH-MMKG from image and query.
        """
        graph = DHMMKG()
        
        # 1. Visual Phase
        nodes, edges, context = self.visual_extractor.extract(image)
        for node in nodes:
            graph.add_node(node)
        for edge in edges:
            graph.add_edge(edge)
        graph.scene_context = context
        
        # 2. Knowledge Enrichment Phase (Initial)
        # Enrich detected objects
        for node in nodes:
            if node.type == "object":
                enrichment = self.knowledge_integrator.enrich_entity(node.label)
                if enrichment.get("conceptnet_relations"):
                    # Add external knowledge as attribute for now, or new nodes
                    # For simplicity, let's add them as attributes
                    node.attributes["external_knowledge"] = enrichment
        
        # 3. Query Analysis (Simple entity extraction from query)
        # In real system, use NER
        query_terms = query.split() # Very naive
        for term in query_terms:
            if len(term) > 3: # Filter strict stopwords roughly
                # Check if this term matches any object
                matched = False
                for node in nodes:
                    if term.lower() in node.label.lower():
                        matched = True
                        break
                
                if not matched:
                    # It's a concept from the text not in image (yet)
                    # Add as external concept node
                    enrichment = self.knowledge_integrator.enrich_entity(term)
                    
                    new_node = Node(
                        id=f"ext_{term}", 
                        label=term,
                        type="concept",
                        source="external_kb",
                        attributes={"external_knowledge": enrichment}
                    )
                    graph.add_node(new_node)
                    
        return graph
