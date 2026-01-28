import torch
from PIL import Image
from typing import Dict, Any, List, Optional
from .config import CMVKGConfig
from .graph.builder import DHMMKGBuilder
from .verification.engine import UnifiedVerificationEngine
from .correction.calibrator import AdaptiveCalibrator
from .correction.corrector import RealTimeCorrector
from .correction.explainer import ExplanationGenerator

class CMVKGGuard:
    """
    Main middleware for CMVKG-Guard.
    """
    
    def __init__(self, config: Optional[CMVKGConfig] = None):
        if config is None:
            config = CMVKGConfig()
        self.config = config
        
        # Initialize modules
        self.graph_builder = DHMMKGBuilder(device=config.device)
        self.verification_engine = UnifiedVerificationEngine(config)
        self.calibrator = AdaptiveCalibrator(config)
        self.corrector = RealTimeCorrector(config)
        self.explainer = ExplanationGenerator()
        
    def generate(self, image: Image.Image, query: str, max_tokens: int = 50) -> Dict[str, Any]:
        """
        Simulates the guarded generation process.
        In a real scenario, this would wrap `model.generate`.
        """
        # 1. Build DH-MMKG
        graph = self.graph_builder.build(image, query)
        
        generated_tokens = []
        corrections = []
        
        # Pseudo-generation loop (mocking a VLM for demonstration)
        # We'll pretend the VLM generates a sequence "The cat is flying on the table"
        # "flying" should be hallucinations if visual graph has "sitting".
        
        mock_vlm_output = ["The", "cat", "is", "flying", "on", "the", "table"]
        
        current_context = ""
        
        for token in mock_vlm_output:
            # 2. Verify
            uvs, breakdown = self.verification_engine.verify_token(token, image, graph, current_context)
            
            # 3. Calibrate Threshold
            threshold = self.calibrator.compute_threshold()
            
            # 4. Decide
            final_token = token
            if uvs < threshold:
                # Hallucination detected
                corrected, expl, alts = self.corrector.correct(token, graph)
                
                # If correction found and better (placeholder check)
                if corrected != token:
                    final_token = corrected
                    corrections.append({
                        "original": token,
                        "corrected": corrected,
                        "explanation": expl,
                        "uvs": uvs,
                        "threshold": threshold
                    })
            
            generated_tokens.append(final_token)
            current_context += final_token + " "
            
        return {
            "generated_text": " ".join(generated_tokens),
            "original_vlm_text": " ".join(mock_vlm_output), # For comparison in demo
            "corrections": corrections,
            "graph_stats": {
                "nodes": len(graph.nodes),
                "edges": len(graph.edges)
            }
        }
