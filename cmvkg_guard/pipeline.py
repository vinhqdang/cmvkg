import torch
from PIL import Image
from typing import Dict, Any, List, Optional
import time

from .config import CMVKGConfig
from .graph.builder import DHMMKGBuilder
from .verification.engine import UnifiedVerificationEngine
from .correction.calibrator import AdaptiveCalibrator
from .correction.corrector import RealTimeCorrector
from .models import get_vlm_driver

class CMVKGGuard:
    """Main middleware for CMVKG-Guard using modular VLM drivers."""
    
    def __init__(self, config: Optional[CMVKGConfig] = None):
        if config is None:
            config = CMVKGConfig()
        self.config = config
        
        # Initialize modules
        self.graph_builder = DHMMKGBuilder(device=config.device)
        self.verification_engine = UnifiedVerificationEngine(config)
        self.calibrator = AdaptiveCalibrator(config)
        self.corrector = RealTimeCorrector(config)
        
        # Load VLM Driver abstracting away specific implementation details
        self.vlm_driver = get_vlm_driver(config.vlm_model_path)
        self.vlm_driver.load_model(config.vlm_model_path, config.device)
        
    def generate(self, image: Image.Image, query: str, max_tokens: int = 50) -> Dict[str, Any]:
        """Runs the guarded generation process passing queries to the active VLM."""
        # 1. Build DH-MMKG
        graph = self.graph_builder.build(image, query)
        
        generated_tokens = []
        corrections = []
        full_text = ""
        
        print(f"Starting generation with {self.config.vlm_model_path}...")
        
        for _ in range(max_tokens):
            # Generate next speculative token
            out = self.vlm_driver.generate_token(image, full_text)
            token_text = out["token"]
            
            if out["eos"]:
                break
                
            current_context = full_text
            
            # 2. Verify against DH-MMKG
            uvs, breakdown = self.verification_engine.verify_token(token_text, image, graph, current_context)
            
            # 3. Calibrate Dynamic Threshold
            threshold = self.calibrator.compute_threshold()
            
            # 4. Decide & Correct
            final_token_text = token_text
            
            if uvs < threshold:
                # Hallucination detected, trigger Look-Ahead Beam Search (Eq. 10)
                corrected_text, expl, alts = self.corrector.correct(
                    hallucinated_token=token_text, 
                    graph=graph, 
                    vlm_driver=self.vlm_driver, 
                    context_history=current_context, 
                    image=image
                )
                
                if corrected_text != token_text:
                    final_token_text = corrected_text
                    corrections.append({
                        "original": token_text,
                        "corrected": corrected_text,
                        "explanation": expl,
                        "uvs": uvs,
                        "threshold": threshold
                    })
            
            generated_tokens.append(final_token_text)
            full_text += final_token_text + " "
            
        return {
            "generated_text": full_text.strip(),
            "corrections": corrections,
            "graph_stats": {
                "nodes": len(graph.nodes),
                "edges": len(graph.edges)
            }
        }
