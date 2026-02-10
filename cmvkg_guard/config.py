import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

@dataclass
class CMVKGConfig:
    """Configuration for CMVKG-Guard."""
    
    # Model Configuration
    vlm_model_path: str = "llava-hf/llava-1.5-7b-hf"
    import torch
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # DH-MMKG Configuration
    wikidata_endpoint: str = "https://query.wikidata.org/sparql"
    conceptnet_endpoint: str = "http://api.conceptnet.io"
    
    # Verification Weights
    weight_visual: float = 0.3
    weight_kg: float = 0.4
    weight_reasoning: float = 0.3
    
    # Correction Configuration
    base_threshold: float = 0.7
    correction_lambda: float = 0.6
    
    def __post_init__(self):
        # Allow environment variable overrides
         if os.environ.get("CMVKG_VLM_MODEL"):
             self.vlm_model_path = os.environ.get("CMVKG_VLM_MODEL")
