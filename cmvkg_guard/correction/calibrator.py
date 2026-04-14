from typing import Dict, Any, Optional
from ..config import CMVKGConfig

class AdaptiveCalibrator:
    """Computes dynamic verification thresholds."""
    
    def __init__(self, config: CMVKGConfig):
        self.config = config
        
    def compute_threshold(self, 
                          query_complexity: float = 0.5, 
                          domain_risk: float = 0.5, 
                          kg_density: float = 0.5,
                          history_accuracy: float = 1.0,
                          generation_step: int = 1) -> float:
        """
        Dynamically computes threshold based on context.
        """
        # Base threshold from config
        theta = self.config.base_threshold
        
        # High Risk domain -> Increase threshold (Catch more potential hallucinations)
        risk_factor = 0.1 * (domain_risk - 0.5)
        
        # High History Accuracy -> Decrease threshold (Trust model more)
        history_factor = -0.1 * (history_accuracy - 0.5) 
        
        # High Query Complexity -> Higher strictness needed
        complexity_factor = 0.05 * (query_complexity - 0.5)
        
        # Generation Step factor: later tokens are more prone to snowballing hallucinations
        # We slowly increase strictness for long generations.
        step_factor = min(0.1, 0.01 * generation_step)
        
        # Calculate final dynamic threshold
        final_theta = theta + risk_factor + history_factor + complexity_factor + step_factor
        
        # Keep threshold neatly bounded between 0.1 and 0.9
        return max(0.1, min(0.9, final_theta))
