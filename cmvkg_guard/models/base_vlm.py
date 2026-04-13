from abc import ABC, abstractmethod
from typing import Dict, Any, List
from PIL import Image
import torch

class BaseVLMDriver(ABC):
    """Abstract driver for multi-VLM support."""

    @abstractmethod
    def load_model(self, model_path: str, device: str = "cpu"):
        pass

    @abstractmethod
    def generate_token(self, image: Image.Image, history: str) -> Dict[str, Any]:
        """
        Generate a single token given image and text history.
        Returns {"token": str, "logits": torch.Tensor, "eos": bool}
        """
        pass
        
    @abstractmethod
    def get_token_logprobs(self, image: Image.Image, history: str, candidate_tokens: List[str]) -> List[float]:
        """
        Evaluate and return log probabilities for a speculative set of candidate tokens
        for Constrained Look-Ahead Beam Search (Eq. 10).
        """
        pass
