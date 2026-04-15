import os
import json
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

def _load_settings() -> Dict[str, Any]:
    """Load settings from settings.local.json (preferred) or settings.json."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for fname in ["settings.local.json", "settings.json"]:
        path = os.path.join(base_dir, fname)
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    return {}

_SETTINGS = _load_settings()

@dataclass
class CMVKGConfig:
    """Configuration for CMVKG-Guard. Values loaded from settings.local.json > settings.json > env vars > defaults."""

    import torch

    # Model Configuration
    vlm_model_path: str = _SETTINGS.get("vlm", {}).get("default_model", "llava-hf/llava-1.5-7b-hf")
    device: str = ("cuda" if torch.cuda.is_available() else "cpu") if _SETTINGS.get("device", "auto") == "auto" else _SETTINGS.get("device", "cpu")

    # API Keys — loaded from settings file; fallback to environment variables
    huggingface_token: Optional[str] = (_SETTINGS.get("vlm", {}).get("huggingface_token") or os.environ.get("HF_TOKEN"))
    openai_api_key: Optional[str] = (_SETTINGS.get("vlm", {}).get("openai_api_key") or os.environ.get("OPENAI_API_KEY"))
    openai_base_url: str = _SETTINGS.get("vlm", {}).get("openai_base_url", "https://api.openai.com/v1")
    qwen_api_key: Optional[str] = (_SETTINGS.get("vlm", {}).get("qwen_api_key") or os.environ.get("DASHSCOPE_API_KEY"))
    qwen_base_url: str = _SETTINGS.get("vlm", {}).get("qwen_base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    together_api_key: Optional[str] = (_SETTINGS.get("vlm", {}).get("together_api_key") or os.environ.get("TOGETHER_API_KEY"))
    replicate_api_key: Optional[str] = (_SETTINGS.get("vlm", {}).get("replicate_api_key") or os.environ.get("REPLICATE_API_TOKEN"))

    # DH-MMKG Configuration
    wikidata_endpoint: str = _SETTINGS.get("knowledge_graph", {}).get("wikidata_endpoint", "https://query.wikidata.org/sparql")
    conceptnet_endpoint: str = _SETTINGS.get("knowledge_graph", {}).get("conceptnet_endpoint", "http://api.conceptnet.io")
    cache_dir: str = _SETTINGS.get("knowledge_graph", {}).get("cache_dir", ".cmvkg_cache")

    # Verification Weights
    weight_visual: float = _SETTINGS.get("verification", {}).get("weight_visual", 0.3)
    weight_kg: float = _SETTINGS.get("verification", {}).get("weight_kg", 0.4)
    weight_reasoning: float = _SETTINGS.get("verification", {}).get("weight_reasoning", 0.3)

    # Correction Configuration
    base_threshold: float = _SETTINGS.get("verification", {}).get("base_threshold", 0.7)
    correction_lambda: float = _SETTINGS.get("verification", {}).get("correction_lambda", 0.6)

    # Ablation Flags
    use_external_kg: bool = True
    dynamic_kg: bool = True

    def __post_init__(self):
        # Environment variable overrides (highest priority)
        if os.environ.get("CMVKG_VLM_MODEL"):
            self.vlm_model_path = os.environ["CMVKG_VLM_MODEL"]
        # Set HuggingFace token for model downloads
        if self.huggingface_token and not self.huggingface_token.startswith("YOUR_"):
            os.environ.setdefault("HF_TOKEN", self.huggingface_token)
