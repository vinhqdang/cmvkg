from .base_vlm import BaseVLMDriver
from .llava_driver import LLaVADriver
from .qwen_driver import QwenVLDriver
from .instructblip_driver import InstructBLIPDriver

def get_vlm_driver(model_name: str) -> BaseVLMDriver:
    model_name = model_name.lower()
    if "llava" in model_name:
        return LLaVADriver()
    elif "qwen" in model_name:
        return QwenVLDriver()
    elif "instructblip" in model_name:
        return InstructBLIPDriver()
    else:
        raise ValueError(f"Unsupported model: {model_name}")
