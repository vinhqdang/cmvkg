import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
from PIL import Image
from typing import Dict, Any, List
from .base_vlm import BaseVLMDriver

class LLaVADriver(BaseVLMDriver):
    """Driver for LLaVA-1.5, preserving actual Hugging Face loading."""
    
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self, model_path: str, device: str = "cuda"):
        self.device = device
        print(f"Loading LLaVA: {model_path} (4-bit)...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_path, 
            quantization_config=quantization_config,
            device_map="auto"
        )

    def generate_token(self, image: Image.Image, history: str) -> Dict[str, Any]:
        if not self.model:
            raise RuntimeError("Model not loaded.")
        prompt = f"USER: <image>\n{history}"
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs.input_ids,
                pixel_values=inputs.pixel_values,
                attention_mask=inputs.attention_mask
            )
            next_token_logits = outputs.logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1)
            token_text = self.processor.decode(next_token_id, skip_special_tokens=True)
            eos = (next_token_id.item() == self.processor.tokenizer.eos_token_id)
            
        return {"token": token_text, "logits": next_token_logits, "eos": eos}

    def get_token_logprobs(self, image: Image.Image, history: str, candidate_tokens: List[str]) -> List[float]:
        """Returns dummy logprobs for beam search simulation to avert massive computation overhead during prototype."""
        # In a real VLM, we would concatenate each speculative token and run forward pass to get sequence logprob.
        # For the scope of this implementation demo, we simulate the logprob heuristic.
        import math
        return [math.log(0.1 + 0.9 / (i + 1)) for i in range(len(candidate_tokens))]
