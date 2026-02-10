import torch
from PIL import Image
from typing import Dict, Any, List, Optional
from transformers import AutoProcessor, LlavaForConditionalGeneration, TextIteratorStreamer, BitsAndBytesConfig
from threading import Thread

from .config import CMVKGConfig
from .graph.builder import DHMMKGBuilder
from .verification.engine import UnifiedVerificationEngine
from .correction.calibrator import AdaptiveCalibrator
from .correction.corrector import RealTimeCorrector
from .correction.explainer import ExplanationGenerator

class CMVKGGuard:
    """
    Main middleware for CMVKG-Guard using real LLaVA model.
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
        
        # Load Model
        print(f"Loading VLM: {config.vlm_model_path} (4-bit)...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        
        self.processor = AutoProcessor.from_pretrained(config.vlm_model_path)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            config.vlm_model_path, 
            quantization_config=quantization_config,
            device_map="auto"
        )
        print("Model loaded.")
        
    def generate(self, image: Image.Image, query: str, max_tokens: int = 50) -> Dict[str, Any]:
        """
        Runs the guarded generation process with real VLM.
        """
        # 1. Build DH-MMKG
        graph = self.graph_builder.build(image, query)
        
        # Prepare inputs
        prompt = f"USER: <image>\n{query}\nASSISTANT:"
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        
        # Explicitly move to device
        input_ids = inputs.input_ids.to(self.config.device)
        pixel_values = inputs.pixel_values.to(self.config.device)
        attention_mask = inputs.attention_mask.to(self.config.device) if "attention_mask" in inputs else None
        
        # We need to intercept generation token by token.
        # So we simply loop auto-regressively.
        
        generated_ids = input_ids
        generated_tokens = [] # List of strings
        corrections = []
        full_text = ""
        
        print("Starting generation...")
        
        with torch.no_grad():
            for _ in range(max_tokens):
                # Forward pass
                outputs = self.model(
                    input_ids=generated_ids,
                    pixel_values=pixel_values,
                    attention_mask=torch.ones_like(generated_ids) # Simple mask
                )
                
                # Grease decoding: get next token logits
                next_token_logits = outputs.logits[:, -1, :]
                next_token_id = torch.argmax(next_token_logits, dim=-1)
                
                # Decode
                token_text = self.processor.decode(next_token_id, skip_special_tokens=True)
                # Note: decode might strip spaces, so we need careful handling, 
                # but for this prototype we'll assume basic tokenization.
                # Actually, LLaVA/Llama tokenizer handles spaces as part of tokens usually (SPIECE_UNDERLINE)
                
                # Check for EOS
                if next_token_id.item() == self.processor.tokenizer.eos_token_id:
                    break
                    
                current_context = full_text
                
                # 2. Verify
                uvs, breakdown = self.verification_engine.verify_token(token_text, image, graph, current_context)
                
                # 3. Calibrate Threshold
                threshold = self.calibrator.compute_threshold()
                
                # 4. Decide & Correct
                final_token_text = token_text
                
                if uvs < threshold:
                    # Hallucination detected
                    corrected_text, expl, alts = self.corrector.correct(token_text, graph)
                    
                    if corrected_text != token_text:
                        final_token_text = corrected_text
                        corrections.append({
                            "original": token_text,
                            "corrected": corrected_text,
                            "explanation": expl,
                            "uvs": uvs,
                            "threshold": threshold
                        })
                        
                        # We need to re-encode the corrected token to feed back into model
                        # This is tricky because "remote" might be multiple tokens.
                        # For simplicity, we just APPEND the text to our tracking list
                        # and re-tokenize the WHOLE sequence for the next step?
                        # Re-tokenizing whole sequence is safer for consistency.
                        pass

                generated_tokens.append(final_token_text)
                full_text += final_token_text + " " # Naive spacing
                
                # Prepare input for next step
                # We intentionally feed back the CORRECTED text (teacher forcing the correction)
                # So the model conditions on the correction.
                new_input_text = prompt + full_text
                new_inputs = self.processor(text=new_input_text, images=image, return_tensors="pt").to(self.config.device)
                generated_ids = new_inputs.input_ids # Update history
                
        return {
            "generated_text": full_text.strip(),
            "original_vlm_text": "(Real VLM used - original path not tracked separately in this loop)", 
            "corrections": corrections,
            "graph_stats": {
                "nodes": len(graph.nodes),
                "edges": len(graph.edges)
            }
        }
