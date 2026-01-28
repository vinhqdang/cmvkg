class ExplanationGenerator:
    """Generates natural language explanations for corrections."""
    
    def generate(self, original: str, corrected: str, reason: str, evidence: str = "") -> str:
        return f"Hallucination detected: Model generated '{original}', but visual evidence supports '{corrected}'. {reason}"
