import os
import requests
from PIL import Image
from io import BytesIO
from cmvkg_guard.pipeline import CMVKGGuard

def run_demo():
    print("Initializing CMVKG-Guard...")
    guard = CMVKGGuard()
    
    # Load a sample image
    url = "http://images.cocodataset.org/val2017/000000039769.jpg" # Two cats on pink couch
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    
    query = "What are the cats doing?"
    
    print(f"Processing query: '{query}'")
    print("Running guarded generation...")
    
    result = guard.generate(image, query)
    
    print("\n--- Results ---")
    print(f"Original VLM Output (Simulated): {result['original_vlm_text']}")
    print(f"Guarded Output: {result['generated_text']}")
    print(f"\nCorrections Made: {len(result['corrections'])}")
    for c in result['corrections']:
        print(f"- {c['explanation']} (UVS: {c['uvs']:.2f} < Thresh: {c['threshold']:.2f})")
        
    print(f"\nGraph Stats: {result['graph_stats']}")

if __name__ == "__main__":
    run_demo()
