import torch
import requests
from PIL import Image
from io import BytesIO
import time
import json
from cmvkg_guard.pipeline import CMVKGGuard

def run_experiment():
    print("--- Starting Experiment ---")
    
    # Initialize Guard
    start_time = time.time()
    guard = CMVKGGuard()
    print(f"Initialization took: {time.time() - start_time:.2f}s")
    
    # Dataset: A few samples designed to trigger hallucinations or test grounding
    # 1. The Cat image (COCO)
    # 2. A different image if possible, but let's stick to reliable URLs
    
    samples = [
        {
            "id": 1,
            "image_url": "http://images.cocodataset.org/val2017/000000039769.jpg", # Cats on pink couch
            "query": "What are the cats doing?",
            "description": "Standard cat image"
        },
        {
             "id": 2,
             "image_url": "http://images.cocodataset.org/val2017/000000039769.jpg",
             "query": "Is there a dog in the image?", 
             "description": "Hallucination probe (Dog)"
        },
        {
            "id": 3,
            "image_url": "https://farm4.staticflickr.com/3133/3378902101_3c9fa16b84_z.jpg", # Bus
            "query": "What color is the bus?",
            "description": "Attribute verification"
        }
    ]
    
    results = []
    
    for sample in samples:
        print(f"\nProcessing Sample {sample['id']}: {sample['query']}")
        try:
            response = requests.get(sample['image_url'], timeout=10)
            image = Image.open(BytesIO(response.content))
            
            t0 = time.time()
            output = guard.generate(image, sample['query'], max_tokens=30)
            inference_time = time.time() - t0
            
            result_entry = {
                "sample_id": sample['id'],
                "query": sample['query'],
                "generated_text": output["generated_text"],
                "corrections_count": len(output["corrections"]),
                "corrections": output["corrections"],
                "inference_time": inference_time,
                "graph_stats": output["graph_stats"]
            }
            results.append(result_entry)
            print(f"Generated: {output['generated_text']}")
            print(f"Corrections: {len(output['corrections'])}")
            
        except Exception as e:
            print(f"Error processing sample {sample['id']}: {e}")
            
    # Save results
    with open("experimental_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("\n--- Experiment Complete ---")
    print("Results saved to experimental_results.json")

if __name__ == "__main__":
    run_experiment()
