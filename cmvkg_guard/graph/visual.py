import torch
from PIL import Image
from typing import List, Dict, Any, Tuple
from transformers import DetrImageProcessor, DetrForObjectDetection
from .schema import Node, Edge, SceneContext

class VisualSceneGraphExtractor:
    """Extracts scene graphs from images using Object Detection models."""
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        # Using DETR for robust object detection
        self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
        self.model.to(device)

    def extract(self, image: Image.Image) -> Tuple[List[Node], List[Edge], SceneContext]:
        """
        Main method to process image and return graph components.
        """
        # 1. Detect Objects
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Convert outputs (bounding boxes and class logits) to COCO API
        # Let's keep only high confidence predictions
        target_sizes = torch.tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

        nodes = []
        edges = []
        
        detected_objects = []

        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            label_name = self.model.config.id2label[label.item()]
            
            node_id = f"obj_{len(nodes)}"
            node = Node(
                id=node_id,
                label=label_name,
                type="object",
                confidence=score.item(),
                attributes={"bbox": box},
                source="visual"
            )
            nodes.append(node)
            detected_objects.append((node_id, box))

        # 2. Simple Spatial Relations (Heuristic based on bboxes)
        # In a real system, use a Scene Graph Generation model (e.g., motifs)
        for i, (id1, box1) in enumerate(detected_objects):
            for j, (id2, box2) in enumerate(detected_objects):
                if i == j: continue
                
                relation = self._compute_spatial_relation(box1, box2)
                if relation:
                    edges.append(Edge(
                        source_id=id1,
                        target_id=id2,
                        relation=relation,
                        confidence=0.8, # Heuristic confidence
                        source="visual"
                    ))

        # 3. Scene Context (Placeholder)
        # Could use a scene classification model here
        scene_context = SceneContext(scene_type="general", scene_attributes={})
        
        return nodes, edges, scene_context

    def _compute_spatial_relation(self, box1: List[float], box2: List[float]) -> str:
        """Simple heuristic for spatial relation."""
        # box: [xmin, ymin, xmax, ymax]
        # Center points
        c1_x = (box1[0] + box1[2]) / 2
        c1_y = (box1[1] + box1[3]) / 2
        c2_x = (box2[0] + box2[2]) / 2
        c2_y = (box2[1] + box2[3]) / 2
        
        if c1_x < box2[0]: # box1 is completely left of box2
             return "left of"
        elif c1_x > box2[2]: # box1 is completely right of box2
             return "right of"
        
        # Checking above/below requires flipping y-axis thinking usually, 
        # but image coords increase downwards.
        if c1_y < box2[1]: # box1 is above box2 (lower y value)
             return "above"
        elif c1_y > box2[3]:
             return "below"
             
        return None
