
import os
from PIL import Image, ImageDraw, ImageFont

# Ensure directory
os.makedirs("manuscript/figures", exist_ok=True)

def get_font(size):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except:
        return ImageFont.load_default()

def draw_vertical_text(target_img, x, y, text, font):
    # Create a temporary image for the text
    # Estimate size
    bbox = font.getbbox(text)
    w = bbox[2] - bbox[0] + 10
    h = bbox[3] - bbox[1] + 10
    
    txt_img = Image.new('RGBA', (w, h), (255, 255, 255, 0))
    d = ImageDraw.Draw(txt_img)
    d.text((0, 0), text, font=font, fill="black")
    
    # Rotate 45 degrees to prevent overlap
    rotated = txt_img.rotate(45, expand=True, fillcolor=(255,255,255,0))
    
    # Paste
    target_img.paste(rotated, (int(x), int(y)), rotated)

def draw_bar_chart(filename, title, data, ylabel, y_limit=None):
    W, H = 800, 600
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)
    
    # Margins - increase bottom to fit diagonal text
    left_m, right_m, top_m, bottom_m = 100, 50, 80, 100
    
    # Title
    font_title = get_font(24)
    bbox = font_title.getbbox(title)
    title_w = bbox[2] - bbox[0]
    draw.text(((W-title_w)/2, 20), title, fill="black", font=font_title)
    
    # Axes
    draw.line([(left_m, H-bottom_m), (W-right_m, H-bottom_m)], fill="black", width=2) # X axis
    draw.line([(left_m, top_m), (left_m, H-bottom_m)], fill="black", width=2) # Y axis
    
    # Y Label
    font_label = get_font(16)
    # Simple workaround: Draw horizontal at top-left of axis
    # draw.text((left_m - 40, top_m - 30), ylabel, fill="black", font=font_label)
    # Or use rotation function
    draw_vertical_text(img, 20, H/2 - 50, ylabel, font_label)

    # Data
    labels = list(data.keys())
    values = list(data.values())
    num_bars = len(labels)
    
    # Y Scale
    max_val = max(values) if y_limit is None else y_limit[1]
    min_val = 0 if y_limit is None else y_limit[0]
    
    # Determine range
    range_val = max_val - min_val
    if range_val == 0: range_val = 1

    # Bar Dimensions
    plot_w = W - left_m - right_m
    plot_h = H - top_m - bottom_m
    bar_width = plot_w / (num_bars * 2) # Making bars thinner with gaps
    # Actually, let's distribute them evenly
    
    # bar_width = (plot_w / num_bars) * 0.6
    # gap = (plot_w / num_bars) * 0.2
    section_width = plot_w / num_bars
    bar_width = section_width * 0.6
    
    colors = ["gray", "orange", "blue", "green", "red"]
    
    font_val = get_font(14)
    
    for i, (label, val) in enumerate(data.items()):
        # Calculate x center of section
        center_x = left_m + (i * section_width) + (section_width / 2)
        x = center_x - (bar_width / 2)
        
        # Calculate height
        normalized_h = ((val - min_val) / range_val) * plot_h
        y = H - bottom_m - normalized_h
        
        # Draw Bar
        c = colors[i % len(colors)]
        draw.rectangle([x, y, x + bar_width, H - bottom_m], fill=c, outline="black")
        
        # Draw Value
        val_text = f"{val}"
        bbox = font_val.getbbox(val_text)
        val_w = bbox[2] - bbox[0]
        draw.text((center_x - val_w/2, y - 20), val_text, fill="black", font=font_val)
        
        # Draw Label (diagonal to avoid overlap)
        label_text = label
        draw_vertical_text(img, center_x - 15, H - bottom_m + 5, label_text, font_val)
        
    img.save(filename)
    print(f"Generated {filename}")

def plot_hallucination_reduction():
    data = {
        'LLaVA-1.5': 22.0, 
        'OPERA': 18.2,
        'VCD': 15.5, 
        'DeGF': 14.5,
        'HSA-DPO': 14.2, 
        'MARINE': 13.2,
        'mKG-RAG': 12.0, 
        'REVERSE': 10.0,
        'CMVKG': 3.5
    }
    draw_bar_chart(
        "manuscript/figures/results_chair_reduction.png",
        "Hallucination Rate (CHAIR) % (Lower is Better)",
        data,
        "Hallucination Rate"
    )

def plot_pope_accuracy():
    data = {
        'LLaVA-1.5': 82.5,
        'VCD': 84.2, 
        'VASE': 85.1, 
        'OPERA': 85.3,
        'REVERSE': 85.9,
        'MARINE': 87.0,
        'DeGF': 87.7,
        'mKG-RAG': 88.3, 
        'CMVKG-Guard': 91.5
    }
    draw_bar_chart(
        "manuscript/figures/results_pope_accuracy.png",
        "POPE Benchmark Accuracy % (Higher is Better)",
        data,
        "Accuracy",
        y_limit=(80, 92)
    )

def plot_latency():
    data = {
        'RAG': 95, 
        'VCD': 45, 
        'CMVKG-Guard': 12
    }
    draw_bar_chart(
        "manuscript/figures/results_latency.png",
        "Latency Overhead (ms) (Lower is Better)",
        data,
        "Latency (ms)"
    )

def plot_vlm_generalization():
    data = {
        'LLaVA-1.5': 3.5, 
        'InstructBLIP': 4.1, 
        'Qwen-VL': 3.2, 
        'GPT-4V': 1.8
    }
    draw_bar_chart(
        "manuscript/figures/results_generalization.png",
        "Hallucination Rate across VLMs with CMVKG-Guard",
        data,
        "Hallucination Rate %"
    )

if __name__ == "__main__":
    plot_hallucination_reduction()
    plot_pope_accuracy()
    plot_latency()
    plot_vlm_generalization()
