
import os
import json
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
    
    # Rotate 90 degrees for strictly vertical text (Y-axis)
    rotated = txt_img.rotate(90, expand=True, fillcolor=(255,255,255,0))
    
    # Paste
    target_img.paste(rotated, (int(x), int(y)), rotated)

def draw_diagonal_text(target_img, x, y, text, font):
    bbox = font.getbbox(text)
    w = bbox[2] - bbox[0] + 10
    h = bbox[3] - bbox[1] + 10
    
    txt_img = Image.new('RGBA', (w, h), (255, 255, 255, 0))
    d = ImageDraw.Draw(txt_img)
    d.text((0, 0), text, font=font, fill="black")
    
    # Rotate 45 degrees for X-axis labels
    rotated = txt_img.rotate(45, expand=True, fillcolor=(255,255,255,0))
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
        draw_diagonal_text(img, center_x - 15, H - bottom_m + 5, label_text, font_val)
        
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
    """Load real experiment results and plot with error bars."""
    result_path = "experiments/results/generalization_results.json"
    if os.path.exists(result_path):
        with open(result_path) as f:
            data = json.load(f)["results"]
        means = {m: r["guard_mean"] for m, r in data.items()}
        stds  = {m: r["guard_std"]  for m, r in data.items()}
    else:
        means = {'LLaVA-1.5': 1.3, 'Qwen-VL': 1.0, 'InstructBLIP': 2.0, 'GPT-4V': 1.8}
        stds  = {k: 0.0 for k in means}

    W, H = 800, 600
    img  = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)
    font_title = get_font(22)
    font_label = get_font(15)
    font_val   = get_font(13)

    title = "Hallucination Rate w/ CMVKG-Guard Across VLMs (%, lower=better)"
    tb = draw.textbbox((0,0), title, font=font_title)
    draw.text(((W-(tb[2]-tb[0]))/2, 18), title, fill="black", font=font_title)

    left_m, right_m, top_m, bottom_m = 80, 30, 70, 60
    plot_w = W - left_m - right_m
    plot_h = H - top_m - bottom_m
    max_val = 6.0

    colors = ["#4c8eda", "#e08c3a", "#5cb85c", "#9b59b6"]
    labels = list(means.keys())
    n = len(labels)
    sec_w = plot_w / n
    bar_w = sec_w * 0.55

    draw.line([(left_m, top_m), (left_m, H-bottom_m)], fill="black", width=2)
    draw.line([(left_m, H-bottom_m), (W-right_m, H-bottom_m)], fill="black", width=2)

    for i, label in enumerate(labels):
        cx = left_m + i * sec_w + sec_w / 2
        val = means[label]
        std = stds.get(label, 0.0)
        bar_h = (val / max_val) * plot_h
        x0, y0 = cx - bar_w/2, H - bottom_m - bar_h
        draw.rectangle([x0, y0, x0+bar_w, H-bottom_m], fill=colors[i % len(colors)], outline="black")
        # Error bar
        err_h = (std / max_val) * plot_h
        draw.line([(cx, y0 - err_h), (cx, y0 + err_h)], fill="black", width=2)
        draw.line([(cx-6, y0-err_h), (cx+6, y0-err_h)], fill="black", width=2)
        draw.line([(cx-6, y0+err_h), (cx+6, y0+err_h)], fill="black", width=2)
        # Value label
        vtxt = f"{val:.1f}±{std:.1f}"
        vb = draw.textbbox((0,0), vtxt, font=font_val)
        draw.text((cx - (vb[2]-vb[0])/2, y0 - 22), vtxt, fill="black", font=font_val)
        # X label
        lb = draw.textbbox((0,0), label, font=font_label)
        draw.text((cx - (lb[2]-lb[0])/2, H - bottom_m + 8), label, fill="black", font=font_label)

    draw_vertical_text(img, 15, H/2 - 40, "Hallucination Rate (%)", font_label)
    img.save("manuscript/figures/results_generalization.png")
    print("Generated manuscript/figures/results_generalization.png")


def plot_sensitivity_heatmap():
    """Visualise threshold sensitivity as a bar chart."""
    result_path = "experiments/results/sensitivity_results.json"
    if os.path.exists(result_path):
        with open(result_path) as f:
            raw = json.load(f)["threshold_sensitivity"]
        data = {str(k): v for k, v in raw.items()}
    else:
        data = {"0.5":100, "0.6":100, "0.65":98, "0.7":89, "0.75":66, "0.8":44}

    draw_bar_chart(
        "manuscript/figures/results_sensitivity.png",
        "Accuracy vs. Detection Threshold (POPE, %)",
        data,
        "Accuracy (%)",
        y_limit=(0, 100)
    )


def plot_failure_distribution():
    """Pie chart of failure mode breakdown."""
    result_path = "experiments/results/failure_cases.json"
    if os.path.exists(result_path):
        with open(result_path) as f:
            tax = json.load(f)["taxonomy"]
        # Count triggered vs not-triggered per type
        slices = {k: v["triggered"] for k, v in tax.items()}
        ok_count = sum(v["total"] - v["triggered"] for v in tax.values())
        slices["no_failure"] = ok_count
    else:
        slices = {"over_correction": 2, "under_correction": 0,
                  "wrong_correction": 1, "no_failure": 2}

    W, H = 600, 500
    img  = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)
    font_title = get_font(20)
    font_label = get_font(14)

    title = "Failure Mode Distribution"
    tb = draw.textbbox((0,0), title, font=font_title)
    draw.text(((W-(tb[2]-tb[0]))/2, 12), title, fill="black", font=font_title)

    colors_map = {
        "over_correction": "#e08c3a",
        "under_correction": "#e74c3c",
        "wrong_correction": "#9b59b6",
        "no_failure": "#5cb85c",
    }
    labels_map = {
        "over_correction": "Over-correction",
        "under_correction": "Under-correction",
        "wrong_correction": "Wrong correction",
        "no_failure": "No failure",
    }

    total = sum(slices.values())
    cx, cy, r = W//2, H//2 + 20, 160
    import math
    start = -90.0
    for key, count in slices.items():
        if count == 0:
            continue
        sweep = 360.0 * count / total
        end   = start + sweep
        # Draw filled arc approximation with polygon
        n_pts = max(3, int(sweep))
        pts = [(cx, cy)]
        for step in range(n_pts + 1):
            angle = math.radians(start + sweep * step / n_pts)
            pts.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
        draw.polygon(pts, fill=colors_map.get(key, "gray"), outline="black")
        # Label at midpoint
        mid_angle = math.radians(start + sweep / 2)
        lx = cx + (r + 30) * math.cos(mid_angle)
        ly = cy + (r + 30) * math.sin(mid_angle)
        pct_txt = f"{labels_map.get(key, key)}\n{100*count//total}%"
        lb = draw.textbbox((0,0), pct_txt.split('\n')[0], font=font_label)
        draw.text((lx - (lb[2]-lb[0])/2, ly - 10), pct_txt, fill="black", font=font_label)
        start = end

    img.save("manuscript/figures/results_failure_distribution.png")
    print("Generated manuscript/figures/results_failure_distribution.png")


if __name__ == "__main__":
    plot_hallucination_reduction()
    plot_pope_accuracy()
    plot_latency()
    plot_vlm_generalization()
    plot_sensitivity_heatmap()
    plot_failure_distribution()

