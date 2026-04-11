
import os
from PIL import Image, ImageDraw, ImageFont

# Ensure directory
os.makedirs("manuscript/figures", exist_ok=True)

def get_font(size):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except:
        return ImageFont.load_default()

def draw_box(draw, x, y, w, h, text, color="lightblue", outline="black"):
    draw.rectangle([x, y, x+w, y+h], fill=color, outline=outline, width=2)
    
    font = get_font(22)
    # text centering
    # getbbox returns (left, top, right, bottom)
    bbox = font.getbbox(text)
    # Use accurate dimension mapping
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    text_w = right - left
    text_h = bottom - top
    
    draw.text((x + (w-text_w)/2, y + (h-text_h)/2 - 5), text, fill="black", font=font)

def draw_arrow(draw, start, end):
    draw.line([start, end], fill="black", width=2)
    # Simple arrowhead
    # Not implementing complex arrowhead for simplicity, just a line or small circle
    draw.ellipse([end[0]-3, end[1]-3, end[0]+3, end[1]+3], fill="black")

def generate_cmve_flow():
    W, H = 1000, 800
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)
    
    # Title
    draw.text((30, 30), "Fig 3: Cross-Modal Verification Engine (CMVE) Flow", fill="black", font=get_font(24))

    # Input Node
    draw_box(draw, 400, 100, 200, 60, "Token Candidate (t)", color="#e0e0e0")

    # Split Arrows
    draw_arrow(draw, (500, 160), (200, 250)) # To L1
    draw_arrow(draw, (500, 160), (500, 250)) # To L2
    draw_arrow(draw, (500, 160), (800, 250)) # To L3

    # Layer 1
    draw_box(draw, 100, 250, 200, 100, "Layer 1:\nVisual-Textual\nAlignment", color="#ffcccc")
    draw.text((110, 360), "- VAR Score\n- Semantic Sim\n- Spatial Consistency", fill="black", font=get_font(14))

    # Layer 2
    draw_box(draw, 400, 250, 200, 100, "Layer 2:\nKnowledge Graph\nGrounding", color="#ccffcc")
    draw.text((410, 360), "- Entity Look-up\n- Rel. Path Check\n- Fact Verification", fill="black", font=get_font(14))

    # Layer 3
    draw_box(draw, 700, 250, 200, 100, "Layer 3:\nReasoning Chain\nVerification", color="#ccccff")
    draw.text((710, 360), "- Logical Entailment\n- Temporal Check\n- Causal Consistency", fill="black", font=get_font(14))

    # Merge to UVS
    draw_arrow(draw, (200, 450), (500, 550))
    draw_arrow(draw, (500, 450), (500, 550))
    draw_arrow(draw, (800, 450), (500, 550))

    # UVS Node
    draw_box(draw, 350, 550, 300, 80, "Unified Verification Score\n(UVS)", color="#ffffcc")

    # Decision
    draw_arrow(draw, (500, 630), (500, 700))
    draw_box(draw, 400, 700, 200, 60, "Decision:\nAccept / Correct", color="#e0e0e0")

    img.save("manuscript/figures/cmve_verification_flow.png")
    print("Generated cmve_verification_flow.png")

def generate_dh_mmkg_process():
    W, H = 800, 800
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)
    
    draw.text((30, 30), "Fig 2: DH-MMKG Construction Process", fill="black", font=get_font(24))

    # Bottom Up approach
    
    # Layer 1
    draw.text((50, 600), "L1: Visual Scene Graph", fill="black", font=get_font(20))
    draw_box(draw, 300, 600, 100, 50, "Obj 1", color="#eeeeee")
    draw_box(draw, 450, 600, 100, 50, "Obj 2", color="#eeeeee")
    draw.line([(350, 625), (450, 625)], fill="black", width=2)
    draw.text((390, 605), "rel", fill="black", font=get_font(14))

    # Arrow Up
    draw_arrow(draw, (425, 600), (425, 450))

    # Layer 2
    draw.text((50, 400), "L2: Knowledge Enrichment", fill="black", font=get_font(20))
    draw_box(draw, 300, 400, 250, 60, "Entity Linking (Wikidata)", color="#ccffcc")
    
    # Arrow Up
    draw_arrow(draw, (425, 400), (425, 250))

    # Layer 3
    draw.text((50, 200), "L3: Hierarchical MMKG", fill="black", font=get_font(20))
    draw_box(draw, 250, 150, 350, 120, "Final Graph:\n{Context + Facts + Relations}", color="#ccccff")

    img.save("manuscript/figures/dh_mmkg_process.png")
    print("Generated dh_mmkg_process.png")

def generate_system_arch():
    W, H = 1200, 600
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)
    
    draw.text((30, 30), "Fig 1: CMVKG-Guard System Architecture", fill="black", font=get_font(24))

    # Left to Right flow
    x = 50
    y = 250
    
    # Input
    draw_box(draw, x, y, 150, 80, "Input:\nImage + Query", color="#e0e0e0")
    draw_arrow(draw, (x+150, y+40), (x+200, y+40))
    
    # VLM Encoder
    x += 200
    draw_box(draw, x, y, 150, 80, "VLM Encoder", color="#ffe0b3")
    draw_arrow(draw, (x+150, y+40), (x+200, y+40))
    
    # Middleware Box
    x += 200
    draw.rectangle([x-20, y-100, x+420, y+200], outline="blue", width=3)
    draw.text((x, y-90), "CMVKG-Guard Middleware", fill="blue", font=get_font(18))
    
    # Module 1
    draw_box(draw, x, y-50, 180, 60, "1. DH-MMKG", color="#ccffcc")
    
    # Module 2
    draw_box(draw, x, y+50, 180, 60, "2. Verification\nEngine (CMVE)", color="#ffcccc")
    
    # Module 3
    draw_box(draw, x+220, y, 180, 80, "3. Adaptive\nCalibration & RTC", color="#ccccff")
    
    # Arrows inside middleware
    draw_arrow(draw, (x+180, y-20), (x+220, y+40))
    draw_arrow(draw, (x+180, y+80), (x+220, y+40))

    # Output Arrow
    draw_arrow(draw, (x+400, y+40), (x+450, y+40))
    
    # Final Output
    x += 450
    draw_box(draw, x, y, 150, 80, "Verified Output", color="#e0e0e0")

    img.save("manuscript/figures/system_architecture.png")
    print("Generated system_architecture.png")

if __name__ == "__main__":
    generate_cmve_flow()
    generate_dh_mmkg_process()
    generate_system_arch()
