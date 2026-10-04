"""
Experiment 16 (real, GPU): extend POPE to the remaining adversarial probes (rows START..END of the same
stream used by colab_exp3.py), doubling the number of distinct images from 250 to 500.
Run:  colab run --gpu T4 --timeout 7200 colab_exp16_pope_ext.py 1500 3000

Same prompt, same yes/no logit read-out and same OWLv2 query as colab_exp3.py / colab_exp5_owlv2.py, so
the new rows are exchangeable with the cached ones. Prints a RAW_JSON line every CHUNK items so a timeout
cannot lose finished work. Also records the COCO image id (`image_source`) of every probe.
"""
import json, sys, time, subprocess
import numpy as np
START = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
END   = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
CHUNK = 250
def log(*a): print(*a, flush=True)
t0 = time.time()
for pkg in ["datasets", "bitsandbytes"]:
    try: __import__(pkg)
    except Exception: subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True)
import torch
from datasets import load_dataset
from transformers import (AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig,
                          Owlv2Processor, Owlv2ForObjectDetection)
device = "cuda" if torch.cuda.is_available() else "cpu"
log(f"[env] torch {torch.__version__} dev={torch.cuda.get_device_name(0) if device=='cuda' else 'cpu'}")

def obj_of(q):
    q = q.lower()
    for k in ["is there a", "is there an", "in the image?", "in the picture?", "?"]: q = q.replace(k, "")
    return q.strip()

MODEL = "llava-hf/llava-1.5-7b-hf"
proc = AutoProcessor.from_pretrained(MODEL)
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
llava = LlavaForConditionalGeneration.from_pretrained(MODEL, quantization_config=bnb, torch_dtype=torch.float16,
                                                      low_cpu_mem_usage=True, device_map="auto").eval()
tok = proc.tokenizer
def ids_for(words):
    out = set()
    for w in words:
        for v in (w, " " + w):
            t = tok(v, add_special_tokens=False).input_ids
            if len(t) == 1: out.add(t[0])
    return list(out)
YES, NO = ids_for(["yes", "Yes", "YES"]), ids_for(["no", "No", "NO"])
OM = "google/owlv2-base-patch16-ensemble"
oproc = Owlv2Processor.from_pretrained(OM)
odet = Owlv2ForObjectDetection.from_pretrained(OM).to(device).eval()
log(f"[model] ready (+{time.time()-t0:.0f}s)")

ds = load_dataset("lmms-lab/POPE", split="test", streaming=True)
rec = dict(row=[], image_source=[], question=[], gold=[], p_yes=[], answer=[], ground_det=[])
for i, r in enumerate(ds):
    if i < START: continue
    if i >= END: break
    img = r["image"].convert("RGB"); q = r["question"]
    prompt = f"USER: <image>\n{q}\nAnswer the question using a single word yes or no. ASSISTANT:"
    inp = proc(images=img, text=prompt, return_tensors="pt").to(device)
    inp["pixel_values"] = inp["pixel_values"].to(torch.float16)
    with torch.no_grad():
        lg = llava(**inp).logits[0, -1].float()
        p = torch.sigmoid(torch.logsumexp(lg[YES], 0) - torch.logsumexp(lg[NO], 0)).item()
        oi = oproc(text=[[f"a photo of a {obj_of(q)}"]], images=img, return_tensors="pt").to(device)
        o = float(odet(**oi).logits.sigmoid().max().item())
    rec["row"].append(i); rec["image_source"].append(r.get("image_source", "")); rec["question"].append(q)
    rec["gold"].append(1 if str(r["answer"]).lower().startswith("yes") else 0)
    rec["p_yes"].append(round(p, 6)); rec["answer"].append(int(p >= 0.5)); rec["ground_det"].append(round(o, 5))
    n = len(rec["row"])
    if n % 100 == 0: log(f"  {n}/{END-START} (+{time.time()-t0:.0f}s)")
    if n % CHUNK == 0 or i == END - 1: log("RAW_JSON " + json.dumps(rec))
log("RAW_JSON " + json.dumps(rec)); log(f"[done] +{time.time()-t0:.0f}s")
