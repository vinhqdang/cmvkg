"""
Experiment 5 (real, GPU): STRONGER grounding score via open-vocabulary detection.
Run:  colab run --gpu T4 --timeout 1800 colab_exp5_owlv2.py 1500

Rationale: exp3 used weak full-image CLIP similarity as the grounding signal.
Here we use OWLv2 open-vocabulary DETECTION (closer to CMVKG-Guard's L1): the
max detection confidence that the queried object is actually present. This is a
much stronger structured grounding signal.

Only the detector runs here (LLaVA confidences are already extracted in
raw_scores.json). We stream the SAME 1500 POPE probes in the SAME order and also
dump `gold` so alignment with raw_scores.json can be verified before fusing.
"""
import json, sys, time, subprocess
import numpy as np

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
def log(*a): print(*a, flush=True)
t0 = time.time()
for pkg in ["datasets"]:
    try: __import__(pkg)
    except Exception: subprocess.run([sys.executable,"-m","pip","install","-q",pkg], check=True)

import torch
from datasets import load_dataset
from transformers import Owlv2Processor, Owlv2ForObjectDetection
device = "cuda" if torch.cuda.is_available() else "cpu"
log(f"[env] torch {torch.__version__} dev={torch.cuda.get_device_name(0) if device=='cuda' else 'cpu'} (+{time.time()-t0:.0f}s)")

log("[data] streaming lmms-lab/POPE (same order as extraction) ...")
ds = load_dataset("lmms-lab/POPE", split="test", streaming=True)
rows = []
for r in ds:
    rows.append(r)
    if len(rows) >= N: break
def obj_of(q):
    q=q.lower()
    for k in ["is there a","is there an","in the image?","in the picture?","?"]: q=q.replace(k,"")
    return q.strip()
objs = [obj_of(r["question"]) for r in rows]
gold = [1 if str(r["answer"]).lower().startswith("yes") else 0 for r in rows]
log(f"[data] {len(rows)} probes (+{time.time()-t0:.0f}s)")

MODEL = "google/owlv2-base-patch16-ensemble"
log(f"[model] loading {MODEL} ...")
proc = Owlv2Processor.from_pretrained(MODEL)
det = Owlv2ForObjectDetection.from_pretrained(MODEL).to(device).eval()
log(f"[model] ready (+{time.time()-t0:.0f}s)")

ground_det = np.zeros(len(rows))
for i,(r,obj) in enumerate(zip(rows, objs)):
    img = r["image"].convert("RGB")
    with torch.no_grad():
        inp = proc(text=[[f"a photo of a {obj}"]], images=img, return_tensors="pt").to(device)
        out = det(**inp)
        ground_det[i] = float(out.logits.sigmoid().max().item())  # best detection conf
    if (i+1)%100==0: log(f"  [owlv2] {i+1}/{len(rows)} (+{time.time()-t0:.0f}s)")

raw = {"ground_det":[round(float(x),5) for x in ground_det],
       "gold":[int(x) for x in gold],
       "obj":objs}
log("RAW_JSON "+json.dumps(raw))
log(f"[done] +{time.time()-t0:.0f}s")
