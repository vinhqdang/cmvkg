"""
Experiment 13 (real, GPU): BCEA vs CCRC vs composition.
Run:  colab run --gpu T4 --timeout 3000 colab_exp13_bcea.py 400

Implements BCEA (arXiv 2606.16667, "Look Again Before You Abstain"): when a claim is
borderline, ACQUIRE extra visual evidence (zoomed views) and RE-SCORE the model's
original assertion. Their post-acquisition score:

    s_acq(x,c) = max( l(x), max_b l(x^(b)) ) - l(empty)

where l is the assertion's logit margin and l(empty) is the image-free (blind)
baseline. BCEA never changes the answer; the guarantee comes from calibrating the
threshold on post-acquisition scores (their Theorem 1).

We extract, per POPE item:
  l_full     : logit margin for the model's asserted direction, full image
  l_crops[B] : same, on B zoomed crops           -> BCEA acquisition
  l_blind    : same, on a blank image            -> l(empty)
  p_yes      : full-image P(yes)                 -> our channel 1
  owl        : OWLv2 detection confidence        -> our channel 2 (repair)
  gold, answer, correct

This supports three arms downstream (all CPU, same FST+CP protocol):
  BCEA        : filter on s_acq                       (their mechanism: re-score)
  CCRC        : filter on ch.1 + repair via ch.2      (ours: replace)
  COMPOSED    : s_acq as ch.1 + repair via ch.2       (both)
"""
import json, sys, time, subprocess
import numpy as np
N = int(sys.argv[1]) if len(sys.argv) > 1 else 400
B = 3                                   # zoomed views per image
def log(*a): print(*a, flush=True)
t0 = time.time()
for pkg in ["datasets", "bitsandbytes"]:
    try: __import__(pkg)
    except Exception: subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True)
import torch
from PIL import Image
from datasets import load_dataset
from transformers import (AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig,
                          Owlv2Processor, Owlv2ForObjectDetection)
device = "cuda" if torch.cuda.is_available() else "cpu"
log(f"[env] {torch.cuda.get_device_name(0) if device=='cuda' else 'cpu'} (+{time.time()-t0:.0f}s)")

ds = load_dataset("lmms-lab/POPE", split="test", streaming=True)
rows = []
for r in ds:
    rows.append(r)
    if len(rows) >= N: break
log(f"[data] POPE {len(rows)} (+{time.time()-t0:.0f}s)")

def extract_object(q):
    ql = q.lower()
    if "is there" not in ql: return None
    s = ql.split("is there", 1)[1].strip()
    for k in ("a ", "an ", "the "):
        if s.startswith(k): s = s[len(k):]; break
    for k in ["in the image", "in this image", "in the picture", "please answer yes or no", ".", "?"]:
        s = s.replace(k, "")
    s = s.strip(); return s if 1 <= len(s.split()) <= 3 and s else None

M = "llava-hf/llava-1.5-7b-hf"
log("[model] loading LLaVA (4-bit) ...")
proc = AutoProcessor.from_pretrained(M)
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
llava = LlavaForConditionalGeneration.from_pretrained(M, quantization_config=bnb,
        torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto").eval()
tok = proc.tokenizer
def ids_for(ws):
    o = set()
    for w in ws:
        for v in (w, " " + w):
            t = tok(v, add_special_tokens=False).input_ids
            if len(t) == 1: o.add(t[0])
    return list(o)
YES, NO = ids_for(["yes", "Yes", "YES"]), ids_for(["no", "No", "NO"])
log(f"[model] ready (+{time.time()-t0:.0f}s)")

def yes_no_logits(img, q):
    inp = proc(images=img, text=f"USER: <image>\n{q} Please answer yes or no. ASSISTANT:",
               return_tensors="pt").to(device)
    inp["pixel_values"] = inp["pixel_values"].to(torch.float16)
    with torch.no_grad():
        lg = llava(**inp).logits[0, -1].float()
    return float(torch.logsumexp(lg[YES], 0)), float(torch.logsumexp(lg[NO], 0))

def zoom_views(img, b=B):
    """b zoomed crops: center + two halves (cheap, deterministic)."""
    w, h = img.size
    out = [img.crop((w // 6, h // 6, 5 * w // 6, 5 * h // 6))]          # center 67%
    if b >= 2: out.append(img.crop((0, 0, w // 2 + w // 8, h)))          # left
    if b >= 3: out.append(img.crop((w // 2 - w // 8, 0, w, h)))          # right
    return out[:b]

n = len(rows)
p_yes = np.zeros(n); ans = np.zeros(n, int)
l_full = np.zeros(n); l_blind = np.zeros(n); l_crops = np.zeros((n, B))
gold = np.array([1 if str(r["answer"]).lower().startswith("yes") else 0 for r in rows])
objs = [extract_object(r["question"]) for r in rows]
blank = Image.new("RGB", (336, 336), (127, 127, 127))
for i, r in enumerate(rows):
    img = r["image"].convert("RGB"); q = r["question"]
    ly, ln = yes_no_logits(img, q)
    p_yes[i] = 1 / (1 + np.exp(-(ly - ln))); ans[i] = int(p_yes[i] >= .5)
    sgn = 1.0 if ans[i] == 1 else -1.0                 # margin for the ASSERTED direction
    l_full[i] = sgn * (ly - ln)
    by, bn = yes_no_logits(blank, q); l_blind[i] = sgn * (by - bn)
    for j, v in enumerate(zoom_views(img)):
        vy, vn = yes_no_logits(v, q); l_crops[i, j] = sgn * (vy - vn)
    if (i + 1) % 25 == 0: log(f"  bcea {i+1}/{n} (+{time.time()-t0:.0f}s)")
del llava; torch.cuda.empty_cache()

log("[model] OWLv2 ...")
op = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
od = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(device).eval()
owl = np.full(n, -1.0)
for i, r in enumerate(rows):
    if not objs[i]: continue
    with torch.no_grad():
        oi = op(text=[[f"a photo of a {objs[i]}"]], images=r["image"].convert("RGB"),
                return_tensors="pt").to(device)
        owl[i] = float(od(**oi).logits.sigmoid().max().item())
    if (i + 1) % 100 == 0: log(f"  owl {i+1}/{n} (+{time.time()-t0:.0f}s)")

correct = (ans == gold).astype(int)
s_acq = np.maximum(l_full, l_crops.max(1)) - l_blind          # BCEA post-acquisition score
out = dict(dataset="POPE-bcea", p_yes=p_yes.tolist(), answer=ans.tolist(), gold=gold.tolist(),
           correct=correct.tolist(), owl=owl.tolist(), obj=objs,
           l_full=l_full.tolist(), l_blind=l_blind.tolist(),
           l_crops=l_crops.tolist(), s_acq=s_acq.tolist(),
           category=[r["category"] for r in rows])
log(f"[bcea] acc={correct.mean():.4f} grounded={(owl>=0).sum()}/{n} "
    f"s_acq range {s_acq.min():.2f}..{s_acq.max():.2f}")
log("RAW_JSON " + json.dumps(out)); log(f"[done] +{time.time()-t0:.0f}s")
