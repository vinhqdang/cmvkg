"""
Experiment 14 (real, GPU): the MONOTONICITY falsification test for sequential CCRC.
Run:  colab run --gpu T4 --timeout 3000 colab_exp14_monotonicity.py 100

QUESTION (gating item from SEQUENTIAL.md): when we repair a hallucinated token mid-caption,
does the hallucination rate of the REMAINING tokens go DOWN (monotone -> the fixed-point
calibration argument is worth formalizing) or UP (a wrong/disruptive repair derails the
caption -> no fixed point; fall back to the coupling bound)?

DESIGN (paired, matched-prefix):
  1. Greedily generate a caption for an AMBER image.
  2. Using AMBER's per-image `truth` / `hallu` object lists, find the FIRST hallucinated
     object mention at position t.
  3. Build two prefixes that are identical up to t:
        prefix_orig : caption[:t] + <hallucinated word>
        prefix_rep  : caption[:t] + <repair word>      (a `truth` object, the one with the
                                                        highest OWLv2 detection confidence)
  4. Continue decoding from each prefix for the same number of new tokens.
  5. Count hallucinated object mentions in each CONTINUATION only.
Because the prefixes are matched, the difference isolates the causal effect of the repair
on downstream hallucination.

Output: per-image counts, so the paired test runs on CPU afterwards.
"""
import json, sys, os, time, subprocess, zipfile, urllib.request, glob, re
import numpy as np
N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
MAXNEW = 64
def log(*a): print(*a, flush=True)
t0 = time.time()
for pkg in ["bitsandbytes"]:
    try: __import__(pkg)
    except Exception: subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True)
import torch
from PIL import Image
from transformers import (AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig,
                          Owlv2Processor, Owlv2ForObjectDetection)
device = "cuda" if torch.cuda.is_available() else "cpu"
log(f"[env] {torch.cuda.get_device_name(0) if device=='cuda' else 'cpu'} (+{time.time()-t0:.0f}s)")

# ---- AMBER generative annotations (truth / hallu object lists per image) ----
RAW = "https://raw.githubusercontent.com/junyangwang0410/AMBER/master/data/"
annots = json.loads(urllib.request.urlopen(RAW + "annotations.json", timeout=120).read().decode())
gen = {r["id"]: r for r in annots if r.get("type") == "generative"}
log(f"[data] {len(gen)} generative annotations (+{time.time()-t0:.0f}s)")

ZIP = "/content/AMBER.zip"; IMGDIR = "/content/amber_images"
if not os.path.isdir(IMGDIR):
    url = "https://huggingface.co/datasets/stzhao/amber/resolve/main/AMBER.zip"
    log("[data] downloading AMBER.zip ...")
    _l = [0]
    def hb(b, bs, tot):
        mb = b * bs / 1e6
        if mb - _l[0] >= 100: _l[0] = mb; log(f"    ...{mb:.0f}MB (+{time.time()-t0:.0f}s)")
    urllib.request.urlretrieve(url, ZIP, reporthook=hb)
    os.makedirs(IMGDIR, exist_ok=True)
    with zipfile.ZipFile(ZIP) as z: z.extractall(IMGDIR)
index = {os.path.basename(p): p for p in glob.glob(IMGDIR + "/**/*.jpg", recursive=True)}
log(f"[data] {len(index)} images (+{time.time()-t0:.0f}s)")

ids = sorted(gen.keys())[:N]
VOCAB = set()
for r in gen.values(): VOCAB |= set(r.get("truth", [])) | set(r.get("hallu", []))
stem = lambda w: w[:-1] if len(w) > 3 and w.endswith("s") else w
VSTEM = {stem(v): v for v in VOCAB}
log(f"[data] object vocabulary {len(VOCAB)}")

M = "llava-hf/llava-1.5-7b-hf"
proc = AutoProcessor.from_pretrained(M)
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
llava = LlavaForConditionalGeneration.from_pretrained(M, quantization_config=bnb,
        torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto").eval()
tok = proc.tokenizer
op = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
od = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(device).eval()
log(f"[model] ready (+{time.time()-t0:.0f}s)")

PROMPT = "USER: <image>\nDescribe this image in detail. ASSISTANT:"
def generate(img, assistant_prefix="", max_new=MAXNEW):
    text = PROMPT + (" " + assistant_prefix if assistant_prefix else "")
    inp = proc(images=img, text=text, return_tensors="pt").to(device)
    inp["pixel_values"] = inp["pixel_values"].to(torch.float16)
    with torch.no_grad():
        out = llava.generate(**inp, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, inp["input_ids"].shape[1]:], skip_special_tokens=True)

def mentions(text, truth, hallu):
    """Return list of (word_index, word, kind) for recognised object words."""
    ws = re.findall(r"[a-zA-Z]+", text.lower())
    T, H = {stem(x) for x in truth}, {stem(x) for x in hallu}
    out = []
    for i, w in enumerate(ws):
        s = stem(w)
        if s in H: out.append((i, w, "hallu"))
        elif s in T: out.append((i, w, "truth"))
    return out, ws

results = []
for c, iid in enumerate(ids):
    r = gen[iid]; fn = f"AMBER_{iid}.jpg"
    if fn not in index: continue
    img = Image.open(index[fn]).convert("RGB")
    truth, hallu = r.get("truth", []), r.get("hallu", [])
    cap = generate(img)
    ms, ws = mentions(cap, truth, hallu)
    first_h = next((m for m in ms if m[2] == "hallu"), None)
    if first_h is None: continue                       # no hallucination to repair
    idx_w, bad_word, _ = first_h
    # choose the repair: the truth object with the strongest detection evidence
    best, best_s = None, -1
    for o in truth[:8]:
        with torch.no_grad():
            oi = op(text=[[f"a photo of a {o}"]], images=img, return_tensors="pt").to(device)
            sc = float(od(**oi).logits.sigmoid().max().item())
        if sc > best_s: best, best_s = o, sc
    if best is None: continue
    # matched prefixes (identical up to the intervention point)
    pre = " ".join(ws[:idx_w])
    prefix_orig = (pre + " " + bad_word).strip()
    prefix_rep  = (pre + " " + best).strip()
    cont_o = generate(img, prefix_orig, MAXNEW // 2)
    cont_r = generate(img, prefix_rep,  MAXNEW // 2)
    mo, wo = mentions(cont_o, truth, hallu)
    mr, wr = mentions(cont_r, truth, hallu)
    results.append(dict(id=iid, bad=bad_word, repair=best, det=best_s, pos=idx_w,
                        n_hallu_orig=sum(1 for m in mo if m[2] == "hallu"),
                        n_hallu_rep=sum(1 for m in mr if m[2] == "hallu"),
                        n_truth_orig=sum(1 for m in mo if m[2] == "truth"),
                        n_truth_rep=sum(1 for m in mr if m[2] == "truth"),
                        len_orig=len(wo), len_rep=len(wr)))
    if len(results) % 10 == 0:
        log(f"  {len(results)} paired cases from {c+1} images (+{time.time()-t0:.0f}s)")
    if time.time() - t0 > 2400: log("[budget] stopping early"); break

log(f"[done] {len(results)} paired cases")
log("RAW_JSON " + json.dumps(results))
log(f"[done] +{time.time()-t0:.0f}s")
