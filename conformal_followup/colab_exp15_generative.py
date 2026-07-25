"""
Experiment 15 (real, GPU): claim-level GENERATIVE substrate for CCRC / CTS.
Run:  colab run --gpu T4 --timeout 3000 colab_exp15_generative.py 200

Moves from yes/no VQA to free-form captioning -- the setting ConfLVLM and BCEA operate
in, and the setting where certified CORRECTION is meaningful (replace a hallucinated
object word rather than delete the sentence).

Per AMBER generative image we extract, for every object CLAIM in the caption:
  claim         : the mentioned object word
  label         : 1 if the object is in the image's `truth` list else 0   (CHAIR-style)
  s_vlm         : mean token logprob of the claim's word(s) under teacher forcing
                  -> the model's own confidence in that claim   (channel 1)
  ev_claim      : OWLv2 detection confidence for the claimed object        (channel 2)
  rep_obj       : best alternative object (a `truth` item) by OWLv2 score  (repair)
  ev_rep        : that alternative's OWLv2 confidence
  rep_label     : 1 (repairs substitute a truth object, so they are faithful by
                  construction ONLY if ev_rep is trustworthy -- we store the raw
                  detection score so the analysis can certify it, not assume it)
  pos           : claim's word index (needed for sequence/trajectory analysis)
  img_id        : groups claims belonging to the same caption

This supports, on CPU afterwards:
  - claim-level FILTER (drop) vs CCRC (repair) risk-coverage
  - the CTS study: sequential/trajectory selection over per-claim repair choices
"""
import json, sys, os, time, subprocess, zipfile, urllib.request, glob, re
import numpy as np
N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
MAXNEW = 96
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

RAW = "https://raw.githubusercontent.com/junyangwang0410/AMBER/master/data/"
annots = json.loads(urllib.request.urlopen(RAW + "annotations.json", timeout=120).read().decode())
gen = {r["id"]: r for r in annots if r.get("type") == "generative"}
ZIP = "/content/AMBER.zip"; IMGDIR = "/content/amber_images"
if not os.path.isdir(IMGDIR):
    log("[data] downloading AMBER.zip ...")
    _l = [0]
    def hb(b, bs, tot):
        mb = b * bs / 1e6
        if mb - _l[0] >= 100: _l[0] = mb; log(f"    ...{mb:.0f}MB (+{time.time()-t0:.0f}s)")
    urllib.request.urlretrieve(
        "https://huggingface.co/datasets/stzhao/amber/resolve/main/AMBER.zip", ZIP, reporthook=hb)
    os.makedirs(IMGDIR, exist_ok=True)
    with zipfile.ZipFile(ZIP) as z: z.extractall(IMGDIR)
index = {os.path.basename(p): p for p in glob.glob(IMGDIR + "/**/*.jpg", recursive=True)}
log(f"[data] {len(gen)} annots, {len(index)} images (+{time.time()-t0:.0f}s)")

VOCAB = set()
for r in gen.values(): VOCAB |= set(r.get("truth", [])) | set(r.get("hallu", []))
stem = lambda w: w[:-1] if len(w) > 3 and w.endswith("s") else w
VSTEM = {stem(v) for v in VOCAB}
ids = sorted(gen.keys())[:N]

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

def caption_with_logprobs(img):
    """Greedy caption + per-token logprob via one teacher-forced pass."""
    inp = proc(images=img, text=PROMPT, return_tensors="pt").to(device)
    inp["pixel_values"] = inp["pixel_values"].to(torch.float16)
    with torch.no_grad():
        out = llava.generate(**inp, max_new_tokens=MAXNEW, do_sample=False,
                             pad_token_id=tok.eos_token_id, output_scores=True,
                             return_dict_in_generate=True)
    seq = out.sequences[0, inp["input_ids"].shape[1]:]
    lps = []
    for i, sc in enumerate(out.scores):
        if i >= len(seq): break
        lps.append(float(torch.log_softmax(sc[0].float(), -1)[seq[i]]))
    toks = tok.convert_ids_to_tokens(seq.tolist())   # keeps the "\u2581" word marker
    return tok.decode(seq, skip_special_tokens=True), toks, lps

def owl_score(img, obj, cache):
    if obj in cache: return cache[obj]
    with torch.no_grad():
        oi = op(text=[[f"a photo of a {obj}"]], images=img, return_tensors="pt").to(device)
        s = float(od(**oi).logits.sigmoid().max().item())
    cache[obj] = s; return s

rows = []
for c, iid in enumerate(ids):
    fn = f"AMBER_{iid}.jpg"
    if fn not in index: continue
    img = Image.open(index[fn]).convert("RGB")
    truth = gen[iid].get("truth", [])
    cap, toks, lps = caption_with_logprobs(img)
    cache = {}
    # map word -> mean logprob of the tokens that spell it
    words, wlp, buf, blp = [], [], "", []
    SP = "\u2581"                                   # SentencePiece word boundary
    for t, lp in zip(toks, lps):
        if t.startswith(SP) and buf:
            words.append(buf); wlp.append(float(np.mean(blp))); buf, blp = "", []
        buf += t.replace(SP, ""); blp.append(lp)
    if buf: words.append(buf); wlp.append(float(np.mean(blp)))
    Tst = {stem(x) for x in truth}
    # best repair candidate for this image (strongest detected truth object)
    reps = [(owl_score(img, o, cache), o) for o in truth[:5]]
    ev_rep, rep_obj = max(reps) if reps else (-1.0, None)
    nclaim = 0
    for wi, (w, lp) in enumerate(zip(words, wlp)):
        st = stem(re.sub(r"[^a-z]", "", w.lower()))
        if st not in VSTEM: continue
        rows.append(dict(img_id=iid, pos=wi, claim=st, label=int(st in Tst),
                         s_vlm=lp, ev_claim=owl_score(img, st, cache),
                         rep_obj=rep_obj, ev_rep=ev_rep,
                         n_words=len(words)))
        nclaim += 1
    if (c + 1) % 20 == 0:
        log(f"  {c+1}/{len(ids)} images, {len(rows)} claims (+{time.time()-t0:.0f}s)")
    if time.time() - t0 > 2500: log("[budget] stop"); break

lab = np.array([r["label"] for r in rows])
log(f"[gen] {len(rows)} claims from {len(set(r['img_id'] for r in rows))} captions; "
    f"hallucination rate {1-lab.mean():.3f}")
log("RAW_JSON " + json.dumps(rows)); log(f"[done] +{time.time()-t0:.0f}s")
