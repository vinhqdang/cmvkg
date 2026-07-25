"""
Experiment 12 (real, GPU): AMBER discriminative benchmark.
Run:  colab run --gpu T4 --timeout 3000 colab_exp12_amber.py existence 700
      subsets: existence | attribute | relation

AMBER (Wang et al.) is the standard LLM-free hallucination benchmark used by
REVERSE and Attention Lens. Images come from the public HF mirror
`stzhao/amber` (AMBER.zip, ~415MB); queries/annotations from the official GitHub.

Subsets (annotation `type`):
  existence -> discriminative-hallucination (4924 "Is there a X in this image?")
  attribute -> discriminative-attribute-* (state/number/action)
  relation  -> discriminative-relation
Extracts LLaVA-1.5-7B (4-bit) p_yes + OWLv2 grounding + correctness. Dumps RAW_JSON.
"""
import json, sys, os, time, subprocess, zipfile, urllib.request, glob
import numpy as np
SUBSET = sys.argv[1] if len(sys.argv)>1 else "existence"
N      = int(sys.argv[2]) if len(sys.argv)>2 else 700
def log(*a): print(*a,flush=True)
t0=time.time()
for pkg in ["bitsandbytes"]:
    try: __import__(pkg)
    except Exception: subprocess.run([sys.executable,"-m","pip","install","-q",pkg],check=True)
import torch
from PIL import Image
from transformers import (AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig,
                          Owlv2Processor, Owlv2ForObjectDetection)
device="cuda" if torch.cuda.is_available() else "cpu"
log(f"[env] {torch.cuda.get_device_name(0) if device=='cuda' else 'cpu'} (+{time.time()-t0:.0f}s)")

# ---- 1. metadata from official repo ----
RAW="https://raw.githubusercontent.com/junyangwang0410/AMBER/master/data/"
def fetch_json(url):
    return json.loads(urllib.request.urlopen(url,timeout=120).read().decode())
log("[data] fetching AMBER queries + annotations ...")
queries=fetch_json(RAW+"query/query_discriminative.json")
annots =fetch_json(RAW+"annotations.json")
truth={r["id"]:r for r in annots}
ATTR=("discriminative-attribute-state","discriminative-attribute-number",
      "discriminative-attribute-action")
WANT={"existence":("discriminative-hallucination",),
      "attribute":ATTR,
      "relation":("discriminative-relation",),
      # standard AMBER(d): all discriminative types combined -> mixed yes/no labels.
      # NOTE: single-type subsets are label-degenerate (hallucination is all "no",
      # relation all "yes"), which makes correctness a deterministic function of the
      # model's own answer. Use "all" for selective-prediction experiments.
      "all":("discriminative-hallucination",)+ATTR+("discriminative-relation",)}[SUBSET]
cand=[]
for q in queries:
    a=truth.get(q["id"])
    if not a or a.get("type") not in WANT: continue
    t=str(a.get("truth","")).lower().strip()
    if t not in ("yes","no"): continue
    cand.append((q["id"],q["image"],q["query"],1 if t=="yes" else 0,a["type"]))
if SUBSET=="all":                      # shuffle so all types/labels are represented
    import random; random.Random(0).shuffle(cand)
picks=cand[:N]
log(f"[data] {SUBSET}: {len(picks)} items (+{time.time()-t0:.0f}s)")

# ---- 2. images from public HF mirror ----
ZIP="/content/AMBER.zip"; IMGDIR="/content/amber_images"
if not os.path.isdir(IMGDIR):
    url="https://huggingface.co/datasets/stzhao/amber/resolve/main/AMBER.zip"
    log("[data] downloading AMBER.zip (~415MB) ...")
    _last=[0]
    def _hb(blocks,bs,total):
        mb=blocks*bs/1e6
        if mb-_last[0]>=50: _last[0]=mb; log(f"    ...{mb:.0f}MB (+{time.time()-t0:.0f}s)")
    urllib.request.urlretrieve(url,ZIP,reporthook=_hb)
    log(f"[data] unzipping (+{time.time()-t0:.0f}s)")
    os.makedirs(IMGDIR,exist_ok=True)
    with zipfile.ZipFile(ZIP) as z: z.extractall(IMGDIR)
index={os.path.basename(p):p for p in glob.glob(IMGDIR+"/**/*.jpg",recursive=True)}
log(f"[data] {len(index)} images indexed (+{time.time()-t0:.0f}s)")
picks=[p for p in picks if p[1] in index]
log(f"[data] {len(picks)} items with images")

def extract_object(q):
    ql=q.lower()
    if "is there" not in ql: return None
    s=ql.split("is there",1)[1].strip()
    for k in ("a ","an ","the "):
        if s.startswith(k): s=s[len(k):]; break
    for k in ["in this image","in the image","in this picture","in the picture",".","?"]:
        s=s.replace(k,"")
    s=s.strip(); return s if 1<=len(s.split())<=3 and s else None

# ---- 3. LLaVA ----
M="llava-hf/llava-1.5-7b-hf"
log("[model] fetching LLaVA weights (may take ~5min, silent) ...")
proc=AutoProcessor.from_pretrained(M)
bnb=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_compute_dtype=torch.float16)
llava=LlavaForConditionalGeneration.from_pretrained(M,quantization_config=bnb,
      torch_dtype=torch.float16,low_cpu_mem_usage=True,device_map="auto").eval()
tok=proc.tokenizer
def ids_for(ws):
    o=set()
    for w in ws:
        for v in (w," "+w):
            t=tok(v,add_special_tokens=False).input_ids
            if len(t)==1: o.add(t[0])
    return list(o)
YES,NO=ids_for(["yes","Yes","YES"]),ids_for(["no","No","NO"])
log(f"[model] LLaVA ready (+{time.time()-t0:.0f}s)")
n=len(picks);p_yes=np.zeros(n);ans=np.zeros(n,int);owl=np.full(n,-1.0)
gold=np.array([p[3] for p in picks]);objs=[extract_object(p[2]) for p in picks]
for i,(qid,fn,q,g,typ) in enumerate(picks):
    im=Image.open(index[fn]).convert("RGB")
    inp=proc(images=im,text=f"USER: <image>\n{q} Please answer yes or no. ASSISTANT:",
             return_tensors="pt").to(device)
    inp["pixel_values"]=inp["pixel_values"].to(torch.float16)
    with torch.no_grad():
        lg=llava(**inp).logits[0,-1].float()
        p_yes[i]=torch.sigmoid(torch.logsumexp(lg[YES],0)-torch.logsumexp(lg[NO],0)).item()
        ans[i]=int(p_yes[i]>=.5)
    if (i+1)%50==0: log(f"  amber-{SUBSET} {i+1}/{n} (+{time.time()-t0:.0f}s)")
del llava; torch.cuda.empty_cache()

# ---- 4. OWLv2 grounding ----
op=Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
od=Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(device).eval()
for i,(qid,fn,q,g,typ) in enumerate(picks):
    if not objs[i]: continue
    with torch.no_grad():
        oi=op(text=[[f"a photo of a {objs[i]}"]],images=Image.open(index[fn]).convert("RGB"),
              return_tensors="pt").to(device)
        owl[i]=float(od(**oi).logits.sigmoid().max().item())
    if (i+1)%100==0: log(f"  owl {i+1}/{n} (+{time.time()-t0:.0f}s)")
correct=(ans==gold).astype(int)
out=dict(dataset=f"AMBER-{SUBSET}",p_yes=p_yes.tolist(),answer=ans.tolist(),gold=gold.tolist(),
         correct=correct.tolist(),owl=owl.tolist(),obj=objs,category=[p[4] for p in picks])
log(f"[amber-{SUBSET}] acc={correct.mean():.4f} grounded={(owl>=0).sum()}/{n}")
log("RAW_JSON "+json.dumps(out)); log(f"[done] +{time.time()-t0:.0f}s")
