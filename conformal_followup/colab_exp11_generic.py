"""
Generic yes/no dataset extractor for the conformal comparison.
Run:  colab run --gpu T4 --timeout 2400 colab_exp11_generic.py <dataset> <N>
  <dataset> in: hallusion | gqa | mme_exist
Extracts LLaVA-1.5-7B (4-bit) p_yes + OWLv2 grounding (object-existence Qs) +
correctness per item. Fast (one forward + optional detection). Dumps RAW_JSON.
"""
import json, sys, time, subprocess
import numpy as np
KEY = sys.argv[1] if len(sys.argv)>1 else "hallusion"
N   = int(sys.argv[2]) if len(sys.argv)>2 else 600
def log(*a): print(*a,flush=True)
t0=time.time()
for pkg in ["datasets","bitsandbytes"]:
    try: __import__(pkg)
    except Exception: subprocess.run([sys.executable,"-m","pip","install","-q",pkg],check=True)
import torch
from datasets import load_dataset
from transformers import (AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig,
                          Owlv2Processor, Owlv2ForObjectDetection)
device="cuda" if torch.cuda.is_available() else "cpu"
log(f"[env] {torch.cuda.get_device_name(0) if device=='cuda' else 'cpu'} (+{time.time()-t0:.0f}s)")

def extract_object(q):
    ql=q.lower()
    if "is there" not in ql: return None
    s=ql.split("is there",1)[1].strip()
    for k in ("a ","an ","the "):
        if s.startswith(k): s=s[len(k):]; break
    for k in ["in the image","in this image","in the picture","please answer yes or no",".","?"]:
        s=s.replace(k,"")
    s=s.strip(); return s if 1<=len(s.split())<=3 and s else None

# ---- dataset loaders normalized to (image, question, gold∈{0,1}, category) ----
def load_items(key,N):
    items=[]
    if key=="hallusion":
        ds=load_dataset("lmms-lab/HallusionBench",split="image",streaming=True)
        for r in ds:
            if int(r.get("visual_input",1))!=1: continue
            g=str(r["gt_answer"]).strip()
            if g not in ("0","1"): continue
            items.append((r["image"],r["question"],int(g),r.get("category","")))
            if len(items)>=N: break
    elif key=="gqa":
        ds=load_dataset("lmms-lab/GQA","testdev_balanced_instructions",split="testdev",streaming=True)
        for r in ds:
            a=str(r["answer"]).lower().strip()
            if a not in ("yes","no"): continue
            items.append((r["image"],r["question"],1 if a=="yes" else 0,"gqa"))
            if len(items)>=N: break
    elif key=="mme_exist":
        ds=load_dataset("lmms-lab/MME",split="test",streaming=True)
        for r in ds:
            if r.get("category","")!="existence": continue
            a=str(r["answer"]).lower()
            items.append((r["image"],r["question"],1 if a.startswith("yes") else 0,"existence"))
            if len(items)>=N: break
    else: raise ValueError(key)
    return items

items=load_items(KEY,N)
log(f"[data] {KEY}: {len(items)} yes/no items (+{time.time()-t0:.0f}s)")

M="llava-hf/llava-1.5-7b-hf"
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
log(f"[model] ready (+{time.time()-t0:.0f}s)")
n=len(items);p_yes=np.zeros(n);ans=np.zeros(n,int);owl=np.full(n,-1.0)
gold=np.array([it[2] for it in items]);objs=[extract_object(it[1]) for it in items]
for i,(img,q,g,cat) in enumerate(items):
    im=img.convert("RGB")
    if "yes or no" not in q.lower(): q=q+" Please answer yes or no."
    inp=proc(images=im,text=f"USER: <image>\n{q} ASSISTANT:",return_tensors="pt").to(device)
    inp["pixel_values"]=inp["pixel_values"].to(torch.float16)
    with torch.no_grad():
        lg=llava(**inp).logits[0,-1].float()
        p_yes[i]=torch.sigmoid(torch.logsumexp(lg[YES],0)-torch.logsumexp(lg[NO],0)).item()
        ans[i]=int(p_yes[i]>=.5)
    if (i+1)%100==0: log(f"  {KEY} {i+1}/{n} (+{time.time()-t0:.0f}s)")
del llava; torch.cuda.empty_cache()
op=Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
od=Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(device).eval()
for i,(img,q,g,cat) in enumerate(items):
    if not objs[i]: continue
    with torch.no_grad():
        oi=op(text=[[f"a photo of a {objs[i]}"]],images=img.convert("RGB"),return_tensors="pt").to(device)
        owl[i]=float(od(**oi).logits.sigmoid().max().item())
correct=(ans==gold).astype(int)
out=dict(dataset=KEY,p_yes=p_yes.tolist(),answer=ans.tolist(),gold=gold.tolist(),
         correct=correct.tolist(),owl=owl.tolist(),obj=objs,
         category=[it[3] for it in items])
log(f"[{KEY}] acc={correct.mean():.4f} grounded={(owl>=0).sum()}/{n}")
log("RAW_JSON "+json.dumps(out)); log(f"[done] +{time.time()-t0:.0f}s")
