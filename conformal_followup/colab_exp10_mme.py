"""
Experiment 10 (real, GPU): MME as a second dataset (fast, no self-consistency).
Run:  colab run --gpu T4 --timeout 2000 colab_exp10_mme.py 700
LLaVA-1.5-7B (4-bit) confidence + OWLv2 grounding (where an object is extractable)
+ correctness on MME. Tests whether the conformal selective framework + validity
generalize to a different dataset. Dumps RAW_JSON.
"""
import json, sys, time, subprocess
import numpy as np
N=int(sys.argv[1]) if len(sys.argv)>1 else 700
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
ds=load_dataset("lmms-lab/MME",split="test",streaming=True); rows=[]
for r in ds:
    rows.append(r)
    if len(rows)>=N: break
log(f"[data] MME {len(rows)} (+{time.time()-t0:.0f}s)")
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
p_yes=np.zeros(N);ans=np.zeros(N,int);owl=np.full(N,-1.0)
objs=[extract_object(r["question"]) for r in rows]
for i,r in enumerate(rows):
    img=r["image"].convert("RGB");q=r["question"]
    if "yes or no" not in q.lower(): q+=" Please answer yes or no."
    inp=proc(images=img,text=f"USER: <image>\n{q} ASSISTANT:",return_tensors="pt").to(device)
    inp["pixel_values"]=inp["pixel_values"].to(torch.float16)
    with torch.no_grad():
        lg=llava(**inp).logits[0,-1].float()
        p_yes[i]=torch.sigmoid(torch.logsumexp(lg[YES],0)-torch.logsumexp(lg[NO],0)).item()
        ans[i]=int(p_yes[i]>=.5)
    if (i+1)%100==0: log(f"  mme {i+1}/{N} (+{time.time()-t0:.0f}s)")
del llava; torch.cuda.empty_cache()
op=Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
od=Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(device).eval()
for i,r in enumerate(rows):
    if not objs[i]: continue
    with torch.no_grad():
        oi=op(text=[[f"a photo of a {objs[i]}"]],images=r["image"].convert("RGB"),return_tensors="pt").to(device)
        owl[i]=float(od(**oi).logits.sigmoid().max().item())
gold=np.array([1 if str(r["answer"]).lower().startswith("yes") else 0 for r in rows])
correct=(ans==gold).astype(int)
out=dict(dataset="MME",p_yes=p_yes.tolist(),answer=ans.tolist(),gold=gold.tolist(),
         correct=correct.tolist(),owl=owl.tolist(),obj=objs,category=[r["category"] for r in rows])
log(f"[mme] acc={correct.mean():.4f} grounded={(owl>=0).sum()}/{N}")
log("RAW_JSON "+json.dumps(out)); log(f"[done] +{time.time()-t0:.0f}s")
