"""
Experiment 7 (real VLM, GPU): multi-dataset + self-consistency baseline.
Run:  colab run --gpu T4 --timeout 3600 colab_exp7_multi.py

For POPE and MME (both yes/no), extract per item:
  p_yes       : LLaVA yes/no probability (single forward)  -> confidence signal
  answer,gold : greedy answer vs ground truth -> correctness (fixed across scores)
  sc_yesfrac  : self-consistency: fraction of K sampled generations answering yes
                -> the faithful ConfLVLM-style heuristic-uncertainty signal
  owl         : OWLv2 detection grounding for object-existence questions (else -1)
  category    : dataset subcategory (POPE: random/popular/adversarial; MME: task)

Dumps one RAW_JSON per dataset. All downstream conformal analysis is local/CPU.
"""
import json, sys, time, subprocess, re
import numpy as np
K = 5
NPOPE, NMME = 700, 700
def log(*a): print(*a, flush=True)
t0=time.time()
for pkg in ["datasets","bitsandbytes"]:
    try: __import__(pkg)
    except Exception: subprocess.run([sys.executable,"-m","pip","install","-q",pkg],check=True)
import torch
from datasets import load_dataset
from transformers import (AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig,
                          Owlv2Processor, Owlv2ForObjectDetection)
device="cuda" if torch.cuda.is_available() else "cpu"
log(f"[env] torch {torch.__version__} dev={torch.cuda.get_device_name(0) if device=='cuda' else 'cpu'} (+{time.time()-t0:.0f}s)")

def extract_object(q):
    ql=q.lower()
    if "is there" not in ql: return None
    s=ql.split("is there",1)[1]
    s=s.strip()
    for k in ("a ","an ","the "):
        if s.startswith(k): s=s[len(k):]; break
    for k in ["in the image","in this image","in the picture","in this picture",
              "please answer yes or no",".","?"]:
        s=s.replace(k,"")
    s=s.strip()
    return s if 1<=len(s.split())<=3 and s else None

def collect(name, N):
    ds=load_dataset(name,split="test",streaming=True); rows=[]
    for r in ds:
        rows.append(r)
        if len(rows)>=N: break
    return rows

log("[data] loading POPE + MME ...")
DATA={"POPE":collect("lmms-lab/POPE",NPOPE), "MME":collect("lmms-lab/MME",NMME)}
for k in DATA: log(f"  {k}: {len(DATA[k])} items")

# ---- LLaVA ----
M="llava-hf/llava-1.5-7b-hf"
log(f"[model] LLaVA {M} (4-bit) ...")
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

def run_llava(rows):
    p_yes=np.zeros(len(rows)); ans=np.zeros(len(rows),int); scf=np.zeros(len(rows))
    for i,r in enumerate(rows):
        q=r["question"]
        if "yes or no" not in q.lower(): q=q+"\nAnswer the question using a single word yes or no."
        prompt=f"USER: <image>\n{q} ASSISTANT:"
        inp=proc(images=r["image"].convert("RGB"),text=prompt,return_tensors="pt").to(device)
        inp["pixel_values"]=inp["pixel_values"].to(torch.float16)
        with torch.no_grad():
            lg=llava(**inp).logits[0,-1].float()
            py=torch.logsumexp(lg[YES],0); pn=torch.logsumexp(lg[NO],0)
            p_yes[i]=torch.sigmoid(py-pn).item(); ans[i]=int(p_yes[i]>=.5)
            gen=llava.generate(**inp,do_sample=True,temperature=1.0,top_p=0.9,
                    num_return_sequences=K,max_new_tokens=4,pad_token_id=tok.eos_token_id)
            texts=tok.batch_decode(gen[:,inp["input_ids"].shape[1]:],skip_special_tokens=True)
            yes=sum(1 for t in texts if "yes" in t.lower())
            scf[i]=yes/max(len(texts),1)
        if (i+1)%100==0: log(f"    llava {i+1}/{len(rows)} (+{time.time()-t0:.0f}s)")
    return p_yes,ans,scf

results={}
for name,rows in DATA.items():
    log(f"[llava] {name} ...")
    p_yes,ans,scf=run_llava(rows)
    gold=np.array([1 if str(r["answer"]).lower().startswith("yes") else 0 for r in rows])
    correct=(ans==gold).astype(int)
    results[name]=dict(p_yes=p_yes,answer=ans,gold=gold,correct=correct,sc_yesfrac=scf,
                       category=[r["category"] for r in rows],
                       obj=[extract_object(r["question"]) for r in rows])
    log(f"[llava] {name} acc={correct.mean():.4f} (+{time.time()-t0:.0f}s)")
del llava; torch.cuda.empty_cache()

# ---- OWLv2 grounding (object-existence questions) ----
log("[model] OWLv2 ...")
op=Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
od=Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(device).eval()
for name,rows in DATA.items():
    objs=results[name]["obj"]; owl=np.full(len(rows),-1.0)
    for i,(r,o) in enumerate(zip(rows,objs)):
        if not o: continue
        with torch.no_grad():
            inp=op(text=[[f"a photo of a {o}"]],images=r["image"].convert("RGB"),return_tensors="pt").to(device)
            owl[i]=float(od(**inp).logits.sigmoid().max().item())
        if (i+1)%150==0: log(f"    owl {name} {i+1}/{len(rows)} (+{time.time()-t0:.0f}s)")
    results[name]["owl"]=owl
    ncov=int((owl>=0).sum()); log(f"[owl] {name}: grounding on {ncov}/{len(rows)} items")

for name in results:
    d=results[name]
    out={k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in d.items()}
    out["dataset"]=name
    log(f"RAW_JSON::{name} "+json.dumps(out))
log(f"[done] +{time.time()-t0:.0f}s")
