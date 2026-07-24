"""
Experiment 7 (real VLM, GPU): multi-dataset + self-consistency baseline. ROBUST.
Run:  colab run --gpu T4 --timeout 3600 colab_exp7_multi.py

Per dataset (POPE, MME) extract per item: p_yes (confidence), greedy answer,
self-consistency yes-fraction over K samples (ConfLVLM-style uncertainty), OWLv2
detection grounding for object-existence questions, correctness, category.
Each dataset's RAW_JSON is dumped IMMEDIATELY after it is computed, so a timeout
cannot wipe finished datasets.
"""
import json, sys, time, subprocess
import numpy as np
K, NPOPE, NMME = 3, 450, 450
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
    s=ql.split("is there",1)[1].strip()
    for k in ("a ","an ","the "):
        if s.startswith(k): s=s[len(k):]; break
    for k in ["in the image","in this image","in the picture","in this picture",
              "please answer yes or no",".","?"]:
        s=s.replace(k,"")
    s=s.strip()
    return s if 1<=len(s.split())<=3 and s else None

def collect(name,N):
    ds=load_dataset(name,split="test",streaming=True); rows=[]
    for r in ds:
        rows.append(r)
        if len(rows)>=N: break
    return rows

log("[data] loading POPE + MME ...")
DATA={"POPE":collect("lmms-lab/POPE",NPOPE),"MME":collect("lmms-lab/MME",NMME)}
for k in DATA: log(f"  {k}: {len(DATA[k])}")

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
log("[model] loading OWLv2 ...")
op=Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
od=Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(device).eval()
log(f"[model] ready (+{time.time()-t0:.0f}s)")

def process(name,rows):
    n=len(rows); p_yes=np.zeros(n);ans=np.zeros(n,int);scf=np.zeros(n);owl=np.full(n,-1.0)
    objs=[extract_object(r["question"]) for r in rows]
    for i,r in enumerate(rows):
        img=r["image"].convert("RGB"); q=r["question"]
        if "yes or no" not in q.lower(): q+="\nAnswer the question using a single word yes or no."
        inp=proc(images=img,text=f"USER: <image>\n{q} ASSISTANT:",return_tensors="pt").to(device)
        inp["pixel_values"]=inp["pixel_values"].to(torch.float16)
        with torch.no_grad():
            lg=llava(**inp).logits[0,-1].float()
            py=torch.logsumexp(lg[YES],0);pn=torch.logsumexp(lg[NO],0)
            p_yes[i]=torch.sigmoid(py-pn).item(); ans[i]=int(p_yes[i]>=.5)
            gen=llava.generate(**inp,do_sample=True,temperature=1.0,top_p=0.9,
                  num_return_sequences=K,max_new_tokens=3,pad_token_id=tok.eos_token_id)
            tx=tok.batch_decode(gen[:,inp["input_ids"].shape[1]:],skip_special_tokens=True)
            scf[i]=sum(1 for t in tx if "yes" in t.lower())/max(len(tx),1)
            if objs[i]:
                oi=op(text=[[f"a photo of a {objs[i]}"]],images=img,return_tensors="pt").to(device)
                owl[i]=float(od(**oi).logits.sigmoid().max().item())
        if (i+1)%75==0: log(f"    {name} {i+1}/{n} (+{time.time()-t0:.0f}s)")
    gold=np.array([1 if str(r["answer"]).lower().startswith("yes") else 0 for r in rows])
    correct=(ans==gold).astype(int)
    out=dict(dataset=name,p_yes=p_yes.tolist(),answer=ans.tolist(),gold=gold.tolist(),
             correct=correct.tolist(),sc_yesfrac=scf.tolist(),owl=owl.tolist(),
             category=[r["category"] for r in rows],obj=objs)
    log(f"[{name}] acc={correct.mean():.4f} grounded={(owl>=0).sum()}/{n} (+{time.time()-t0:.0f}s)")
    log(f"RAW_JSON::{name} "+json.dumps(out))

for name,rows in DATA.items():
    try: process(name,rows)
    except Exception as e: log(f"[{name}] FAILED: {e}")
log(f"[done] +{time.time()-t0:.0f}s")
