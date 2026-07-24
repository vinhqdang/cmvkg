"""
Experiment 8 (real, GPU): second VLM backbone for generalization.
Run:  colab run --gpu T4 --timeout 2400 colab_exp8_backbone.py 600

Repeats the POPE selective-conformal signal extraction with Qwen2-VL-2B-Instruct
(architecturally distinct from LLaVA) + OWLv2 grounding, to test whether the
"structured grounding improves conformal selective prediction" result generalizes
across VLM backbones. Dumps RAW_JSON.
"""
import json, sys, time, subprocess
import numpy as np
N = int(sys.argv[1]) if len(sys.argv) > 1 else 600
def log(*a): print(*a, flush=True)
t0=time.time()
for pkg in ["datasets","qwen-vl-utils","accelerate"]:
    try: __import__(pkg.replace("-","_"))
    except Exception: subprocess.run([sys.executable,"-m","pip","install","-q",pkg],check=True)
import torch
from datasets import load_dataset
from transformers import (AutoProcessor, Qwen2VLForConditionalGeneration,
                          Owlv2Processor, Owlv2ForObjectDetection)
device="cuda" if torch.cuda.is_available() else "cpu"
log(f"[env] torch {torch.__version__} dev={torch.cuda.get_device_name(0) if device=='cuda' else 'cpu'} (+{time.time()-t0:.0f}s)")

def extract_object(q):
    ql=q.lower()
    if "is there" not in ql: return None
    s=ql.split("is there",1)[1].strip()
    for k in ("a ","an ","the "):
        if s.startswith(k): s=s[len(k):]; break
    for k in ["in the image","in this image","in the picture","please answer yes or no",".","?"]:
        s=s.replace(k,"")
    s=s.strip(); return s if 1<=len(s.split())<=3 and s else None

ds=load_dataset("lmms-lab/POPE",split="test",streaming=True); rows=[]
for r in ds:
    rows.append(r)
    if len(rows)>=N: break
log(f"[data] POPE {len(rows)} (+{time.time()-t0:.0f}s)")

M="Qwen/Qwen2-VL-2B-Instruct"
log(f"[model] {M} ...")
proc=AutoProcessor.from_pretrained(M)
qwen=Qwen2VLForConditionalGeneration.from_pretrained(M,torch_dtype=torch.float16,
      device_map="auto").eval()
tok=proc.tokenizer
def ids_for(ws):
    o=set()
    for w in ws:
        for v in (w," "+w):
            t=tok(v,add_special_tokens=False).input_ids
            if len(t)==1: o.add(t[0])
    return list(o)
YES,NO=ids_for(["yes","Yes","YES"]),ids_for(["no","No","NO"])
log(f"[model] ready yes={YES} no={NO} (+{time.time()-t0:.0f}s)")

p_yes=np.zeros(N);ans=np.zeros(N,int);owl=np.full(N,-1.0)
objs=[extract_object(r["question"]) for r in rows]
for i,r in enumerate(rows):
    img=r["image"].convert("RGB")
    msgs=[{"role":"user","content":[{"type":"image"},
           {"type":"text","text":r["question"]+" Answer yes or no."}]}]
    text=proc.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
    inp=proc(text=[text],images=[img],return_tensors="pt").to(device)
    with torch.no_grad():
        lg=qwen(**inp).logits[0,-1].float()
        py=torch.logsumexp(lg[YES],0);pn=torch.logsumexp(lg[NO],0)
        p_yes[i]=torch.sigmoid(py-pn).item();ans[i]=int(p_yes[i]>=.5)
    if (i+1)%100==0: log(f"  qwen {i+1}/{N} (+{time.time()-t0:.0f}s)")
del qwen; torch.cuda.empty_cache()

log("[model] OWLv2 ...")
op=Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
od=Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(device).eval()
for i,r in enumerate(rows):
    if not objs[i]: continue
    with torch.no_grad():
        oi=op(text=[[f"a photo of a {objs[i]}"]],images=r["image"].convert("RGB"),return_tensors="pt").to(device)
        owl[i]=float(od(**oi).logits.sigmoid().max().item())
    if (i+1)%150==0: log(f"  owl {i+1}/{N} (+{time.time()-t0:.0f}s)")

gold=np.array([1 if str(r["answer"]).lower().startswith("yes") else 0 for r in rows])
correct=(ans==gold).astype(int)
out=dict(model="Qwen2-VL-2B",p_yes=p_yes.tolist(),answer=ans.tolist(),gold=gold.tolist(),
         correct=correct.tolist(),owl=owl.tolist(),obj=objs,
         category=[r["category"] for r in rows])
log(f"[qwen] POPE acc={correct.mean():.4f} grounded={(owl>=0).sum()}/{N}")
log("RAW_JSON "+json.dumps(out))
log(f"[done] +{time.time()-t0:.0f}s")
