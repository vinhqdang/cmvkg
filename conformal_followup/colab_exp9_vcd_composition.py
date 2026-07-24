"""
Experiment 9 (real, GPU): COMPOSITION with a mitigation decoder (VCD).
Run:  colab run --gpu T4 --timeout 2400 colab_exp9_vcd_composition.py 600

Shows our conformal grounding layer composes ON TOP of a SOTA-style hallucination-
mitigation decoder. We implement Visual Contrastive Decoding (VCD, Leng et al.
CVPR'24) in modern transformers -- no fork needed:
    logit_vcd = (1+a)*logit(image) - a*logit(noised image)
For POPE we compute p_yes under both plain and VCD-adjusted logits, then compare
selective risk-coverage of:
    VCD confidence            (mitigation decoder alone)
    VCD confidence + grounding (mitigation + our conformal layer)
Dumps RAW_JSON with both confidence signals + OWLv2 grounding + correctness.
"""
import json, sys, time, subprocess
import numpy as np
N = int(sys.argv[1]) if len(sys.argv) > 1 else 600
VCD_ALPHA, VCD_NOISE = 1.0, 500   # contrast strength; diffusion-style noise steps
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
    for k in ["in the image","in this image","in the picture","please answer yes or no",".","?"]:
        s=s.replace(k,"")
    s=s.strip(); return s if 1<=len(s.split())<=3 and s else None

ds=load_dataset("lmms-lab/POPE",split="test",streaming=True); rows=[]
for r in ds:
    rows.append(r)
    if len(rows)>=N: break
log(f"[data] POPE {len(rows)} (+{time.time()-t0:.0f}s)")

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

def add_noise(px, steps=VCD_NOISE):
    # diffusion-style Gaussian corruption used by VCD to build the contrastive image
    b=torch.linspace(1e-4,2e-2,1000,device=px.device)
    ab=torch.cumprod(1-b,0)[min(steps,999)]
    return torch.sqrt(ab)*px + torch.sqrt(1-ab)*torch.randn_like(px)

def p_from_logits(lg):
    py=torch.logsumexp(lg[YES],0);pn=torch.logsumexp(lg[NO],0)
    return torch.sigmoid(py-pn).item()

p_plain=np.zeros(N);p_vcd=np.zeros(N);ans=np.zeros(N,int);owl=np.full(N,-1.0)
objs=[extract_object(r["question"]) for r in rows]
for i,r in enumerate(rows):
    img=r["image"].convert("RGB")
    prompt=f"USER: <image>\n{r['question']} Answer yes or no. ASSISTANT:"
    inp=proc(images=img,text=prompt,return_tensors="pt").to(device)
    inp["pixel_values"]=inp["pixel_values"].to(torch.float16)
    with torch.no_grad():
        lg=llava(**inp).logits[0,-1].float()
        inp2=dict(inp); inp2["pixel_values"]=add_noise(inp["pixel_values"])
        lg_n=llava(**inp2).logits[0,-1].float()
        lg_vcd=(1+VCD_ALPHA)*lg - VCD_ALPHA*lg_n
        p_plain[i]=p_from_logits(lg); p_vcd[i]=p_from_logits(lg_vcd)
        ans[i]=int(p_vcd[i]>=.5)
    if (i+1)%100==0: log(f"  vcd {i+1}/{N} (+{time.time()-t0:.0f}s)")
del llava; torch.cuda.empty_cache()

log("[model] OWLv2 ...")
op=Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
od=Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(device).eval()
for i,r in enumerate(rows):
    if not objs[i]: continue
    with torch.no_grad():
        oi=op(text=[[f"a photo of a {objs[i]}"]],images=r["image"].convert("RGB"),return_tensors="pt").to(device)
        owl[i]=float(od(**oi).logits.sigmoid().max().item())

gold=np.array([1 if str(r["answer"]).lower().startswith("yes") else 0 for r in rows])
correct=(ans==gold).astype(int)
out=dict(method="VCD",p_plain=p_plain.tolist(),p_vcd=p_vcd.tolist(),answer=ans.tolist(),
         gold=gold.tolist(),correct=correct.tolist(),owl=owl.tolist(),obj=objs,
         category=[r["category"] for r in rows])
log(f"[vcd] POPE acc(VCD)={correct.mean():.4f} grounded={(owl>=0).sum()}/{N}")
log("RAW_JSON "+json.dumps(out))
log(f"[done] +{time.time()-t0:.0f}s")
