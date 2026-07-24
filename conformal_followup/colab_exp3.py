"""
Experiment 3 (real VLM, GPU): conformal SELECTIVE prediction on POPE.
Run:  colab run --gpu L4 --timeout 2400 colab_exp3.py 1000

Task: POPE yes/no object-existence. The VLM answers; we selectively TRUST answers
so the error rate among trusted answers <= alpha (RCPS, Clopper-Pearson bound),
and abstain otherwise. Coverage = fraction we can confidently answer.

Compares two confidence scores:
  base       = VLM self-confidence (softmax margin over yes/no tokens)
  structured = base fused with CLIP visual grounding agreement
Thesis: structured -> higher coverage at the SAME valid error guarantee.
"""
import json, sys, time, subprocess
import numpy as np

ALPHA, DELTA = 0.10, 0.10
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
np.random.seed(0)
def log(*a): print(*a, flush=True)
t0 = time.time()

for pkg in ["datasets","bitsandbytes"]:
    try: __import__(pkg)
    except Exception: subprocess.run([sys.executable,"-m","pip","install","-q",pkg], check=True)

import torch
from datasets import load_dataset
from transformers import (AutoProcessor, LlavaForConditionalGeneration,
                          CLIPProcessor, CLIPModel, BitsAndBytesConfig)
from scipy.stats import beta
device = "cuda" if torch.cuda.is_available() else "cpu"
log(f"[env] torch {torch.__version__} dev={torch.cuda.get_device_name(0) if device=='cuda' else 'cpu'} (+{time.time()-t0:.0f}s)")

# ---- data ----
log("[data] streaming lmms-lab/POPE ...")
ds = load_dataset("lmms-lab/POPE", split="test", streaming=True)
rows = []
for r in ds:
    rows.append(r)
    if len(rows) >= N: break
imgs   = [r["image"].convert("RGB") for r in rows]
def obj_of(q):
    q=q.lower()
    for k in ["is there a","is there an","in the image?","in the picture?","?"]: q=q.replace(k,"")
    return q.strip()
objs   = [obj_of(r["question"]) for r in rows]
gold   = np.array([1 if str(r["answer"]).lower().startswith("yes") else 0 for r in rows])
log(f"[data] {len(rows)} probes (+{time.time()-t0:.0f}s)")

# ---- LLaVA ----
MODEL="llava-hf/llava-1.5-7b-hf"
log(f"[model] loading {MODEL} (4-bit) ...")
proc = AutoProcessor.from_pretrained(MODEL)
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
llava = LlavaForConditionalGeneration.from_pretrained(
    MODEL, quantization_config=bnb, torch_dtype=torch.float16,
    low_cpu_mem_usage=True, device_map="auto").eval()
tok = proc.tokenizer
def ids_for(words):
    out=set()
    for w in words:
        for v in (w, " "+w):
            t=tok(v, add_special_tokens=False).input_ids
            if len(t)==1: out.add(t[0])
    return list(out)
YES, NO = ids_for(["yes","Yes","YES"]), ids_for(["no","No","NO"])
log(f"[model] ready (+{time.time()-t0:.0f}s)  yes_ids={YES} no_ids={NO}")

p_yes = np.zeros(len(rows)); answer = np.zeros(len(rows), dtype=int)
for i,(img,q) in enumerate(zip(imgs, [r["question"] for r in rows])):
    prompt=f"USER: <image>\n{q}\nAnswer the question using a single word yes or no. ASSISTANT:"
    inp = proc(images=img, text=prompt, return_tensors="pt").to(device)
    inp["pixel_values"]=inp["pixel_values"].to(torch.float16)
    with torch.no_grad():
        logit = llava(**inp).logits[0,-1].float()
    py = torch.logsumexp(logit[YES],0); pn = torch.logsumexp(logit[NO],0)
    prob_yes = torch.sigmoid(py-pn).item()          # P(yes) over {yes,no}
    p_yes[i]=prob_yes; answer[i]=int(prob_yes>=0.5)
    if (i+1)%100==0: log(f"  [llava] {i+1}/{len(rows)} (+{time.time()-t0:.0f}s)")

correct = (answer==gold).astype(int)
acc = correct.mean()
log(f"[vlm] raw POPE accuracy={acc:.4f}  (+{time.time()-t0:.0f}s)")

# ---- CLIP grounding ----
log("[model] CLIP-B/32 for grounding ...")
clip=CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
cproc=CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
ground=np.zeros(len(rows))
for i,(img,obj) in enumerate(zip(imgs,objs)):
    with torch.no_grad():
        ci=cproc(text=[f"a photo of a {obj}"],images=img,return_tensors="pt",padding=True).to(device)
        ground[i]=clip(**ci).logits_per_image.item()/100.0
    if (i+1)%200==0: log(f"  [clip] {i+1}/{len(rows)} (+{time.time()-t0:.0f}s)")

def mm(x):
    r=x.max()-x.min(); return (x-x.min())/r if r>0 else x*0
base = np.abs(p_yes-0.5)*2                     # VLM confidence margin in [0,1]
g = mm(ground)
# grounding AGREEMENT with the model's answer: yes-answers want high grounding
agree = np.where(answer==1, g, 1-g)
structured = mm(mm(base)+agree)                # late fusion of confidence + agreement

# ---- RCPS selective risk control (Clopper-Pearson upper bound) ----
def cp_upper(e,k,delta):                        # 1-delta upper conf bound on error rate
    if k==0: return 1.0
    return 1.0 if e==k else float(beta.ppf(1-delta, e+1, k-e))
def select_tau(s,corr,alpha,delta):
    """most-accepting tau s.t. CP upper bound on error among accepted <= alpha."""
    best_tau, best_cov = None, -1
    for tau in np.unique(s):
        acc_mask=s>=tau; k=int(acc_mask.sum())
        if k==0: continue
        e=int((1-corr[acc_mask]).sum())
        if cp_upper(e,k,delta)<=alpha and k>best_cov:
            best_cov, best_tau = k, float(tau)
    return best_tau
def eval_tau(s,corr,tau):
    if tau is None: return dict(coverage=0.0,err_among_accepted=0.0,accepted=0)
    m=s>=tau; k=int(m.sum())
    return dict(coverage=round(k/len(s),4),
                err_among_accepted=round(float((1-corr[m]).mean()),4), accepted=k)

idx=np.random.permutation(len(rows)); h=len(idx)//2; cal,test=idx[:h],idx[h:]
def auroc(s,c):
    from sklearn.metrics import roc_auc_score
    try: return round(float(roc_auc_score(c,s)),4)
    except Exception: return None
res={"alpha":ALPHA,"delta":DELTA,"n":len(rows),"vlm_pope_acc":round(float(acc),4)}
for name,s in [("base_vlm_confidence",base),("structured_conf+grounding",structured)]:
    tau=select_tau(s[cal],correct[cal],ALPHA,DELTA)
    res[name]={"tau":round(tau,4) if tau is not None else None,
               "auroc_correctness":auroc(s[test],correct[test]),
               **eval_tau(s[test],correct[test],tau)}
log("RESULT_JSON "+json.dumps(res))
# dump raw per-item signals so ALL further fusion/calibration analysis runs
# locally on CPU (no more GPU spend).
raw={"p_yes":[round(float(x),5) for x in p_yes],
     "ground":[round(float(x),5) for x in ground],
     "answer":[int(x) for x in answer],
     "gold":[int(x) for x in gold],
     "correct":[int(x) for x in correct]}
log("RAW_JSON "+json.dumps(raw))
log(f"[done] +{time.time()-t0:.0f}s")
