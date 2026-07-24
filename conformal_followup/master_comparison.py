"""
Master comparison across all datasets/backbones/methods collected this session.
For each row: base accuracy, and (confidence-only vs +grounding) selective
prediction under RCPS with a proper 3-way split (train combiner / calibrate / test).
Reports AURC, coverage@10% risk, and realized test error (validity). CPU-only.
"""
import json, os, numpy as np, warnings; warnings.filterwarnings("ignore")
from scipy.stats import beta
from sklearn.linear_model import LogisticRegression
_here=os.path.dirname(os.path.abspath(__file__))
J=lambda f: json.load(open(os.path.join(_here,f)))
mm=lambda x:(x-x.min())/(x.max()-x.min()+1e-9)
GRID=np.linspace(.02,1,99)
def rc(s,c):
    c=c[np.argsort(-s)];return np.interp(GRID,np.arange(1,len(c)+1)/len(c),np.cumsum(1-c)/np.arange(1,len(c)+1))
def cpu(e,k,dl):return 1.0 if k==0 or e==k else float(beta.ppf(1-dl,e+1,k-e))
def cov_err(sc,cc,st,ct,al=.1,dl=.1):
    tau,b=None,-1
    for t in np.unique(sc):
        m=sc>=t;k=int(m.sum())
        if k and cpu(int((1-cc[m]).sum()),k,dl)<=al and k>b:b,tau=k,float(t)
    if tau is None:return 0,0
    m=st>=tau;k=int(m.sum());return k/len(st),((1-ct[m]).mean() if k else 0)

def evaluate(p_yes,answer,correct,owl,extra=None):
    """returns dict of metrics for confidence-only and +grounding (if owl available)."""
    n=len(correct);conf=np.abs(p_yes-.5)*2
    feats={"conf":[p_yes,conf]}
    if extra is not None: feats["conf"].append(extra)      # e.g. self-consistency
    has_g=owl is not None and (np.array(owl)>=0).any()
    if has_g:
        g=np.array(owl,float);g[g<0]=np.median(g[g>=0]);gn=mm(g);gag=np.where(answer==1,gn,1-gn)
        feats["grnd"]=feats["conf"]+[gn,gag]
    rng=np.random.default_rng(0);R={k:{"aurc":[],"cov":[],"err":[]} for k in feats}
    for _ in range(20):
        idx=rng.permutation(n);t=n//3;tr,cal,te=idx[:t],idx[t:2*t],idx[2*t:]
        if len(np.unique(correct[tr]))<2: continue          # tiny/degenerate fold
        for k,cols in feats.items():
            X=np.column_stack(cols)
            clf=LogisticRegression(max_iter=1000).fit(X[tr],correct[tr]);pc=clf.predict_proba(X)[:,1]
            cv,er=cov_err(pc[cal],correct[cal],pc[te],correct[te])
            R[k]["aurc"].append(rc(pc[te],correct[te]).mean());R[k]["cov"].append(cv);R[k]["err"].append(er)
    if not R["conf"]["aurc"]: return None, has_g            # too small to evaluate
    return {k:{m:np.mean(v) for m,v in d.items()} for k,d in R.items()}, has_g

ROWS=[]
# POPE-1500 LLaVA (CLIP+OWL available; use OWL as grounding)
r=J("raw_scores.json");o=J("owlv2_scores.json")
ROWS.append(("POPE (1500)","LLaVA-1.5",np.array(r["p_yes"]),np.array(r["answer"]),np.array(r["correct"]),o["ground_det"],None))
# POPE-adv LLaVA + self-consistency
d=J("exp7_pope.json")
ROWS.append(("POPE-adv (450)","LLaVA-1.5",np.array(d["p_yes"]),np.array(d["answer"]),np.array(d["correct"]),d["owl"],np.array(d["sc_yesfrac"])))
# POPE-adv Qwen
d=J("exp8_qwen_pope.json")
ROWS.append(("POPE-adv (600)","Qwen2-VL-2B",np.array(d["p_yes"]),np.array(d["answer"]),np.array(d["correct"]),d["owl"],None))
# POPE-adv VCD
d=J("exp9_vcd_pope.json");ans=np.array(d["answer"])
ROWS.append(("POPE-adv (600)","LLaVA+VCD",np.array(d["p_vcd"]),ans,np.array(d["correct"]),d["owl"],None))
# MME LLaVA (no grounding)
d=J("exp10_mme.json")
ROWS.append(("MME (700)","LLaVA-1.5",np.array(d["p_yes"]),np.array(d["answer"]),np.array(d["correct"]),d["owl"],None))
# MME-existence LLaVA (fully grounded, small n)
d=J("exp11_mme_exist.json")
ROWS.append(("MME-exist (60)","LLaVA-1.5",np.array(d["p_yes"]),np.array(d["answer"]),np.array(d["correct"]),d["owl"],None))
# GQA LLaVA (no grounding)
d=J("exp11_gqa.json")
ROWS.append(("GQA (700)","LLaVA-1.5",np.array(d["p_yes"]),np.array(d["answer"]),np.array(d["correct"]),d["owl"],None))
# HallusionBench LLaVA (no grounding)
d=J("exp11_hallusion.json")
ROWS.append(("HallusionBench (447)","LLaVA-1.5",np.array(d["p_yes"]),np.array(d["answer"]),np.array(d["correct"]),d["owl"],None))

print("="*104)
print(f"{'dataset':22s}{'backbone':13s}{'acc':>6s}{'grnd':>6s}"
      f"{'AURC conf':>11s}{'AURC +g':>10s}{'cov@10 conf':>13s}{'cov@10 +g':>11s}{'valid':>8s}")
print("-"*104)
for name,bb,p,a,c,owl,extra in ROWS:
    res,has_g=evaluate(p,a,c,owl,extra)
    acc=c.mean()
    if res is None:
        print(f"{name:22s}{bb:13s}{acc*100:5.1f}%{('yes' if has_g else 'no'):>6s}"
              f"{'  n too small for conformal split':>44s}")
        continue
    cf=res["conf"];gf=res.get("grnd")
    au_c=f"{cf['aurc']:.4f}";cov_c=f"{cf['cov']*100:.1f}%";err=cf['err']
    if gf:
        au_g=f"{gf['aurc']:.4f}";cov_g=f"{gf['cov']*100:.1f}%";err=max(err,gf['err'])
    else: au_g,cov_g="  n/a","  n/a"
    valid="OK" if err<=0.11 else "hi"
    print(f"{name:22s}{bb:13s}{acc*100:5.1f}%{('yes' if has_g else 'no'):>6s}"
          f"{au_c:>11s}{au_g:>10s}{cov_c:>13s}{cov_g:>11s}{valid:>8s}")
print("="*104)
print("grnd=yes: object-existence questions where OWLv2 grounding applies (POPE).")
print("MME/HallusionBench: grounding n/a (non-object questions); confidence-only, validity still holds.")
