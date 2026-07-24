"""
Cross-backbone check: does 'grounding improves conformal selective prediction'
hold on a second VLM? Reads exp8_qwen_pope.json (Qwen2-VL-2B, POPE-adversarial).
CPU-only, 20 splits.
"""
import json, os, sys, numpy as np
from scipy.stats import beta
from sklearn.linear_model import LogisticRegression
_here=os.path.dirname(os.path.abspath(__file__))
path=sys.argv[1] if len(sys.argv)>1 else os.path.join(_here,"exp8_qwen_pope.json")
d=json.load(open(path))
p_yes=np.array(d["p_yes"]);answer=np.array(d["answer"]);correct=np.array(d["correct"]);owl=np.array(d["owl"])
n=len(correct);conf=np.abs(p_yes-.5)*2
g=owl.copy();g[g<0]=np.median(g[g>=0]);mm=lambda x:(x-x.min())/(x.max()-x.min()+1e-9)
gn=mm(g);g_ag=np.where(answer==1,gn,1-gn)
SC={"Confidence only":np.column_stack([p_yes,conf]),
    "Confidence + grounding (ours)":np.column_stack([p_yes,conf,gn,g_ag])}
GRID=np.linspace(.02,1,99)
def rc(s,c):
    c=c[np.argsort(-s)];return np.interp(GRID,np.arange(1,len(c)+1)/len(c),np.cumsum(1-c)/np.arange(1,len(c)+1))
def cpu(e,k,dl):return 1.0 if k==0 or e==k else float(beta.ppf(1-dl,e+1,k-e))
def cov(s,cc,st,ct,a=.1,dl=.1):
    tau,b=None,-1
    for t in np.unique(s):
        m=s>=t;k=int(m.sum())
        if k and cpu(int((1-cc[m]).sum()),k,dl)<=a and k>b:b,tau=k,float(t)
    return 0 if tau is None else (st>=tau).sum()/len(st)
rng=np.random.default_rng(0);A={k:{"a":[],"c":[]} for k in SC}
for _ in range(20):
    idx=rng.permutation(n);h=n//2;cal,te=idx[:h],idx[h:]
    for name,X in SC.items():
        clf=LogisticRegression(max_iter=1000).fit(X[cal],correct[cal]);pc=clf.predict_proba(X)[:,1]
        A[name]["a"].append(rc(pc[te],correct[te]).mean());A[name]["c"].append(cov(pc[cal],correct[cal],pc[te],correct[te]))
print(f"{d.get('model','model')} on POPE-adversarial  acc={correct.mean():.3f}  (20 splits)")
print(f"{'score':32s}{'AURC':>10s}{'cov@10%':>10s}")
for k in SC:
    a=np.array(A[k]['a']);c=np.array(A[k]['c']);print(f"{k:32s}{a.mean():.4f}   {c.mean()*100:5.1f}%")
gc=(np.array(A['Confidence + grounding (ours)']['c'])-np.array(A['Confidence only']['c']))*100
ga=np.array(A['Confidence only']['a'])-np.array(A['Confidence + grounding (ours)']['a'])
print(f"Grounding gain: Δcov@10%={gc.mean():+.1f}±{gc.std():.1f}pp  ΔAURC={ga.mean():+.4f}±{ga.std():.4f}")
