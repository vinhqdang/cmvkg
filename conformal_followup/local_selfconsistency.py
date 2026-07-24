"""
Faithful ConfLVLM head-to-head on POPE-adversarial (hardest split), LLaVA-1.5-7B.
Data: exp7_pope.json (real self-consistency over K=3 samples + OWLv2 grounding).

Baseline (ConfLVLM-faithful): learned combiner over model-INTERNAL signals only,
including self-consistency  [p_yes, conf, sc_yesfrac, sc_conf].
Ours: internal + OWLv2 structured grounding.
Reports risk-coverage (AURC, cov@10%) over 20 splits + pure self-consistency ref.
"""
import json, os, numpy as np
from scipy.stats import beta
from sklearn.linear_model import LogisticRegression

_here=os.path.dirname(os.path.abspath(__file__))
d=json.load(open(os.path.join(_here,"exp7_pope.json")))
p_yes=np.array(d["p_yes"]);answer=np.array(d["answer"]);correct=np.array(d["correct"])
sc=np.array(d["sc_yesfrac"]);owl=np.array(d["owl"]);n=len(correct)
conf=np.abs(p_yes-.5)*2; sc_conf=np.abs(sc-.5)*2
# ungrounded (-1) -> neutral median of grounded
g=owl.copy(); med=np.median(g[g>=0]); g[g<0]=med
mm=lambda x:(x-x.min())/(x.max()-x.min()+1e-9)
gn=mm(g); g_ag=np.where(answer==1,gn,1-gn)

SCORES={
 "self-consistency only":sc_conf,                                   # literal ConfLVLM signal
 "internal (conf+self-cons) [ConfLVLM-faithful]":np.column_stack([p_yes,conf,sc,sc_conf]),
 "internal + grounding (ours)":np.column_stack([p_yes,conf,sc,sc_conf,gn,g_ag]),
}
GRID=np.linspace(.02,1,99)
def rc(s,c):
    c=c[np.argsort(-s)];cov=np.arange(1,len(c)+1)/len(c)
    return np.interp(GRID,cov,np.cumsum(1-c)/np.arange(1,len(c)+1))
def cp_upper(e,k,dl):
    if k==0: return 1.0
    return 1.0 if e==k else float(beta.ppf(1-dl,e+1,k-e))
def cov_at(s,c_cal,s_te,c_te,alpha=.10,dl=.10):
    tau,best=None,-1
    for t in np.unique(s):
        m=s>=t;k=int(m.sum())
        if k==0: continue
        e=int((1-c_cal[m]).sum())
        if cp_upper(e,k,dl)<=alpha and k>best: best,tau=k,float(t)
    if tau is None: return 0.0
    m=s_te>=tau;return int(m.sum())/len(s_te)

rng=np.random.default_rng(0)
agg={k:{"aurc":[],"cov":[]} for k in SCORES}
for _ in range(20):
    idx=rng.permutation(n);h=n//2;cal,te=idx[:h],idx[h:]
    for name,X in SCORES.items():
        if X.ndim==1:
            sc_cal,sc_te=X[cal],X[te]
        else:
            clf=LogisticRegression(max_iter=1000).fit(X[cal],correct[cal])
            pc=clf.predict_proba(X)[:,1]; sc_cal,sc_te=pc[cal],pc[te]
        agg[name]["aurc"].append(rc(sc_te,correct[te]).mean())
        agg[name]["cov"].append(cov_at(sc_cal,correct[cal],sc_te,correct[te]))

print(f"POPE-adversarial (hardest split)  n={n}  LLaVA acc={correct.mean():.3f}  (20 splits)\n")
print(f"{'score':48s}{'AURC↓':>13s}{'cov@10%↑':>13s}")
for name in SCORES:
    a=np.array(agg[name]['aurc']);c=np.array(agg[name]['cov'])
    print(f"{name:48s}{a.mean():.4f}±{a.std():.3f}{c.mean()*100:6.1f}±{c.std()*100:.1f}")
base=np.array(agg["internal (conf+self-cons) [ConfLVLM-faithful]"]["cov"])
ours=np.array(agg["internal + grounding (ours)"]["cov"])
ba=np.array(agg["internal (conf+self-cons) [ConfLVLM-faithful]"]["aurc"])
oa=np.array(agg["internal + grounding (ours)"]["aurc"])
print(f"\nGrounding gain over ConfLVLM-faithful internal (paired): "
      f"Δcov@10%={(ours-base).mean()*100:+.1f}±{(ours-base).std()*100:.1f}pp  "
      f"ΔAURC={(ba-oa).mean():+.4f}±{(ba-oa).std():.4f}")
