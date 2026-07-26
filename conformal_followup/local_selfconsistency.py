"""
Faithful ConfLVLM head-to-head on POPE-adversarial (hardest split), LLaVA-1.5-7B.
Data: exp7_pope.json (real self-consistency over K=3 samples + OWLv2 grounding).

Baseline (ConfLVLM-faithful): learned combiner over model-INTERNAL signals only,
including self-consistency  [p_yes, conf, sc_yesfrac, sc_conf].
Ours: internal + OWLv2 structured grounding.
Reports risk-coverage (AURC, cov@10%) over 20 THREE-WAY DISJOINT splits
(fit combiner / calibrate tau / test) + pure self-consistency ref.

Audit statistic: the guarantee is Pr[Risk <= alpha] >= 1-delta, so besides the mean
realised risk we report the FRACTION of splits with realised risk > alpha, vs
delta=0.10.  exc(te)=held-out test fold (unbiased, noisy);  exc(all)=selected policy
scored on all n items (low-noise proxy, slightly optimistic).
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
ALPHA=.10; DELTA=.10
def cov_at(s,c_cal,s_te,c_te,s_all,c_all,alpha=ALPHA,dl=DELTA):
    """Returns (coverage on test, realised risk on test, realised risk on all items)."""
    tau,best=None,-1
    for t in np.unique(s):
        m=s>=t;k=int(m.sum())
        if k==0: continue
        e=int((1-c_cal[m]).sum())
        if cp_upper(e,k,dl)<=alpha and k>best: best,tau=k,float(t)
    if tau is None: return 0.0,0.0,0.0
    m=s_te>=tau;k=int(m.sum())
    cov,err=(k/len(s_te),float((1-c_te[m]).mean())) if k else (0.0,0.0)
    ma=s_all>=tau;ka=int(ma.sum())
    return cov,err,(float((1-c_all[ma]).mean()) if ka else 0.0)

rng=np.random.default_rng(0)
agg={k:{"aurc":[],"cov":[],"risk":[],"risk_all":[]} for k in SCORES}
for _ in range(20):
    idx=rng.permutation(n);t3=n//3;tr,cal,te=idx[:t3],idx[t3:2*t3],idx[2*t3:]
    for name,X in SCORES.items():
        if X.ndim==1:
            s_all=X
        else:
            clf=LogisticRegression(max_iter=1000).fit(X[tr],correct[tr])
            s_all=clf.predict_proba(X)[:,1]
        sc_cal,sc_te=s_all[cal],s_all[te]
        agg[name]["aurc"].append(rc(sc_te,correct[te]).mean())
        cv,rk,rka=cov_at(sc_cal,correct[cal],sc_te,correct[te],s_all,correct)
        agg[name]["cov"].append(cv);agg[name]["risk"].append(rk);agg[name]["risk_all"].append(rka)

print(f"POPE-adversarial (hardest split)  n={n}  LLaVA acc={correct.mean():.3f}  "
      f"(20 three-way disjoint splits)\n")
print(f"{'score':48s}{'AURC↓':>13s}{'cov@10%↑':>13s}{'risk@10%':>10s}{'exc(te)':>9s}{'exc(all)':>10s}")
for name in SCORES:
    a=np.array(agg[name]['aurc']);c=np.array(agg[name]['cov'])
    rk=np.array(agg[name]['risk']);rka=np.array(agg[name]['risk_all'])
    print(f"{name:48s}{a.mean():.4f}±{a.std():.3f}{c.mean()*100:6.1f}±{c.std()*100:.1f}"
          f"{rk.mean():9.3f}{(rk>ALPHA+1e-12).mean():8.2f} {(rka>ALPHA+1e-12).mean():9.2f}")
print(f"\nExceedance = fraction of the 20 splits with realised risk > alpha={ALPHA:.2f}; "
      f"guarantee requires <= delta={DELTA:.2f}.")
print("  exc(te)=test fold (n_te=%d, unbiased/noisy)   exc(all)=all %d items (low-noise proxy)"
      % (n-2*(n//3), n))
base=np.array(agg["internal (conf+self-cons) [ConfLVLM-faithful]"]["cov"])
ours=np.array(agg["internal + grounding (ours)"]["cov"])
ba=np.array(agg["internal (conf+self-cons) [ConfLVLM-faithful]"]["aurc"])
oa=np.array(agg["internal + grounding (ours)"]["aurc"])
print(f"\nGrounding gain over ConfLVLM-faithful internal (paired): "
      f"Δcov@10%={(ours-base).mean()*100:+.1f}±{(ours-base).std()*100:.1f}pp  "
      f"ΔAURC={(ba-oa).mean():+.4f}±{(ba-oa).std():.4f}")
