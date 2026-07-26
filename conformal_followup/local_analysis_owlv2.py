"""
Stronger-grounding analysis: does a detection-based grounding signal (OWLv2)
beat weak full-image CLIP similarity for conformal selective prediction?

Reads raw_scores.json (LLaVA confidence + CLIP grounding) and owlv2_scores.json
(OWLv2 detection grounding), verified item-aligned. Compares learned combiners
over 20 random THREE-WAY DISJOINT splits (fit combiner / calibrate tau / test).
CPU-only.

Audit statistic: the RCPS guarantee is Pr[Risk <= alpha] >= 1-delta, so besides the
mean realised risk we report the FRACTION of splits whose realised risk exceeds
alpha, to be compared against delta=0.10:
  exc(te)  = exceedance on the held-out test fold (unbiased but noisy, n_te=n/3)
  exc(all) = exceedance when the selected policy is scored on all n items
             (lower noise proxy; slightly optimistic since fit/cal folds are included)
"""
import json, os, numpy as np
from scipy.stats import beta
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

_here = os.path.dirname(os.path.abspath(__file__))
r = json.load(open(os.path.join(_here, "raw_scores.json")))
# owlv2_scores.json lives beside the raw file (scratch during dev, repo after commit)
owl_path = os.path.join(_here, "owlv2_scores.json")
if not os.path.exists(owl_path):
    owl_path = "/tmp/claude-0/-home-user-cmvkg/a1d38ea3-395b-5cee-88b0-bee647582f36/scratchpad/owlv2_scores.json"
o = json.load(open(owl_path))
assert o["gold"] == r["gold"], "misaligned!"

p_yes=np.array(r["p_yes"]); clip=np.array(r["ground"]); answer=np.array(r["answer"])
correct=np.array(r["correct"]); det=np.array(o["ground_det"]); n=len(correct)
conf=np.abs(p_yes-0.5)*2
def mm(x):
    rng=x.max()-x.min(); return (x-x.min())/rng if rng>0 else x*0
clipn=mm(clip); detn=mm(det)
clip_agree=np.where(answer==1, clipn, 1-clipn)
det_agree =np.where(answer==1, detn , 1-detn)

FEATS={
 "learned_conf":        np.column_stack([conf,p_yes]),
 "learned_conf+CLIP":   np.column_stack([conf,p_yes,clipn,clip_agree]),
 "learned_conf+OWLv2":  np.column_stack([conf,p_yes,detn,det_agree]),
 "learned_conf+both":   np.column_stack([conf,p_yes,clipn,clip_agree,detn,det_agree]),
}
ALPHA,DELTA=0.10,0.10
def cp_upper(e,k,d):
    if k==0: return 1.0
    return 1.0 if e==k else float(beta.ppf(1-d,e+1,k-e))
def rcps(sc,cc,st,ct,s_all,c_all):
    """Returns (coverage on test, realised risk on test, realised risk on all items)."""
    tau,best=None,-1
    for t in np.unique(sc):
        m=sc>=t; k=int(m.sum())
        if k==0: continue
        e=int((1-cc[m]).sum())
        if cp_upper(e,k,DELTA)<=ALPHA and k>best: best,tau=k,float(t)
    if tau is None: return 0.0,0.0,0.0
    m=st>=tau; k=int(m.sum())
    cov,err=(k/len(st),float((1-ct[m]).mean())) if k else (0.0,0.0)
    ma=s_all>=tau; ka=int(ma.sum())
    err_all=float((1-c_all[ma]).mean()) if ka else 0.0
    return cov,err,err_all
def aurc(s,c):
    c=c[np.argsort(-s)]; return float((np.cumsum(1-c)/np.arange(1,len(c)+1)).mean())

rng=np.random.default_rng(0); agg={}
for _ in range(20):
    idx=rng.permutation(n); t3=n//3; tr,cal,te=idx[:t3],idx[t3:2*t3],idx[2*t3:]
    cov,err,err_all=rcps(conf[cal],correct[cal],conf[te],correct[te],conf,correct)
    agg.setdefault("raw_conf",[]).append((aurc(conf[te],correct[te]),cov,err,roc_auc_score(correct[te],conf[te]),err_all))
    for name,X in FEATS.items():
        clf=LogisticRegression(max_iter=1000).fit(X[tr],correct[tr])
        pc=clf.predict_proba(X)[:,1]
        cov,err,err_all=rcps(pc[cal],correct[cal],pc[te],correct[te],pc,correct)
        agg.setdefault(name,[]).append((aurc(pc[te],correct[te]),cov,err,roc_auc_score(correct[te],pc[te]),err_all))

print(f"POPE n={n}  VLM acc={correct.mean():.3f}  (20 three-way disjoint splits, mean±std)\n")
print(f"{'score':22s}{'AURC↓':>13s}{'cov@10%↑':>13s}{'err@10%':>11s}{'AUROC↑':>12s}{'exc(te)':>9s}{'exc(all)':>10s}")
for name in ["raw_conf","learned_conf","learned_conf+CLIP","learned_conf+OWLv2","learned_conf+both"]:
    a=np.array(agg[name]); m,s=a.mean(0),a.std(0)
    xt=float((a[:,2]>ALPHA+1e-12).mean()); xa=float((a[:,4]>ALPHA+1e-12).mean())
    print(f"{name:22s}{m[0]:.4f}±{s[0]:.3f}{m[1]:.3f}±{s[1]:.3f}{m[2]:.3f}±{s[2]:.3f}{m[3]:.4f}±{s[3]:.3f}{xt:8.2f} {xa:9.2f}")
print(f"\nExceedance = fraction of the 20 splits with realised risk > alpha={ALPHA:.2f}; "
      f"guarantee requires <= delta={DELTA:.2f}.")
print("  exc(te)=test fold (n_te=%d, unbiased/noisy)   exc(all)=all %d items (low-noise proxy)"
      % (n-2*(n//3), n))

base=np.array(agg["learned_conf"])
print("\nPaired grounding gains vs learned_conf (same splits):")
for name in ["learned_conf+CLIP","learned_conf+OWLv2","learned_conf+both"]:
    a=np.array(agg[name])
    dcov=(a[:,1]-base[:,1]); daurc=(base[:,0]-a[:,0])
    print(f"  {name:20s} Δcov@10%={dcov.mean():+.4f}±{dcov.std():.4f}   ΔAURC={daurc.mean():+.4f}±{daurc.std():.4f}")
