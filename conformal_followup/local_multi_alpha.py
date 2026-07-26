"""
Multi-alpha results: how much coverage each score buys at several risk targets,
under RCPS (Clopper-Pearson). Real POPE / LLaVA-1.5-7B, 20 THREE-WAY DISJOINT
splits (fit combiner / calibrate tau / test). CPU-only.
Also writes coverage_vs_alpha.png.

Audit statistic: the guarantee is Pr[Risk <= alpha] >= 1-delta, so besides the mean
realised risk we report the FRACTION of splits with realised risk > alpha, vs
delta=0.10.  exc(te)=held-out test fold (unbiased, noisy);  exc(all)=selected policy
scored on all n items (low-noise proxy, slightly optimistic).
"""
import json, os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import beta
from sklearn.linear_model import LogisticRegression

_here=os.path.dirname(os.path.abspath(__file__))
r=json.load(open(os.path.join(_here,"raw_scores.json")))
o=json.load(open(os.path.join(_here,"owlv2_scores.json")))
p_yes=np.array(r["p_yes"]);answer=np.array(r["answer"]);correct=np.array(r["correct"])
det=np.array(o["ground_det"]);n=len(correct);conf=np.abs(p_yes-.5)*2
mm=lambda x:(x-x.min())/(x.max()-x.min())
detn=mm(det);det_ag=np.where(answer==1,detn,1-detn)
clipn=mm(np.array(r["ground"]));clip_ag=np.where(answer==1,clipn,1-clipn)
SC={"Confidence only":np.column_stack([conf,p_yes]),
    "+ CLIP grounding":np.column_stack([conf,p_yes,clipn,clip_ag]),
    "+ OWLv2 grounding (ours)":np.column_stack([conf,p_yes,detn,det_ag])}
ALPHAS=[0.05,0.10,0.15,0.20]; DELTA=0.10
def cp_upper(e,k,d):
    if k==0: return 1.0
    return 1.0 if e==k else float(beta.ppf(1-d,e+1,k-e))
def rcps(sc,cc,st,ct,s_all,c_all,alpha):
    """Returns (coverage on test, realised risk on test, realised risk on all items)."""
    tau,best=None,-1
    for t in np.unique(sc):
        m=sc>=t;k=int(m.sum())
        if k==0: continue
        e=int((1-cc[m]).sum())
        if cp_upper(e,k,DELTA)<=alpha and k>best: best,tau=k,float(t)
    if tau is None: return 0.0,0.0,0.0
    m=st>=tau;k=int(m.sum())
    cov,err=(k/len(st),float((1-ct[m]).mean())) if k else (0.0,0.0)
    ma=s_all>=tau;ka=int(ma.sum())
    return cov,err,(float((1-c_all[ma]).mean()) if ka else 0.0)

rng=np.random.default_rng(0)
res={k:{a:[] for a in ALPHAS} for k in SC}
rsk={k:{a:[] for a in ALPHAS} for k in SC}
rska={k:{a:[] for a in ALPHAS} for k in SC}
for _ in range(20):
    idx=rng.permutation(n);t3=n//3;tr,cal,te=idx[:t3],idx[t3:2*t3],idx[2*t3:]
    for name,X in SC.items():
        clf=LogisticRegression(max_iter=1000).fit(X[tr],correct[tr])
        pc=clf.predict_proba(X)[:,1]
        for a in ALPHAS:
            cv,rk,rka=rcps(pc[cal],correct[cal],pc[te],correct[te],pc,correct,a)
            res[name][a].append(cv);rsk[name][a].append(rk);rska[name][a].append(rka)

print(f"Coverage (%) at each risk target α  (POPE, LLaVA-1.5-7B, "
      f"20 three-way disjoint splits, mean±std)\n")
print(f"{'score':26s}"+"".join(f"α={a:.0%}".rjust(14) for a in ALPHAS))
for name in SC:
    row=name.ljust(26)
    for a in ALPHAS:
        v=np.array(res[name][a])*100; row+=f"{v.mean():5.1f}±{v.std():4.1f}".rjust(14)
    print(row)

print(f"\nMean realised risk (%) / exceedance fraction on test fold / on all {n} items"
      f"   [delta={DELTA:.2f}]\n")
print(f"{'score':26s}"+"".join(f"α={a:.0%}".rjust(22) for a in ALPHAS))
for name in SC:
    row=name.ljust(26)
    for a in ALPHAS:
        rr=np.array(rsk[name][a]);ra=np.array(rska[name][a])
        row+=f"{rr.mean()*100:5.1f} {(rr>a+1e-12).mean():.2f}/{(ra>a+1e-12).mean():.2f}".rjust(22)
    print(row)
print("  exc(te)=test fold (n_te=%d, unbiased/noisy)   exc(all)=all %d items (low-noise proxy)"
      % (n-2*(n//3), n))

# plot
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11})
COL={"Confidence only":"#9aa3b2","+ CLIP grounding":"#e0a458","+ OWLv2 grounding (ours)":"#4b57c8"}
fig,ax=plt.subplots(figsize=(7,5),dpi=150)
for name in SC:
    m=[np.mean(res[name][a])*100 for a in ALPHAS]; sd=[np.std(res[name][a])*100 for a in ALPHAS]
    ax.errorbar([a*100 for a in ALPHAS],m,yerr=sd,marker="o",lw=2.4 if "ours" in name else 1.8,
                capsize=3,color=COL[name],label=name)
ax.set_xlabel("Risk target α (%)");ax.set_ylabel("Guaranteed coverage (%)")
ax.set_title("Coverage bought at each guarantee level (POPE)",fontweight="bold",loc="left")
ax.grid(alpha=.2);ax.legend(frameon=False,fontsize=10,loc="lower right")
for s in ["top","right"]:ax.spines[s].set_visible(False)
fig.tight_layout();fig.savefig(os.path.join(_here,"coverage_vs_alpha.png"),bbox_inches="tight",facecolor="white")
print("\nsaved coverage_vs_alpha.png")
