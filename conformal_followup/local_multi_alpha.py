"""
Multi-alpha results: how much coverage each score buys at several risk targets,
under RCPS (Clopper-Pearson). Real POPE / LLaVA-1.5-7B, 20 splits. CPU-only.
Also writes coverage_vs_alpha.png.
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
def rcps(sc,cc,st,ct,alpha):
    tau,best=None,-1
    for t in np.unique(sc):
        m=sc>=t;k=int(m.sum())
        if k==0: continue
        e=int((1-cc[m]).sum())
        if cp_upper(e,k,DELTA)<=alpha and k>best: best,tau=k,float(t)
    if tau is None: return 0.0
    m=st>=tau;return int(m.sum())/len(st)

rng=np.random.default_rng(0)
res={k:{a:[] for a in ALPHAS} for k in SC}
for _ in range(20):
    idx=rng.permutation(n);h=n//2;cal,te=idx[:h],idx[h:]
    for name,X in SC.items():
        clf=LogisticRegression(max_iter=1000).fit(X[cal],correct[cal])
        pc=clf.predict_proba(X)[:,1]
        for a in ALPHAS: res[name][a].append(rcps(pc[cal],correct[cal],pc[te],correct[te],a))

print(f"Coverage (%) at each risk target α  (POPE, LLaVA-1.5-7B, 20 splits, mean±std)\n")
print(f"{'score':26s}"+"".join(f"α={a:.0%}".rjust(14) for a in ALPHAS))
for name in SC:
    row=name.ljust(26)
    for a in ALPHAS:
        v=np.array(res[name][a])*100; row+=f"{v.mean():5.1f}±{v.std():4.1f}".rjust(14)
    print(row)

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
