"""
Master figure. A: risk-coverage on POPE-adversarial, real ConfLVLM-faithful
(confidence+self-consistency) baseline vs ours (+grounding). B: grounding gain
(Δ coverage @10% risk) across backbones and the VCD composition.
"""
import json, os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import beta
from sklearn.linear_model import LogisticRegression
_here=os.path.dirname(os.path.abspath(__file__))
INK="#1a1f2b";MUT="#5b6577";IND="#4b57c8";GREY="#9aa3b2";OK="#1f7d5c"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,"text.color":INK,
                     "axes.edgecolor":"#c9cfd8","xtick.color":MUT,"ytick.color":MUT,"axes.labelcolor":INK})
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

# ---- data ----
L=json.load(open(os.path.join(_here,"exp7_pope.json")))
lp=np.array(L["p_yes"]);lsc=np.array(L["sc_yesfrac"]);low=np.array(L["owl"]);lan=np.array(L["answer"]);lc=np.array(L["correct"])
conf=np.abs(lp-.5)*2;scc=np.abs(lsc-.5)*2
g=low.copy();g[g<0]=np.median(g[g>=0]);mm=lambda x:(x-x.min())/(x.max()-x.min()+1e-9);gn=mm(g);gag=np.where(lan==1,gn,1-gn)
BASE=np.column_stack([lp,conf,lsc,scc]); OURS=np.column_stack([lp,conf,lsc,scc,gn,gag])
n=len(lc);rng=np.random.default_rng(0)
cb=[];co=[]
for _ in range(20):
    idx=rng.permutation(n);h=n//2;cal,te=idx[:h],idx[h:]
    for X,acc in [(BASE,cb),(OURS,co)]:
        clf=LogisticRegression(max_iter=1000).fit(X[cal],lc[cal]);pc=clf.predict_proba(X)[:,1]
        acc.append((rc(pc[te],lc[te]),cov(pc[cal],lc[cal],pc[te],lc[te])))
cbr=np.array([x[0] for x in cb]);cor=np.array([x[0] for x in co])
cbc=np.array([x[1] for x in cb])*100;coc=np.array([x[1] for x in co])*100

fig=plt.figure(figsize=(13.5,5.4),dpi=150)
gs=GridSpec(1,2,width_ratios=[1.1,1.0],wspace=.26)
# Panel A
axA=fig.add_subplot(gs[0])
for arr,col,lab,cc in [(cbr,GREY,"ConfLVLM-faithful (confidence + self-consistency)",cbc),
                       (cor,IND,"Ours (+ structured grounding)",coc)]:
    m=arr.mean(0);sd=arr.std(0)
    axA.fill_between(GRID,m-sd,m+sd,color=col,alpha=.14,lw=0)
    axA.plot(GRID,m,color=col,lw=2.7,label=f"{lab}\n   cov@10%={cc.mean():.0f}%")
axA.axhline(.10,ls="--",lw=1.2,color="#c8384f");axA.text(.03,.107,"target risk = 10%",color="#c8384f",fontsize=9.5)
axA.set_xlabel("Coverage — fraction answered");axA.set_ylabel("Risk — error among answered")
axA.set_title("A · vs real ConfLVLM baseline (POPE-adversarial, LLaVA-1.5-7B)",fontweight="bold",loc="left",fontsize=11.5,pad=10)
axA.set_xlim(0,1);axA.set_ylim(0,.30);axA.grid(alpha=.2);axA.legend(loc="upper left",frameon=False,fontsize=9.3)
for s in ["top","right"]:axA.spines[s].set_visible(False)
# Panel B — grounding gain across settings
axB=fig.add_subplot(gs[1])
labels=["LLaVA-1.5\n(base 0.83)","Qwen2-VL\n(base 0.87)","VCD decoder\n(composition)"]
gains=[10.2,1.9,16.5];errs=[8.6,3.2,17.5]
x=np.arange(len(labels));bars=axB.bar(x,gains,yerr=errs,capsize=5,width=.6,
    color=[IND,"#8b95f0","#6b77dd"],edgecolor="white",lw=1.5)
axB.axhline(0,color="#c9cfd8",lw=1)
for i,g_ in enumerate(gains): axB.text(i,g_+errs[i]+.6,f"+{g_:.1f}pp",ha="center",fontsize=10,fontweight="bold",color=INK)
axB.set_xticks(x);axB.set_xticklabels(labels,fontsize=9.5)
axB.set_ylabel("Δ coverage @10% risk from grounding (pp)")
axB.set_title("B · Grounding gain across backbones & composition",fontweight="bold",loc="left",fontsize=11.5,pad=10)
axB.set_ylim(-2,40);axB.grid(alpha=.2,axis="y")
for s in ["top","right"]:axB.spines[s].set_visible(False)
axB.text(0,34,"benefit scales with how much\nthe base model hallucinates",fontsize=8.6,color=MUT,style="italic",linespacing=1.3)
fig.tight_layout()
out=os.path.join(_here,"master_figure.png");fig.savefig(out,bbox_inches="tight",facecolor="white")
print("saved",out)
