"""
Comparison figure: our conformal + structured grounding vs other methods.
Panel A: risk-coverage curves (real, POPE/LLaVA-1.5-7B, 20 splits) for confidence
         vs +CLIP vs +OWLv2 structured grounding.
Panel B: capability matrix across recent hallucination methods (factual, sourced
         from each paper) -- the honest positioning vs full-coverage mitigation
         methods that our selective+guaranteed approach composes with.
"""
import json, os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.linear_model import LogisticRegression

_here=os.path.dirname(os.path.abspath(__file__))
r=json.load(open(os.path.join(_here,"raw_scores.json")))
o=json.load(open(os.path.join(_here,"owlv2_scores.json")))
p_yes=np.array(r["p_yes"]);answer=np.array(r["answer"]);correct=np.array(r["correct"])
det=np.array(o["ground_det"]);n=len(correct);conf=np.abs(p_yes-.5)*2
mm=lambda x:(x-x.min())/(x.max()-x.min())
detn=mm(det);det_ag=np.where(answer==1,detn,1-detn);clipn=mm(np.array(r["ground"]))
clip_ag=np.where(answer==1,clipn,1-clipn)
SC={"Confidence only":np.column_stack([conf,p_yes]),
    "+ CLIP grounding":np.column_stack([conf,p_yes,clipn,clip_ag]),
    "+ OWLv2 grounding (ours)":np.column_stack([conf,p_yes,detn,det_ag])}
GRID=np.linspace(.02,1,99)
def rc(s,c):
    c=c[np.argsort(-s)];cov=np.arange(1,len(c)+1)/len(c)
    return np.interp(GRID,cov,np.cumsum(1-c)/np.arange(1,len(c)+1))
rng=np.random.default_rng(0);curves={k:[] for k in SC};cov10={k:[] for k in SC}
for _ in range(20):
    idx=rng.permutation(n);h=n//2;cal,te=idx[:h],idx[h:]
    for k,X in SC.items():
        clf=LogisticRegression(max_iter=1000).fit(X[cal],correct[cal])
        pc=clf.predict_proba(X)[:,1];curve=rc(pc[te],correct[te]);curves[k].append(curve)
        b=GRID[curve<=.10];cov10[k].append(b.max() if len(b) else 0)

# ---------- figure ----------
INK="#1a1f2b";MUT="#5b6577";GRID_C="#e4e7ec"
COL={"Confidence only":"#9aa3b2","+ CLIP grounding":"#e0a458","+ OWLv2 grounding (ours)":"#4b57c8"}
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,"text.color":INK,
                     "axes.edgecolor":"#c9cfd8","axes.labelcolor":INK,
                     "xtick.color":MUT,"ytick.color":MUT})
fig=plt.figure(figsize=(14.5,6.2),dpi=150)
gs=GridSpec(1,2,width_ratios=[1.0,1.25],wspace=.32)

# Panel A
axA=fig.add_subplot(gs[0])
for k in SC:
    A=np.array(curves[k]);m=A.mean(0);sd=A.std(0)
    axA.fill_between(GRID,m-sd,m+sd,color=COL[k],alpha=.13,lw=0)
    axA.plot(GRID,m,color=COL[k],lw=2.8 if "ours" in k else 2.0,
             label=f"{k}  (cov@10%={np.mean(cov10[k])*100:.0f}%)")
axA.axhline(.10,ls="--",lw=1.2,color="#c8384f")
axA.text(.03,.107,"target risk = 10%",color="#c8384f",fontsize=9.5)
axA.set_xlabel("Coverage — fraction of questions answered")
axA.set_ylabel("Risk — error rate among answered")
axA.set_title("A · Selective risk–coverage · POPE / LLaVA-1.5-7B",
              fontweight="bold",loc="left",fontsize=12,pad=12)
axA.set_xlim(0,1);axA.set_ylim(0,.30);axA.grid(alpha=.2)
axA.legend(loc="upper left",frameon=False,fontsize=9.7)
for s in ["top","right"]: axA.spines[s].set_visible(False)

# Panel B — capability matrix
axB=fig.add_subplot(gs[1]);axB.axis("off")
methods=["OPERA  (CVPR'24)","Attention Lens  (CVPR'25)","REVERSE  (NeurIPS'25)",
         "ConfLVLM  (EMNLP'25)","Ours  (CMVKG-conformal)"]
caps=["Training-\nfree","Visual/KG\ngrounding","Real-time\ncorrection",
      "Statistical\nguarantee","Selective\ncoverage"]
M=np.array([
 [1,0,1,0,0],      # OPERA
 [1,0.5,1,0,0],    # Attention Lens
 [0,0.5,1,0,0.5],  # REVERSE (needs finetuning; rejection sampling ~ partial selective)
 [1,0,0,1,1],      # ConfLVLM
 [1,1,1,1,1],      # Ours
])
nR,nC=M.shape
for i in range(nR):
    for j in range(nC):
        v=M[i,j]
        c="#2f8f6b" if v==1 else ("#d9a441" if v==0.5 else "#dfe3e9")
        sym="✓" if v==1 else ("~" if v==0.5 else "–")
        axB.add_patch(plt.Rectangle((j+.03,nR-1-i+.03),.94,.94,color=c,ec="white",lw=2))
        axB.text(j+.5,nR-1-i+.5,sym,ha="center",va="center",
                 color="white" if v else "#9aa3b2",fontsize=15,fontweight="bold")
for j,cap in enumerate(caps):
    axB.text(j+.5,nR+.18,cap,ha="center",va="bottom",fontsize=8.6,color=INK,
             fontweight="bold",linespacing=.95)
for i,mth in enumerate(methods):
    fw="bold" if "Ours" in mth else "normal"
    axB.text(-.2,nR-1-i+.5,mth,ha="right",va="center",fontsize=9.3,color=INK,fontweight=fw)
axB.set_xlim(-3.4,nC+.1);axB.set_ylim(-0.7,nR+1.15)
axB.set_title("B · Capability vs recent methods",fontweight="bold",loc="left",
              fontsize=12,pad=12,x=0)
axB.text(-3.4,-0.55,"✓ full    ~ partial    – none        Mitigation methods "
         "(OPERA / Attention Lens / REVERSE) operate only at full\ncoverage; ours adds a "
         "distribution-free guarantee + coverage control and composes on top of them.",
         fontsize=8.2,color=MUT,va="top",linespacing=1.3)

fig.tight_layout()
out=os.path.join(_here,"comparison_figure.png");fig.savefig(out,bbox_inches="tight",facecolor="white")
print("saved",out)
