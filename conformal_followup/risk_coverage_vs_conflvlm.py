"""
Head-to-head risk-coverage comparison vs a ConfLVLM-style baseline.

ConfLVLM-style baseline: conformal selective prediction using ONLY model-internal
heuristic uncertainty (no external grounding) -- steelmanned as a learned combiner
of internal features [confidence-margin, p_yes]. This is the faithful analogue of
ConfLVLM's heuristic-uncertainty scoring in our single-pass POPE setting.

Ours: the same conformal procedure but with a STRUCTURED grounding signal added
(OWLv2 detection agreement) -- the CMVKG-Guard-style contribution.

Output: mean risk-coverage curves (20 splits) + AURC + coverage@10%-risk, saved
to risk_coverage.png. Lower curve / larger area-below-target = better.
"""
import json, os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

_here = os.path.dirname(os.path.abspath(__file__))
r = json.load(open(os.path.join(_here, "raw_scores.json")))
o = json.load(open(os.path.join(_here, "owlv2_scores.json")))
assert o["gold"] == r["gold"]
p_yes=np.array(r["p_yes"]); answer=np.array(r["answer"]); correct=np.array(r["correct"])
det=np.array(o["ground_det"]); n=len(correct)
conf=np.abs(p_yes-0.5)*2
def mm(x):
    d=x.max()-x.min(); return (x-x.min())/d if d>0 else x*0
detn=mm(det); det_agree=np.where(answer==1, detn, 1-detn)

INTERNAL = np.column_stack([conf, p_yes])                          # ConfLVLM-style
OURS     = np.column_stack([conf, p_yes, detn, det_agree])         # + structured grounding

GRID = np.linspace(0.02, 1.0, 99)          # coverage levels
def rc_curve(score, corr):
    order=np.argsort(-score); c=corr[order]
    cov=np.arange(1,len(c)+1)/len(c)
    risk=np.cumsum(1-c)/np.arange(1,len(c)+1)
    return np.interp(GRID, cov, risk)

rng=np.random.default_rng(0)
curves={"ConfLVLM-style (internal uncertainty)":[], "Ours (+ structured grounding)":[]}
aurcs={k:[] for k in curves}; cov10={k:[] for k in curves}
for _ in range(20):
    idx=rng.permutation(n); h=n//2; cal,te=idx[:h],idx[h:]
    for name,X in [("ConfLVLM-style (internal uncertainty)",INTERNAL),
                   ("Ours (+ structured grounding)",OURS)]:
        clf=LogisticRegression(max_iter=1000).fit(X[cal],correct[cal])
        pc=clf.predict_proba(X)[:,1]
        rc=rc_curve(pc[te],correct[te]); curves[name].append(rc)
        aurcs[name].append(rc.mean())
        # coverage where risk first stays <= 0.10 (from high-score side)
        below=GRID[rc<=0.10]; cov10[name].append(below.max() if len(below) else 0.0)

# ---- plot ----
plt.rcParams.update({"font.size":11,"font.family":"DejaVu Sans"})
fig,ax=plt.subplots(figsize=(7.4,5.0),dpi=150)
COL={"ConfLVLM-style (internal uncertainty)":"#8a94a6","Ours (+ structured grounding)":"#4b57c8"}
for name in curves:
    A=np.array(curves[name]); m=A.mean(0); sd=A.std(0)
    ax.fill_between(GRID, m-sd, m+sd, color=COL[name], alpha=.15, lw=0)
    ax.plot(GRID, m, color=COL[name], lw=2.4,
            label=f"{name}\n   AURC={np.mean(aurcs[name]):.4f} · cov@10%={np.mean(cov10[name])*100:.1f}%")
ax.axhline(0.10, ls="--", lw=1.2, color="#c8384f")
ax.text(0.015,0.104,"target risk α = 10%",color="#c8384f",fontsize=9.5,va="bottom")
ax.set_xlabel("Coverage  (fraction of questions answered)")
ax.set_ylabel("Risk  (error rate among answered)")
ax.set_title("Risk–coverage on POPE (LLaVA-1.5-7B): structured grounding\ndominates the internal-uncertainty baseline",
             fontsize=12.5, loc="left")
ax.set_xlim(0,1); ax.set_ylim(0,0.30)
ax.grid(alpha=.18); ax.legend(loc="upper left", fontsize=9.5, frameon=False, bbox_to_anchor=(0,0.95))
for s in ["top","right"]: ax.spines[s].set_visible(False)
fig.tight_layout()
out=os.path.join(_here,"risk_coverage.png"); fig.savefig(out,bbox_inches="tight")
print("saved",out)
# dominance summary
A=np.array(curves["Ours (+ structured grounding)"]).mean(0)
B=np.array(curves["ConfLVLM-style (internal uncertainty)"]).mean(0)
print(f"Ours risk <= baseline at {100*np.mean(A<=B+1e-9):.0f}% of coverage levels")
print(f"AURC  baseline={np.mean(aurcs['ConfLVLM-style (internal uncertainty)']):.4f}  "
      f"ours={np.mean(aurcs['Ours (+ structured grounding)']):.4f}")
print(f"cov@10% baseline={np.mean(cov10['ConfLVLM-style (internal uncertainty)'])*100:.1f}%  "
      f"ours={np.mean(cov10['Ours (+ structured grounding)'])*100:.1f}%")
