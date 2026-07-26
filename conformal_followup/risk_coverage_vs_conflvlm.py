"""
Head-to-head risk-coverage comparison vs a ConfLVLM-style baseline.

ConfLVLM-style baseline: conformal selective prediction using ONLY model-internal
heuristic uncertainty (no external grounding) -- steelmanned as a learned combiner
of internal features [confidence-margin, p_yes]. This is the faithful analogue of
ConfLVLM's heuristic-uncertainty scoring in our single-pass POPE setting.

Ours: the same conformal procedure but with a STRUCTURED grounding signal added
(OWLv2 detection agreement) -- the CMVKG-Guard-style contribution.

Protocol: 20 THREE-WAY DISJOINT splits -- the combiner is fit on a train fold, the
conformal threshold is calibrated on a second fold, and both the risk-coverage curve
and the realised risk are measured on a third, held-out fold.

Output: mean risk-coverage curves (20 splits) + AURC + coverage@10%-risk, saved
to risk_coverage.png. Lower curve / larger area-below-target = better.

NOTE on cov@10%: this is the ORACLE coverage read off the test-fold risk-coverage
curve (the largest coverage level whose empirical test risk is <= 10%). It is NOT the
conformally calibrated coverage reported in the tables, which pays for a
Clopper-Pearson bound at delta=0.10 on a finite calibration fold; the oracle number
is therefore systematically higher. Both are printed below, kept clearly separate.

Audit statistic: the guarantee is Pr[Risk <= alpha] >= 1-delta, so for the calibrated
arm we report the FRACTION of splits with realised risk > alpha, vs delta=0.10.
exc(te)=held-out test fold (unbiased, noisy); exc(all)=selected policy scored on all
n items (low-noise proxy, slightly optimistic).
"""
import json, os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import beta
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

ALPHA, DELTA = 0.10, 0.10
def cp_upper(e,k,d):
    if k==0: return 1.0
    return 1.0 if e==k else float(beta.ppf(1-d,e+1,k-e))
def rcps(sc,cc,st,ct,s_all,c_all,alpha=ALPHA,dl=DELTA):
    """Calibrated arm: (coverage on test, realised risk on test, realised risk on all)."""
    tau,best=None,-1
    for t in np.unique(sc):
        m=sc>=t; k=int(m.sum())
        if k and cp_upper(int((1-cc[m]).sum()),k,dl)<=alpha and k>best: best,tau=k,float(t)
    if tau is None: return 0.0,0.0,0.0
    m=st>=tau; k=int(m.sum())
    cv,er=(k/len(st),float((1-ct[m]).mean())) if k else (0.0,0.0)
    ma=s_all>=tau; ka=int(ma.sum())
    return cv,er,(float((1-c_all[ma]).mean()) if ka else 0.0)

rng=np.random.default_rng(0)
curves={"ConfLVLM-style (internal uncertainty)":[], "Ours (+ structured grounding)":[]}
aurcs={k:[] for k in curves}; cov10={k:[] for k in curves}
covC={k:[] for k in curves}; rskC={k:[] for k in curves}; rskA={k:[] for k in curves}
for _ in range(20):
    idx=rng.permutation(n); t3=n//3; tr,cal,te=idx[:t3],idx[t3:2*t3],idx[2*t3:]
    for name,X in [("ConfLVLM-style (internal uncertainty)",INTERNAL),
                   ("Ours (+ structured grounding)",OURS)]:
        clf=LogisticRegression(max_iter=1000).fit(X[tr],correct[tr])
        pc=clf.predict_proba(X)[:,1]
        rc=rc_curve(pc[te],correct[te]); curves[name].append(rc)
        aurcs[name].append(rc.mean())
        # oracle coverage where test-fold risk stays <= 0.10 (from high-score side)
        below=GRID[rc<=0.10]; cov10[name].append(below.max() if len(below) else 0.0)
        cv,er,era=rcps(pc[cal],correct[cal],pc[te],correct[te],pc,correct)
        covC[name].append(cv); rskC[name].append(er); rskA[name].append(era)

# ---- plot ----
plt.rcParams.update({"font.size":11,"font.family":"DejaVu Sans"})
fig,ax=plt.subplots(figsize=(7.4,5.0),dpi=150)
COL={"ConfLVLM-style (internal uncertainty)":"#8a94a6","Ours (+ structured grounding)":"#4b57c8"}
for name in curves:
    A=np.array(curves[name]); m=A.mean(0); sd=A.std(0)
    ax.fill_between(GRID, m-sd, m+sd, color=COL[name], alpha=.15, lw=0)
    ax.plot(GRID, m, color=COL[name], lw=2.4,
            label=f"{name}\n   AURC={np.mean(aurcs[name]):.4f} · oracle cov@10%={np.mean(cov10[name])*100:.1f}%"
                  f"\n   calibrated cov@10%={np.mean(covC[name])*100:.1f}%")
ax.axhline(0.10, ls="--", lw=1.2, color="#c8384f")
ax.text(0.015,0.104,"target risk α = 10%",color="#c8384f",fontsize=9.5,va="bottom")
ax.set_xlabel("Coverage  (fraction of questions answered)")
ax.set_ylabel("Risk  (error rate among answered)")
ax.set_title("Risk–coverage on POPE (LLaVA-1.5-7B): structured grounding\ndominates the internal-uncertainty baseline"
             "\n(20 three-way disjoint splits: fit / calibrate / test)",
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
print(f"ORACLE cov@10% (test-fold curve, NOT calibrated) baseline="
      f"{np.mean(cov10['ConfLVLM-style (internal uncertainty)'])*100:.1f}%  "
      f"ours={np.mean(cov10['Ours (+ structured grounding)'])*100:.1f}%")
print(f"\nCalibrated RCPS arm (tau from cal fold, CP bound at delta={DELTA:.2f}), "
      f"20 three-way splits:")
print(f"{'arm':40s}{'cov@10%':>10s}{'risk@10%':>10s}{'exc(te)':>9s}{'exc(all)':>10s}")
for name in curves:
    cv=np.array(covC[name]);rr=np.array(rskC[name]);ra=np.array(rskA[name])
    print(f"{name:40s}{cv.mean()*100:8.1f}%{rr.mean()*100:9.1f}%"
          f"{(rr>ALPHA+1e-12).mean():8.2f} {(ra>ALPHA+1e-12).mean():9.2f}")
d_=(np.array(covC['Ours (+ structured grounding)'])
    -np.array(covC['ConfLVLM-style (internal uncertainty)']))*100
print(f"paired calibrated Delta cov@10% = {d_.mean():+.1f}+-{d_.std():.1f} pp")
print(f"Exceedance = fraction of 20 splits with realised risk > alpha={ALPHA:.2f} "
      f"(guarantee requires <= delta={DELTA:.2f});")
print(f"  exc(te)=test fold n_te={n-2*(n//3)} (unbiased/noisy)   "
      f"exc(all)=all {n} items (low-noise proxy)")
