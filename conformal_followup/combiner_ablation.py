"""
Combiner ablation: does a higher-capacity model beat logistic regression as the
conformal nonconformity combiner? And does it stay VALID?

Key finding: high-capacity models (GBM/RF/MLP) look better only when the combiner
is trained on the SAME data used to calibrate the conformal threshold -- they
overfit, inflate calibration scores, and BREAK the error guarantee on test.
With a proper 3-way split (train combiner / calibrate tau / test, disjoint) all
combiners are valid and give ~equal efficiency, so logistic regression is the
right choice (equal efficiency, lowest variance, no overfit, interpretable).

Real POPE / LLaVA-1.5-7B, 6 features [conf, p_yes, CLIP, CLIP_agree, OWL, OWL_agree].
"""
import json, os, numpy as np, warnings; warnings.filterwarnings("ignore")
from scipy.stats import beta
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
_here=os.path.dirname(os.path.abspath(__file__))
r=json.load(open(os.path.join(_here,"raw_scores.json")));o=json.load(open(os.path.join(_here,"owlv2_scores.json")))
p=np.array(r["p_yes"]);a=np.array(r["answer"]);c=np.array(r["correct"]);det=np.array(o["ground_det"]);clip=np.array(r["ground"])
n=len(c);conf=np.abs(p-.5)*2;mm=lambda x:(x-x.min())/(x.max()-x.min()+1e-9)
dn=mm(det);cn=mm(clip);dag=np.where(a==1,dn,1-dn);cag=np.where(a==1,cn,1-cn)
X=np.column_stack([conf,p,cn,cag,dn,dag]); ALPHA,DELTA=.10,.10
def cpu(e,k,dl):return 1.0 if k==0 or e==k else float(beta.ppf(1-dl,e+1,k-e))
def cov_err(s_cal,c_cal,s_te,c_te):
    tau,b=None,-1
    for t in np.unique(s_cal):
        m=s_cal>=t;k=int(m.sum())
        if k and cpu(int((1-c_cal[m]).sum()),k,DELTA)<=ALPHA and k>b:b,tau=k,float(t)
    if tau is None: return 0,0
    m=s_te>=tau;k=int(m.sum());return k/len(s_te),((1-c_te[m]).mean() if k else 0)
mk={"Logistic":lambda:LogisticRegression(max_iter=1000),
    "GradBoost":lambda:GradientBoostingClassifier(n_estimators=100,max_depth=3),
    "RandForest":lambda:RandomForestClassifier(n_estimators=200,max_depth=5),
    "MLP(64,32)":lambda:MLPClassifier(hidden_layer_sizes=(64,32),max_iter=800,early_stopping=True)}
for proto in ["I: reuse (train==calibrate)","II: 3-way split (disjoint)"]:
    rng=np.random.default_rng(0);R={k:{"c":[],"e":[]} for k in mk}
    for _ in range(30):
        idx=rng.permutation(n)
        if proto.startswith("I:"): h=n//2;tr=cal=idx[:h];te=idx[h:]
        else: t=n//3;tr=idx[:t];cal=idx[t:2*t];te=idx[2*t:]
        for name,f in mk.items():
            clf=f().fit(X[tr],c[tr]);pc=clf.predict_proba(X)[:,1]
            cv,er=cov_err(pc[cal],c[cal],pc[te],c[te]);R[name]["c"].append(cv);R[name]["e"].append(er)
    print(f"\n=== Protocol {proto}  (target error {ALPHA:.0%}) ===")
    print(f"{'combiner':12s}{'cov@10%':>11s}{'test err':>11s}{'valid?':>10s}")
    for k in mk:
        cv=np.array(R[k]['c'])*100;er=np.array(R[k]['e'])*100
        print(f"{k:12s}{cv.mean():7.1f}%   {er.mean():6.1f}%   {'OK' if er.mean()<=ALPHA*100+1 else 'VIOLATED':>9s}")
