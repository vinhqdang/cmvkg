"""Checks that the Newton logistic in ccrc_rev matches sklearn's default
LogisticRegression (L2, C=1) on the actual features, so that the fast fitter can
be used inside the image-level bootstrap."""
import numpy as np, warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
import ccrc_rev as R
S = R.load_settings()
worst = 0
for name, st in S.items():
    for missing in ("drop", "impute"):
        X = st.features(missing, legacy_norm=(missing == "drop"))
        keep = st.g if missing == "drop" else np.ones(st.n, bool)
        Xk, yk = X[keep], st.ok[keep]
        sk = LogisticRegression(max_iter=100000, tol=1e-12).fit(Xk, yk).predict_proba(Xk)[:, 1]
        mine = R.predict(R.fit_logistic(Xk, yk), Xk)
        worst = max(worst, float(np.max(np.abs(sk - mine))))
        print(f"{name:22s}{missing:8s} max|dp|={np.max(np.abs(sk-mine)):.2e}")
print("worst", worst); assert worst < 1e-5
