"""Qualitative figures under the revised protocol (image-disjoint fit and calibration folds; the displayed images
belong to neither). Fig. qualitative: accept / repair / abstain examples. Fig. failures: accepted answers that are
wrong. Images are real POPE (COCO val2014) images; the extra ones are read from the public parquet row groups."""
import io, json, os, numpy as np, pandas as pd, pyarrow.parquet as pq, fsspec
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image
import ccrc_rev as R
H = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(H, "..", "manuscript_revised")
SP = "/tmp/claude-0/-home-user-cmvkg/e237f2e6-6f4d-5454-9e18-e0533bf20b86/scratchpad/pope_meta_0.pkl"
meta_q = pd.read_pickle(SP)
cached = json.load(open(os.path.join(H, "..", "qual_imgs", "meta.json")))
S = R.load_settings()["POPE-1500 LLaVA"]; M = R.Mode(); X = S.features("impute")
shown_accept, shown_repair, shown_abst = [664, 538], [518, 1328], [1399, 1463]
D = shown_accept + shown_repair + shown_abst
rng = np.random.default_rng(0)
imgs = np.unique(S.img); disp = set(S.img[D]); rest = np.array([u for u in imgs if u not in disp]); rng.shuffle(rest)
t = len(imgs) // 3
fit = np.flatnonzero(np.isin(S.img, rest[:t])); cal = np.flatnonzero(np.isin(S.img, rest[t:2 * t]))
w = R.fit_logistic(X[fit], S.ok[fit]); s = R.predict(w, X)
run = R.Runner(S, M, reps=1); run.X = X; ref = run._oof(fit)
thr = np.quantile(ref, 1 - R._grid_lams(len(cal), R.k_start(.1, .1, 1), 40, 1.5))
rg = R.Region(S, "margin", 0.10, fit, ref, M, S.m[cal]); j = R.fst_region(S, s, cal, thr, .1, .1, rg); T, G = float(thr[j]), float(rg.gv)

def get_image(i):
    if str(i) in cached: return Image.open(os.path.join(H, "..", cached[str(i)]["file"]))
    cache = os.path.join(H, "qual_cache"); os.makedirs(cache, exist_ok=True); f = os.path.join(cache, f"q{i}.jpg")
    if not os.path.exists(f):
        pf = pq.ParquetFile(fsspec.filesystem("https").open("https://huggingface.co/api/datasets/lmms-lab-encoder/POPE/parquet/default/test/0.parquet"))
        g = i // 100; tb = pf.read_row_group(g, columns=["image"]).to_pylist()[i - g * 100]["image"]
        open(f, "wb").write(tb["bytes"])
    return Image.open(f)

def obj(i): return meta_q.question[i].lower().replace("is there a", "").replace("is there an", "").replace("in the image?", "").replace("in the imange?", "").strip()
INK, MUT, IND, OK, BAD = "#1a1f2b", "#5b6577", "#4b57c8", "#1f7d5c", "#c8384f"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "text.color": INK})
def action(i): return "ACCEPT" if s[i] >= T else ("REPAIR" if S.m[i] >= G else "ABSTAIN")
def panel(ax, i, colr, tag=None):
    im = get_image(i); ax.imshow(im); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_edgecolor(colr); sp.set_linewidth(2.4)
    wrong = S.ok[i] == 0; det_ok = S.okr[i] == 1; a = action(i); yn = lambda v: "yes" if v else "no"
    lines = [(f'Q: "Is there a {obj(i)}?"   truth: {yn(S.y[i])}', INK, "bold"),
             (f'LLaVA: "{yn(S.a[i])}"  (p={S.p[i]:.2f})   ' + ("wrong" if wrong else "correct"), BAD if wrong else OK, "normal"),
             (f'detector: "{yn(S.b[i])}"  (conf={S.o[i]:.2f}, margin={S.m[i]:.2f})', OK if det_ok else BAD, "normal"),
             (f"score s={s[i]:.2f}  ->  {a}" + ("  (error fixed)" if a == "REPAIR" and wrong and det_ok else "") + ("  (answer wrong)" if a == "ACCEPT" and wrong else ""), colr, "bold")]
    y = -0.06
    for txt, c, wgt in lines:
        ax.text(0.0, y, txt, transform=ax.transAxes, ha="left", va="top", fontsize=8.1, color=c, fontweight=wgt); y -= 0.075

test = np.flatnonzero(np.isin(S.img, rest[2 * t:]))
acc_wrong = [i for i in test if s[i] >= T and S.ok[i] == 0]
fail = [i for i in acc_wrong if S.y[i] == 0][:2] + [i for i in acc_wrong if S.y[i] == 1][:2]
fig, axes = plt.subplots(2, 3, figsize=(10.6, 7.4), dpi=200)
for col, (lab, colr, ids) in enumerate([("ACCEPT", OK, shown_accept), ("REPAIR", IND, shown_repair), ("ABSTAIN", MUT, shown_abst)]):
    for row, i in enumerate(ids):
        assert action(ids[row]) == lab, (lab, i, action(i))
        panel(axes[row, col], i, colr)
    axes[0, col].set_title(lab, color=colr, fontweight="bold", fontsize=13, pad=8)
fig.text(0.012, 0.012, f"One split with image-disjoint fit and calibration folds (shown images in neither): accept if s $\\geq$ {T:.2f}, repair if margin $\\geq$ {G:.2f},\n"
         r"$\alpha=\delta=0.10$. LLaVA-1.5 on POPE, detector OWLv2.", fontsize=8.2, color=MUT, ha="left")
fig.tight_layout(rect=[0, 0.05, 1, 1], h_pad=4.2, w_pad=1.6); fig.savefig(os.path.join(OUT, "fig_qualitative.png")); plt.close(fig)

fig, axes = plt.subplots(1, 4, figsize=(13.2, 4.3), dpi=200)
for ax, i in zip(axes, fail):
    assert action(i) == "ACCEPT" and S.ok[i] == 0, (i, action(i))
    panel(ax, i, BAD)
fig.text(0.012, 0.01, "Accepted answers that are wrong under the same policy: confident false positives (left) and confident misses on which the detector is also weak (right).", fontsize=8.4, color=MUT, ha="left")
fig.tight_layout(rect=[0, 0.05, 1, 1], w_pad=1.2); fig.savefig(os.path.join(OUT, "fig_failures.png")); plt.close(fig)
print("T", round(T, 3), "G", round(G, 3), [(i, action(i)) for i in D], "fail", fail)
