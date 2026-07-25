"""
Qualitative figure: real POPE images with the actual channel-1 score, channel-2 evidence,
and the CCRC decision each one receives. One column per decision region.
Produces fig_qualitative.png.
"""
import json, os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

_here = os.path.dirname(os.path.abspath(__file__))
picks = json.load(open(os.path.join(_here, "qual_picks.json")))
meta = json.load(open(os.path.join(_here, "qual_imgs", "meta.json")))
INK, MUT, IND, OK, BAD = "#1a1f2b", "#5b6577", "#4b57c8", "#1f7d5c", "#c8384f"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "text.color": INK})

ORDER = [("accept", "ACCEPT", OK), ("repair", "REPAIR", IND), ("abstain", "ABSTAIN", MUT)]
NROW = 2                                        # examples shown per region
fig, axes = plt.subplots(NROW, 3, figsize=(10.6, 7.4), dpi=200)

for col, (key, label, colr) in enumerate(ORDER):
    entries = picks["picks"][key][:NROW]
    for row in range(NROW):
        ax = axes[row, col]
        e = entries[row]; m = meta[str(e["idx"])]
        img = Image.open(os.path.join(_here, m["file"]))
        ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_edgecolor(colr); sp.set_linewidth(2.4)
        obj = (m["question"].lower().replace("is there a", "")
               .replace("in the image?", "").strip())
        wrong = (e["ans"] != e["gold"])
        det_ok = (e["det_ans"] == e["gold"])
        # caption block under the image
        lines = [
            (f'Q: "Is there a {obj}?"   truth: {e["gold"]}', INK, "bold"),
            (f'LLaVA: "{e["ans"]}"  (p={e["p_yes"]:.2f})   '
             + ("wrong" if wrong else "correct"), BAD if wrong else OK, "normal"),
            (f'detector: "{e["det_ans"]}"  (conf={e["det"]:.2f}, margin={e["m"]:.2f})',
             OK if det_ok else BAD, "normal"),
            (f'channel-1 score s={e["s"]:.2f}  ->  {label}'
             + ("  → error fixed" if (key == "repair" and wrong and det_ok) else ""),
             colr, "bold"),
        ]
        y = -0.06
        for txt, c, w in lines:
            ax.text(0.0, y, txt, transform=ax.transAxes, ha="left", va="top",
                    fontsize=8.1, color=c, fontweight=w)
            y -= 0.075
        if row == 0:
            ax.set_title(f"{label}", color=colr, fontweight="bold", fontsize=13, pad=8)

fig.suptitle("Real POPE images and the decision CCRC assigns to each",
             fontsize=13, fontweight="bold", x=0.012, ha="left", y=0.995)
fig.text(0.012, 0.012,
         r"Thresholds shown are the calibrated $Q_s(1-\lambda)=%.2f$ and the fixed repair gate "
         r"$Q_m(1-q)=%.2f$ from one split. REPAIR cases are ones where LLaVA misses an object "
         r"that is genuinely present and the detector supplies it." % (picks["tau"], picks["tm"]),
         fontsize=8.2, color=MUT, ha="left")
fig.tight_layout(rect=[0, 0.035, 1, 0.972], h_pad=4.2, w_pad=1.6)
out = os.path.join(_here, "fig_qualitative.png")
fig.savefig(out, bbox_inches="tight", facecolor="white")
print("wrote", out)
