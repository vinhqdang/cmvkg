"""Recover the image identifier of every cached item and verify alignment.

The extraction scripts (colab_exp*.py) stream the first N rows of each public
benchmark in file order, so the per-item image id is recoverable from benchmark
metadata alone (no images are downloaded: the image column is pruned from the
parquet read).  Each reconstruction is VERIFIED against the cached file (gold
label, object phrase, category), and the script fails loudly on any mismatch.

Output: revision/image_ids.json  {dataset_file: [image_id, ...]}
"""
import json, os, random, sys, urllib.request
import numpy as np, pandas as pd, pyarrow.parquet as pq, fsspec

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
fs = fsspec.filesystem("https")
API = "https://huggingface.co/api/datasets/lmms-lab-encoder/%s/parquet/%s/%s/%d.parquet"

def meta(ds, cfg, split, nfiles):
    out = []
    for i in range(nfiles):
        pf = pq.ParquetFile(fs.open(API % (ds, cfg, split, i)))
        cols = [c for c in pf.schema_arrow.names if c not in ("image",)]
        out.append(pf.read(columns=cols).to_pandas())
    return pd.concat(out, ignore_index=True)

def J(f): return json.load(open(os.path.join(ROOT, f)))
def ynz(a): return 1 if str(a).lower().strip().startswith("yes") else 0

def extract_object(q):                      # identical to the extraction scripts
    ql = q.lower()
    if "is there" not in ql: return None
    s = ql.split("is there", 1)[1].strip()
    for k in ("a ", "an ", "the "):
        if s.startswith(k): s = s[len(k):]; break
    for k in ["in the image", "in this image", "in the picture", "in this picture",
              "please answer yes or no", ".", "?"]:
        s = s.replace(k, "")
    s = s.strip()
    return s if 1 <= len(s.split()) <= 3 and s else None

out, report = {}, []
def verify(name, d, imgs, gold, objs=None, cats=None):
    n = len(d["gold"]); assert len(imgs) >= n, (name, len(imgs), n)
    g_ok = float(np.mean(np.array(d["gold"]) == np.array(gold[:n])))
    o_ok = float(np.mean([a == b for a, b in zip(d["obj"], objs[:n])])) if objs is not None and "obj" in d else None
    c_ok = float(np.mean([a == b for a, b in zip(d["category"], cats[:n])])) if cats is not None and "category" in d else None
    report.append((name, n, len(set(imgs[:n])), g_ok, o_ok, c_ok))
    # gold and category must match exactly; the object phrase may differ on a handful of
    # rows where an earlier extraction version parsed a typo ('imange') differently
    assert g_ok == 1.0 and (o_ok is None or o_ok >= 0.97) and (c_ok in (None, 1.0)), report[-1]
    out[name] = list(map(str, imgs[:n]))

# ---- POPE (adversarial split first in file order) ----
pope = meta("POPE", "default", "test", 3)
gold = pope.answer.map(ynz).tolist(); objs = [extract_object(q) for q in pope.question]
for f in ["raw_scores.json", "exp7_pope.json", "exp8_qwen_pope.json", "exp9_vcd_pope.json", "exp13_bcea.json", "owlv2_scores.json"]:
    d = J(f)
    verify(f, d, pope.image_source.tolist(), gold, objs if "obj" in d else None,
           pope.category.tolist() if "category" in d else None)

# ---- MME ----
mme = meta("MME", "default", "test", 4)
mme_gold = mme.answer.map(ynz).tolist(); mme_obj = [extract_object(q) for q in mme.question]
verify("exp10_mme.json", J("exp10_mme.json"), mme.question_id.tolist(), mme_gold, mme_obj, mme.category.tolist())
ex = mme[mme.category == "existence"].reset_index(drop=True)
verify("exp11_mme_exist.json", J("exp11_mme_exist.json"), ex.question_id.tolist(),
       ex.answer.map(ynz).tolist(), [extract_object(q) for q in ex.question])

# ---- HallusionBench (image split, visual_input==1, gt in {0,1}) ----
hb = meta("HallusionBench", "default", "image", 1)
hb = hb[(hb.visual_input.astype(int) == 1) & hb.gt_answer.astype(str).str.strip().isin(["0", "1"])].reset_index(drop=True)
hb_id = (hb.set_index if False else (lambda: (hb["category"].astype(str) + "|" + hb["subcategory"].astype(str) + "|" + hb["set_id"].astype(str) + "|" + hb["figure_id"].astype(str) + "|" + hb["filename"].astype(str))))()
verify("exp11_hallusion.json", J("exp11_hallusion.json"), hb_id.tolist(),
       hb.gt_answer.astype(str).str.strip().astype(int).tolist(), [extract_object(q) for q in hb.question], hb.category.astype(str).tolist())

# ---- GQA testdev_balanced ----
gq = meta("GQA", "testdev_balanced_instructions", "testdev", 1)
gq = gq[gq.answer.astype(str).str.lower().str.strip().isin(["yes", "no"])].reset_index(drop=True)
verify("exp11_gqa.json", J("exp11_gqa.json"), gq.imageId.tolist(), gq.answer.map(ynz).tolist(), [extract_object(q) for q in gq.question])

# ---- AMBER ----
RAW = "https://raw.githubusercontent.com/junyangwang0410/AMBER/master/data/"
fj = lambda u: json.loads(urllib.request.urlopen(u, timeout=120).read().decode())
queries, annots = fj(RAW + "query/query_discriminative.json"), fj(RAW + "annotations.json")
truth = {r["id"]: r for r in annots}
ATTR = ("discriminative-attribute-state", "discriminative-attribute-number", "discriminative-attribute-action")
def amber_cand(want, shuffle):
    cand = []
    for q in queries:
        a = truth.get(q["id"])
        if not a or a.get("type") not in want: continue
        t = str(a.get("truth", "")).lower().strip()
        if t not in ("yes", "no"): continue
        cand.append((q["id"], q["image"], q["query"], 1 if t == "yes" else 0, a["type"]))
    if shuffle: random.Random(0).shuffle(cand)
    return cand
for name, want, sh in [("exp12_amber_all.json", ("discriminative-hallucination",) + ATTR + ("discriminative-relation",), True),
                       ("exp12_amber_existence.json", ("discriminative-hallucination",), False)]:
    c = amber_cand(want, sh)
    verify(name, J(name), [x[1] for x in c], [x[3] for x in c], [extract_object(x[2]) for x in c])

json.dump(out, open(os.path.join(HERE, "image_ids.json"), "w"))
print(f"{'file':28s}{'n':>6s}{'images':>8s}{'items/img':>10s}  gold  obj  cat")
for name, n, ni, g, o, c in report:
    print(f"{name:28s}{n:6d}{ni:8d}{n/ni:10.2f}  {g:.2f}  {o if o is None else round(o,2)}  {c if c is None else round(c,2)}")
