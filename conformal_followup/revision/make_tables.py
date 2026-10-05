"""Builds every LaTeX table of the revised manuscript from the results_*.json files, so that no number
in a table is typed by hand. Writes ../manuscript_revised/tables/*.tex."""
import json, os, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(H, "..", "manuscript_revised", "tables"); os.makedirs(OUT, exist_ok=True)
L = lambda f: json.load(open(os.path.join(H, f)))
NAMES = ["POPE-1500 LLaVA", "POPE-3000 LLaVA", "POPE-adv LLaVA", "POPE-adv Qwen2-VL", "POPE-adv LLaVA+VCD", "AMBER(d) LLaVA"]
SHORT = {"POPE-1500 LLaVA": "POPE-1500, LLaVA", "POPE-3000 LLaVA": "POPE-3000, LLaVA", "POPE-adv LLaVA": "POPE-adv, LLaVA", "POPE-adv Qwen2-VL": "POPE-adv, Qwen2-VL",
         "POPE-adv LLaVA+VCD": "POPE-adv, LLaVA+VCD", "AMBER(d) LLaVA": "AMBER(d), LLaVA"}
def w(name, body): open(os.path.join(OUT, name + ".tex"), "w").write(body)
p1 = lambda x: f"{x*100:.1f}"
sg = lambda x, nd=1: f"{x:+.{nd}f}".replace("-", "$-$")
def ab(x): return f"{x*100:.0f}"

main, boot, ladder, start = L("results_main.json"), L("results_bootstrap.json"), L("results_ladder.json"), L("results_start.json")
coh, sc, selfr, sym, sweep = L("results_cohort.json"), L("results_scores.json"), L("results_selfrepair.json"), L("results_symgate.json"), L("results_sweep.json")

# ---------------------------------------------------------------- cohort
rows = []
for n in NAMES:
    c = coh[n]; reason = {"POPE": f"{c['n_ungrounded']} unparsed object phrases", "AMBER": f"{c['n_ungrounded']} attribute/relation queries"}["AMBER" if n.startswith("AMBER") else "POPE"]
    rows.append(f"{SHORT[n]} & {c['n_items']} & {c['n_images']} & {c['items_per_image']:.1f} & {c['icc_model_error']:+.2f} & {c['n_grounded']} & {reason} & "
                f"{c['fold_images'][0]:.0f}/{c['fold_images'][1]:.0f}/{c['fold_images'][2]:.0f} & {c['mu']*100:.1f}\\\\".replace("+-", "$-$").replace("-0.", "$-$0."))
w("cohort", "\\begin{tabular}{lrrrrrlcc}\n\\toprule\nSetting & items & images & items/img & ICC & detector value & excluded from the grounded cohort & images/fold & $\\mu$ (\\%)\\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

# ---------------------------------------------------------------- main table (all-item cohort)
def cell(m, a, arm):
    x = main[f"{m}|all|{a}"][arm]; return f"{p1(x['cov'])} ({ab(x['abort'])})"
rows = []
for n in NAMES:
    for a in ("0.1", "0.15"):
        F = main[f"{n}|all|{a}"]["filter"]; b = boot[f"{n}|{0.1 if a=='0.1' else 0.15}"]
        e, f_ = b["ext"], b["fitsel"]
        qe = b["q_ext"]
        rows.append(f"{SHORT[n] if a=='0.1' else ''} & {float(a):.2f} & {p1(F['cov'])} ({ab(F['abort'])}) & "
                    f"{cell(n,a,'ext')} & {sg(main[f'{n}|all|{a}']['ext']['gain'])} [{sg(e['gain'][1])}, {sg(e['gain'][2])}] & {e['frac_pos']:.2f} & "
                    f"{cell(n,a,'fitsel')} & {sg(main[f'{n}|all|{a}']['fitsel']['gain'])} [{sg(f_['gain'][1])}, {sg(f_['gain'][2])}]\\\\")
    rows.append("\\addlinespace")
w("main", "\\begin{tabular}{lcccccccc}\n\\toprule\n & & Filter & \\multicolumn{3}{c}{CCRC, external gate} & & \\multicolumn{2}{c}{CCRC, fit-fold gate}\\\\\n\\cmidrule(lr){4-6}\\cmidrule(lr){8-9}\nSetting & $\\alpha$ & cov.\\ (abort) & cov.\\ (abort) & gain [95\\% CI] & $P(\\text{gain}>0)$ & & cov.\\ (abort) & gain [95\\% CI]\\\\\n\\midrule\n" + "\n".join(rows[:-1]) + "\n\\bottomrule\n\\end{tabular}\n")

# --------------------------------------------------- audit + alternatives (all-item cohort)
rows = []
for n in NAMES:
    for a in ("0.1", "0.15"):
        c = main[f"{n}|all|{a}"]; r = f"{SHORT[n] if a=='0.1' else ''} & {float(a):.2f}"
        for arm in ("filter", "ext", "fitsel", "union", "fixed10"):
            x = c[arm]; r += f" & {p1(x['cov'])} ({ab(x['abort'])}) & {x['exc_te']:.2f} / {x['conf_viol']:.2f}"
        rows.append(r + "\\\\")
    rows.append("\\addlinespace")
w("audit", "\\begin{tabular}{lcrrrrrrrrrr}\n\\toprule\n & & \\multicolumn{2}{c}{Filter} & \\multicolumn{2}{c}{External gate} & \\multicolumn{2}{c}{Fit-fold gate} & \\multicolumn{2}{c}{Union ($\\delta/3$)} & \\multicolumn{2}{c}{Fixed $q{=}0.10$}\\\\\n\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\\cmidrule(lr){9-10}\\cmidrule(lr){11-12}\nSetting & $\\alpha$ & cov.\\ (ab) & exc/cv & cov.\\ (ab) & exc/cv & cov.\\ (ab) & exc/cv & cov.\\ (ab) & exc/cv & cov.\\ (ab) & exc/cv\\\\\n\\midrule\n" + "\n".join(rows[:-1]) + "\n\\bottomrule\n\\end{tabular}\n")

# ---------------------------------------------------------------- ladder
RUNGS = ["L0 conventional protocol", "L1 + image-grouped folds", "L2 + fixed value grid and gate", "L3 + analytic start e0=1", "L4 + gate chosen on fit fold", "L5 + all items (missing policy)"]
DESC = ["conventional protocol (item folds, calibration quantiles, $e_0{=}0$, fixed $q{=}0.10$)", "+ image-grouped folds", "+ fixed value grid and gate (fit fold)", "+ analytic start $e_0{=}1$", "+ gate chosen on the fit fold", "+ all-item cohort"]
for a in ("0.1", "0.15"):
    rows = []
    for rn, d in zip(RUNGS, DESC):
        r = d
        for n in NAMES:
            x = ladder[f"{n}|{rn}|{a}"]; r += f" & {p1(x['filter']['cov'])}/{p1(x['ccrc']['cov'])}"
        rows.append(r + "\\\\")
    w(f"ladder_{a.replace('.','')}", "\\begin{tabular}{l" + "c" * len(NAMES) + "}\n\\toprule\n & " + " & ".join(SHORT[n].replace(", ", ",\\ ") for n in NAMES) + "\\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

# ---------------------------------------------------------------- start ablation
for a in ("0.05", "0.1", "0.15"):
    rows = []
    for c in (1.0, 1.5):
        for e0 in (0, 1, 2, 3):
            r = f"{c:.1f} & {e0} & {start[f'{NAMES[0]}|c={c}|e0={e0}|{a}']['ks']}"
            for n in NAMES:
                x = start[f"{n}|c={c}|e0={e0}|{a}"]; r += f" & {p1(x['filter']['cov'])} ({ab(x['filter']['abort'])}) & {p1(x['ccrc']['cov'])} ({ab(x['ccrc']['abort'])})"
            rows.append(r + "\\\\")
        rows.append("\\addlinespace")
    w(f"start_{a.replace('.','')}", "\\begin{tabular}{ccc" + "rr" * len(NAMES) + "}\n\\toprule\n & & & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{SHORT[n].replace(', ', ',~')}}}" for n in NAMES) + "\\\\\n" + "".join(f"\\cmidrule(lr){{{4+2*i}-{5+2*i}}}" for i in range(len(NAMES))) + "\n$c$ & $e_0$ & $k_{\\mathrm{start}}$ & " + " & ".join("filter & CCRC" for _ in NAMES) + "\\\\\n\\midrule\n" + "\n".join(rows[:-1]) + "\n\\bottomrule\n\\end{tabular}\n")

# --------------------------------------------------- repair accounting (external gate, all items)
rows = []
for n in NAMES:
    for a in ("0.1", "0.15"):
        x = main[f"{n}|all|{a}"]["ext"]; A = x["acct"]; kr = max(A["k_rep"], 1)
        rows.append(f"{SHORT[n] if a=='0.1' else ''} & {float(a):.2f} & {x['q_ext'] if x['q_ext'] is not None else '--'} & {A['k_rep']/A['n']*100:.2f} & {A['changed']/A['n']*100:.2f} & "
                    f"{A['changed']/kr*100:.0f} & {A['w2r']/kr*100:.0f} & {A['r2w']/kr*100:.1f} & {A['agree_right']/kr*100:.0f} & {A['agree_wrong']/kr*100:.1f} & "
                    f"{A['fp_corr']}/{A['fp_tot']} & {A['fn_corr']/max(A['fn_tot'],1)*100:.1f}\\\\")
    rows.append("\\addlinespace")
w("acct", "\\begin{tabular}{lccccccccccc}\n\\toprule\n & & & \\multicolumn{2}{c}{\\% of test items} & \\multicolumn{5}{c}{\\% of repaired items} & & \\\\\n\\cmidrule(lr){4-5}\\cmidrule(lr){6-10}\nSetting & $\\alpha$ & $q$ & repaired & answer changed & changed & wrong$\\to$right & right$\\to$wrong & agree, right & agree, wrong & FP corrected & FN corrected (\\%)\\\\\n\\midrule\n" + "\n".join(rows[:-1]) + "\n\\bottomrule\n\\end{tabular}\n")

# ---------------------------------------------------------------- polarity gates
rows = []
for n in NAMES:
    v = sym[f"{n}|0.1"]; r = SHORT[n]
    for rule in ("margin", "pos", "neg", "sym"):
        x = v[f"{rule}|q10"]; A = x["acct"]
        r += f" & {sg(x['gain'])} & {A['k_rep']/A['n']*100:.2f} & {A['fp_corr']}"
    rows.append(r + "\\\\")
w("sym", "\\begin{tabular}{lcccccccccccc}\n\\toprule\n & \\multicolumn{3}{c}{pooled margin (CCRC)} & \\multicolumn{3}{c}{positive only} & \\multicolumn{3}{c}{negative only} & \\multicolumn{3}{c}{symmetric}\\\\\n\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}\\cmidrule(lr){11-13}\nSetting & gain & rep.\\ \\% & FP fixed & gain & rep.\\ \\% & FP fixed & gain & rep.\\ \\% & FP fixed & gain & rep.\\ \\% & FP fixed\\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

# ---------------------------------------------------------------- scores
KIND = [("clip_only", "image--text similarity only (ConfLVLM scorer)"), ("conf", "VLM confidence only"), ("learned", "learned, VLM signals"), ("clip", "+ image--text similarity"),
        ("det", "+ detection grounding"), ("both", "+ both")]
rows = []
for k, d in KIND:
    r = d
    for a in ("0.05", "0.1", "0.15"):
        x = sc[f"POPE-1500 LLaVA|{k}|{a}"]; r += f" & {x['cov']*100:.1f} ({x['abort']*100:.0f})"
    r += f" & {sc[f'POPE-1500 LLaVA|{k}|0.1']['aurc']:.4f}"
    rows.append(r + "\\\\")
w("scores", "\\begin{tabular}{lcccc}\n\\toprule\nScore & $\\alpha{=}0.05$ & $\\alpha{=}0.10$ & $\\alpha{=}0.15$ & AURC\\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
rows = []
for n in NAMES:
    r = SHORT[n]
    for a in ("0.1", "0.15"):
        d = sc[f"{n}|detector_only|{a}"]; f_ = main[f"{n}|all|{a}"]
        r += f" & {p1(f_['filter']['cov'])} & {p1(f_['ext']['cov'])} & {d['cov']*100:.1f}"
    rows.append(r + f" & {sc[f'{n}|detector_only|0.1']['acc']*100:.1f}\\\\")
w("detonly", "\\begin{tabular}{lccccccc}\n\\toprule\n & \\multicolumn{3}{c}{$\\alpha=0.10$} & \\multicolumn{3}{c}{$\\alpha=0.15$} & \\\\\n\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\nSetting & VLM filter & CCRC & detector only & VLM filter & CCRC & detector only & detector acc.\\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

# ---------------------------------------------------------------- self-repair
rows = []
for n in NAMES:
    for e0 in (0, 1):
        v = selfr[f"{n}|e0={e0}|0.1"]; r = f"{SHORT[n] if e0==0 else ''} & {e0} & {p1(v['filter']['cov'])} ({ab(v['filter']['abort'])})"
        for q in ("self0.02", "self0.05", "self0.1"):
            x = v[q]; fr = x["flip_region_risk"]; r += f" & {sg(x['gain'])} ({ab(x['abort'])}) & {('%.0f' % (fr*100)) if fr is not None else '--'}"
        r += f" & {sg(v['ccrc10']['gain'])}"
        rows.append(r + "\\\\")
    rows.append("\\addlinespace")
w("selfrepair", "\\begin{tabular}{lccrrrrrrr}\n\\toprule\n & & & \\multicolumn{2}{c}{$q{=}0.02$} & \\multicolumn{2}{c}{$q{=}0.05$} & \\multicolumn{2}{c}{$q{=}0.10$} & \\\\\n\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}\nSetting & $e_0$ & filter (ab) & gain (ab) & $r_{\\text{flip}}$ & gain (ab) & $r_{\\text{flip}}$ & gain (ab) & $r_{\\text{flip}}$ & CCRC gain\\\\\n\\midrule\n" + "\n".join(rows[:-1]) + "\n\\bottomrule\n\\end{tabular}\n")

# ---------------------------------------------------------------- sweep
rows = [f"{k} & {v['images']} & {v['mu']*100:.1f} & {v['grounded']} & " + " & ".join(f"{v[a]['cov']*100:.1f} ({v[a]['abort']*100:.0f})" for a in ("0.1", "0.15", "0.2")) + "\\\\" for k, v in sweep.items()]
w("sweep", "\\begin{tabular}{lrrrccc}\n\\toprule\nDataset & images & $\\mu$ (\\%) & detector value & $\\alpha{=}0.10$ & $\\alpha{=}0.15$ & $\\alpha{=}0.20$\\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

# ---------------------------------------------------------------- validity (synthetic), if available
if os.path.exists(os.path.join(H, "results_validity.json")):
    V = L("results_validity.json"); rows = []
    for sigma, lab in ((0.0, "0 (i.i.d.)"), (1.0, "0.08"), (2.0, "0.26")):
        for mu in (0.15, 0.25):
            for n_cal in (150, 450):
                r = f"{lab} & {mu:.2f} & {n_cal}"
                def g(proc, arm, calib="full", al=0.10):
                    k = f"{mu}|{al}|{n_cal}|{sigma}|{proc}|{arm}|{calib}"; return V.get(k)
                for proc, arm, calib in (("value", "filter", "full"), ("value", "repair", "full"), ("rank", "filter", "full"), ("rank", "repair", "full"), ("value", "repair", "single")):
                    x = g(proc, arm, calib)
                    r += " & --" if x is None else f" & {x['exc_incl_abort']:.3f}"
                rows.append(r + "\\\\")
        rows.append("\\addlinespace")
    w("validity", "\\begin{tabular}{cccccccc}\n\\toprule\nICC & $\\mu$ & $n_{\\mathrm{cal}}$ & value, filter & value, repair & rank, filter & rank, repair & value, repair, 1 item/image\\\\\n\\midrule\n" + "\n".join(rows[:-1]) + "\n\\bottomrule\n\\end{tabular}\n")
print("tables written:", sorted(os.listdir(OUT)))

# ---------------------------------------------------------------- dilution
dil = L("results_dilution.json"); rows = []
for n in NAMES:
    for a in ("0.1", "0.15"):
        x = dil[f"{n}|{a}"]
        rows.append(f"{SHORT[n] if a=='0.1' else ''} & {float(a):.2f} & {x['moved']*100:.0f} & {x['gain']:.2f} & {x['rep']:.2f} & {x['dil']:.2f} [{x['dil_lo']:.2f}, {x['dil_hi']:.2f}]\\\\")
    rows.append("\\addlinespace")
w("dilution", "\\begin{tabular}{lccccc}\n\\toprule\nSetting & $\\alpha$ & threshold moves (\\%) & gain & (i) repairs at filter's threshold & (ii) dilution [95\\% CI over splits]\\\\\n\\midrule\n" + "\n".join(rows[:-1]) + "\n\\bottomrule\n\\end{tabular}\n")

# ------------------------------------------- grounded-only cohort (appendix)
rows = []
for n in NAMES:
    for a in ("0.1", "0.15"):
        c = main[f"{n}|grounded|{a}"]; F = c["filter"]
        rows.append(f"{SHORT[n] if a=='0.1' else ''} & {float(a):.2f} & {p1(F['cov'])} ({ab(F['abort'])}) & " + " & ".join(f"{p1(c[k]['cov'])} ({ab(c[k]['abort'])}) & {sg(c[k]['gain'])}" for k in ("ext", "fitsel", "fixed10")) + "\\\\")
    rows.append("\\addlinespace")
w("grounded", "\\begin{tabular}{lcccccccc}\n\\toprule\n & & Filter & \\multicolumn{2}{c}{External gate} & \\multicolumn{2}{c}{Fit-fold gate} & \\multicolumn{2}{c}{Fixed $q{=}0.10$}\\\\\n\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}\nSetting & $\\alpha$ & cov.\\ (ab) & cov.\\ (ab) & gain & cov.\\ (ab) & gain & cov.\\ (ab) & gain\\\\\n\\midrule\n" + "\n".join(rows[:-1]) + "\n\\bottomrule\n\\end{tabular}\n")
print("extra tables written")
