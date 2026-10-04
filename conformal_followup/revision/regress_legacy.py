"""Regression test: the legacy mode of ccrc_rev must reproduce the previously
reported Table-5 cells (coverage, filtering -> CCRC q=0.10, alpha=0.10/0.15)."""
import time, numpy as np, ccrc_rev as R
REF = {("POPE-1500 LLaVA", .10): (68.6, 71.8), ("POPE-adv LLaVA", .10): (44.0, 47.2),
       ("POPE-adv Qwen2-VL", .10): (77.9, 79.5), ("POPE-adv LLaVA+VCD", .10): (48.8, 51.0),
       ("POPE-1500 LLaVA", .15): (86.4, 88.4), ("POPE-adv LLaVA", .15): (69.1, 76.7),
       ("POPE-adv Qwen2-VL", .15): (94.8, 95.2), ("POPE-adv LLaVA+VCD", .15): (75.8, 77.2)}
S = R.load_settings(); t0 = time.time()
for name in ["POPE-1500 LLaVA", "POPE-adv LLaVA", "POPE-adv Qwen2-VL", "POPE-adv LLaVA+VCD"]:
    run = R.Runner(S[name], R.Mode.legacy())
    for a in (.10, .15):
        f = R.summarise(run.run(a, gate=None), a); c = R.summarise(run.run(a, gate="fixed"), a)
        ref = REF[(name, a)]
        print(f"{name:22s} a={a:.2f}  filter {f['cov']*100:5.1f} (ref {ref[0]:4.1f}) ab {f['abort']*100:4.1f} | "
              f"ccrc {c['cov']*100:5.1f} (ref {ref[1]:4.1f}) ab {c['abort']*100:4.1f}")
print(f"{time.time()-t0:.0f}s")
