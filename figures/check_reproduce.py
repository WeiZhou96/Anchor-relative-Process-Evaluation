import json, numpy as np, pandas as pd
from paths import *
scan = pd.read_csv(REMOTE04/"r2b/r2/scan_all.csv")
real = scan[~scan.is_block.astype(bool)]
meta = scan.drop_duplicates("pi_hash").set_index("pi_hash")
for h, ph in HASH.items():
    cal = json.load(open(REMOTE04/f"r2b/calib_plaus/{ph}/calibration.json", encoding="utf-8"))
    for m in METRICS:
        R = pd.read_csv(REMOTE04/f"r2b/calib_plaus/{ph}/R_{METRIC_FILE[m]}.csv")
        plaus = set(R.pi_hash)
        piv = real[real.metric==m].pivot_table(index="system_id", columns="pi_hash", values="value", aggfunc="first")
        use = [c for c in piv.columns if c!=ph and c in plaus]
        if m.startswith("RMSCD"):
            use = [c for c in use if abs(meta.loc[c,"h_s"]-h)<1e-9]
        d = np.abs(piv[use].values - piv[[ph]].values); d = d[np.isfinite(d)]
        print(h, m, "MRD %.6f stored %.6f | npts %d | minR %.6f stored %.6f" % (np.percentile(d,95), cal["metrics"][m]["MRD_plaus"], len(use), R.R_M.min(), cal["metrics"][m]["min_R"]))
