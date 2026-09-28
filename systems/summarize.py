"""Roll every answer matrix in outputs/answers into one table.

Selection-proxy numbers only (see systems/devsel.py): they exist so the B track
can report smoke values and so the A track has something to diff its own metric
implementation against. The figures of record come from ape/metrics.py.

Usage:
    python -X utf8 systems/summarize.py --split test --out outputs/b_summary.csv
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from systems import commit as CM  # noqa: E402
from systems import common as C  # noqa: E402
from systems import devsel  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--answers-dir", default=C.ANSWERS_DIR)
    ap.add_argument("--split", default="test")
    ap.add_argument("--h-list", default="3,6,10")
    ap.add_argument("--out", default=os.path.join(C.OUTPUTS, "b_summary.csv"))
    args = ap.parse_args()

    h_list = [float(h) for h in args.h_list.split(",")]
    man = C.load_manifest(args.manifest)
    sub = man[man["split"] == args.split]
    lab = dict(zip(sub["video_id"], sub["class_code"].astype(int)))
    plen = dict(zip(sub["video_id"], sub["post_anchor_length_s"].astype(float)))

    sids = sorted(d for d in os.listdir(args.answers_dir)
                  if os.path.isdir(os.path.join(args.answers_dir, d)))
    rows: List[Dict] = []
    for sid in sids:
        apath = os.path.join(args.answers_dir, sid, "answers.csv")
        if not os.path.exists(apath):
            continue
        df = C.read_answers(apath)
        df = df[df["video_id"].isin(lab)]
        if df.empty:
            continue
        vids, js, preds_all, _probs = C.answers_to_arrays(df)
        post = C.post_anchor_slice(js)
        preds = preds_all[:, post]
        y = np.array([lab[v] for v in vids], dtype=np.int64)
        pl = np.array([plen[v] for v in vids], dtype=np.float64)
        delta_s = float(df["delta_s"].iloc[0])

        cpath = os.path.join(args.answers_dir, sid, "system_card.yaml")
        card = {}
        if os.path.exists(cpath):
            with open(cpath, "r", encoding="utf-8") as fh:
                card = yaml.safe_load(fh) or {}

        row = {"system_id": sid, "family": card.get("family"),
               "parent_system_id": card.get("parent_system_id"),
               "delta_s": delta_s, "n_clips": len(vids),
               "j_min": int(js[0]), "j_max": int(js[-1]),
               "bot_rate_post_anchor": float((preds == C.BOT).mean()),
               "bot_rate_pre_anchor": float((preds_all[:, :post.start] == C.BOT).mean())
               if post.start > 0 else 0.0}
        for h in h_list:
            s = devsel.window_summary(preds, y, pl, h, delta_s)
            tag = f"H{h:g}"
            row[f"{tag}_N"] = s["N_H"]
            row[f"{tag}_rmscd"] = round(s["rmscd"], 4)
            row[f"{tag}_end_macro_acc"] = round(s["end_macro_acc"], 4)
            row[f"{tag}_median_flips"] = s["median_flips"]
        if card.get("family") == "commit":
            committed_all = df["committed"].to_numpy().astype(bool).reshape(len(vids), -1)
            # monotonicity is a contract requirement, so check it rather than assume it
            viol = int((committed_all[:, :-1] & ~committed_all[:, 1:]).sum())
            row["committed_monotone_violations"] = viol
            committed = committed_all[:, post]
            fires = committed.any(axis=1)
            first = np.where(fires, committed.argmax(axis=1), committed.shape[1])
            h = C.H_DEV_SELECT
            m = devsel.cohort_mask(pl, h)
            trip = CM.commit_triplet(first[m], preds[m], y[m], int(round(h / delta_s)), delta_s)
            row.update({"commit_rho": round(trip["rho"], 4),
                        "commit_tau_c": round(trip["tau_c"], 4),
                        "commit_e_c": round(trip["e_c"], 4) if trip["e_c"] == trip["e_c"] else None})
        rows.append(row)

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"split={args.split}  systems={len(out)}  ->  {args.out}\n")
    cols = ["system_id", "family", "n_clips", "bot_rate_post_anchor",
            "H6_N", "H6_end_macro_acc", "H6_rmscd", "H6_median_flips",
            "H3_end_macro_acc", "H10_end_macro_acc"]
    cols = [c for c in cols if c in out.columns]
    print(out[cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
