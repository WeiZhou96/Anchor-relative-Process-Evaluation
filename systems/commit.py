"""Irrevocable commitment arms (MSP threshold and margin threshold).

A commitment system stays silent until its confidence first crosses the
threshold, then names its current Top-1 and never revises it.  Answer rows
before the commitment carry the bot code and ``committed=False``; from the
commitment step onward the prediction is frozen and ``committed=True``
(CONTRACT 5.4).  Clips that never cross stay at bot for the whole grid and must
not be dropped -- the A track charges them the full window in tau_c.

Threshold grids are read on ``split=dev``; every level is shipped as its own
system directory so the A track can trace the (rho, tau_c, e_c) frontier.

Usage:
    python -X utf8 systems/commit.py --base prefix__gru512__seed20260903
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from systems import common as C  # noqa: E402
from systems import devsel  # noqa: E402

MSP_LEVELS = [0.5, 0.6, 0.7, 0.8, 0.9]
MARGIN_LEVELS = [0.1, 0.2, 0.4, 0.6]
# CONTRACT 5.2 pre-registers rho_levels [0.7, 0.9]; the wider grid is shipped so
# the frontier can be drawn, and the two pre-registered levels are flagged.
CONTRACT_RHO = [0.7, 0.9]


def commitment_scores(probs: np.ndarray, rule: str) -> np.ndarray:
    """Confidence score per grid cell. Cells with no probabilities score 0.

    Zero rather than -inf: an all -inf row makes the margin an inf - inf NaN. Zero
    is below every threshold in the grid, and such cells are masked by ``valid``
    anyway, so the rule can never fire on a prefix that produced no answer.
    """
    safe = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    if rule == "msp":
        return safe.max(axis=-1)
    order = np.sort(safe, axis=-1)
    return order[..., -1] - order[..., -2]


def apply_commit(probs: np.ndarray, rule: str, thr: float, start_col: int = 0):
    """Return (preds, committed, first_col) for an irrevocable threshold rule.

    ``start_col`` is the first column the rule is allowed to fire on. The answer
    grid begins before the anchor, and a commitment is a post-anchor act: tau_c is
    measured from the anchor, so firing at delta < 0 would make the triplet
    meaningless. Columns before ``start_col`` carry the base system's raw Top-1
    with ``committed=False`` -- the protocol layer masks them anyway, and keeping
    the raw value there is what the A track asked every system to do.
    """
    n, j, _ = probs.shape
    score = commitment_scores(probs, rule)
    valid = ~np.isnan(probs).any(axis=-1)
    top1 = np.nan_to_num(probs, nan=-np.inf).argmax(axis=-1)

    fires = valid & (score >= thr)
    fires[:, :start_col] = False
    any_fire = fires.any(axis=1)
    first = np.where(any_fire, fires.argmax(axis=1), j)  # j == "never"

    idx = np.arange(n)
    label = np.where(any_fire, top1[idx, np.clip(first, 0, j - 1)], C.BOT)
    step = np.arange(j)[None, :]
    on = step >= first[:, None]

    committed = np.zeros((n, j), dtype=bool)
    committed[on] = True
    # pre-anchor: raw Top-1; post-anchor before the crossing: bot; after: frozen label
    raw = np.where(valid, top1, C.BOT)
    pre = step < start_col
    preds = np.where(on, np.broadcast_to(label[:, None], (n, j)),
                     np.where(pre, raw, C.BOT))
    return preds.astype(np.int64), committed, first


def commit_triplet(first: np.ndarray, preds: np.ndarray, labels: np.ndarray,
                   j_end: int, delta_s: float) -> Dict[str, float]:
    """(rho, tau_c, e_c): commitment rate, delay with the window charged in full, post-commit error."""
    committed_in_window = first <= j_end
    rho = float(committed_in_window.mean())
    tau = np.where(committed_in_window, first * delta_s, j_end * delta_s)
    tau_c = float(tau.mean())
    if committed_in_window.sum() == 0:
        e_c = float("nan")
    else:
        e_c = float((preds[committed_in_window, j_end] != labels[committed_in_window]).mean())
    return {"rho": rho, "tau_c": tau_c, "e_c": e_c}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--h-select", type=float, default=C.H_DEV_SELECT)
    ap.add_argument("--stride", type=int, default=1)
    args = ap.parse_args()

    man = C.load_manifest(args.manifest)
    base_df = C.read_answers(args.base)
    vids, js, _base_preds, base_probs = C.answers_to_arrays(base_df)
    delta_s = float(base_df["delta_s"].iloc[0])
    if args.stride > 1:
        keep = (js % args.stride) == 0
        js = js[keep] // args.stride
        base_probs = base_probs[:, keep]
        delta_s *= args.stride
    post = C.post_anchor_slice(js)
    start_col = post.start
    idx = {v: i for i, v in enumerate(vids)}
    j_end = int(round(args.h_select / delta_s))

    def split_view(split: str):
        sub = man[man["split"] == split]
        keep = [v for v in sub["video_id"] if v in idx]
        rows = np.array([idx[v] for v in keep])
        y = sub.set_index("video_id").loc[keep, "class_code"].to_numpy().astype(np.int64)
        pl = sub.set_index("video_id").loc[keep, "post_anchor_length_s"].to_numpy().astype(np.float64)
        return rows, y, pl

    dv_rows, dv_y, dv_pl = split_view("dev")
    te_rows, te_y, te_pl = split_view("test")
    dv_cohort = devsel.cohort_mask(dv_pl, args.h_select)

    scan = []
    for rule, levels in (("msp", MSP_LEVELS), ("margin", MARGIN_LEVELS)):
        for thr in levels:
            preds, committed, first_col = apply_commit(base_probs, rule, thr, start_col)
            # express the crossing as a post-anchor step index for the triplet
            first = first_col - start_col
            post_preds = preds[:, post]
            trip = commit_triplet(first[dv_rows][dv_cohort], post_preds[dv_rows][dv_cohort],
                                  dv_y[dv_cohort], j_end, delta_s)
            dev_s = devsel.window_summary(post_preds[dv_rows], dv_y, dv_pl, args.h_select, delta_s)
            scan.append({"rule": rule, "thr": thr, "pre_registered": thr in CONTRACT_RHO,
                         **trip, "dev_rmscd": dev_s["rmscd"],
                         "dev_end_macro_acc": dev_s["end_macro_acc"]})

            tag = f"{thr:g}".replace(".", "p")
            sid = f"commit__{rule}{tag}__{args.base}"
            if args.stride > 1:
                sid = f"{sid}__stride{args.stride}"
            card = {
                "system_id": sid,
                "family": "commit",
                "description": (f"irrevocable commitment on the base trajectory: commit the current "
                                f"Top-1 the first time the {rule} score reaches {thr}, then hold"
                                + (f"; regenerated at delta_s={delta_s:g}" if args.stride > 1 else "")),
                "backbone": "none (operates on cached answer matrix)",
                "trained_on_split": "train (via parent)",
                "delta_s": delta_s,
                "stride_from_finest": args.stride,
                "dev_tuned_params": {"rule": rule, "threshold": thr,
                                     "pre_registered_in_pi0": thr in CONTRACT_RHO,
                                     "H_used_for_dev_readout": args.h_select,
                                     "dev_rho": trip["rho"], "dev_tau_c": trip["tau_c"],
                                     "dev_e_c": trip["e_c"],
                                     "grid": levels},
                "train_data_unknown": False,
                "cost_note": "negligible; a causal pass over the base answer matrix",
                "causal": True,
                "parent_system_id": args.base,
                "subsampling_equivalent": False,
                "subsampling_note": ("the first-crossing step is grid dependent, so a coarser delta_s "
                                     "must be regenerated, not sub-sampled. "
                                     + (f"THIS IS A REGENERATED BUILD at delta_s={delta_s:g}: same "
                                        f"rule and same threshold as the finest-grid system, run on "
                                        f"a coarser grid." if args.stride > 1 else
                                        "Coarser steps are produced with --stride k.")),
                "never_commit_policy": "clips that never cross keep pred=-1 for the whole "
                                       "post-anchor grid and are kept in the matrix",
                "pre_anchor_rows": "base raw Top-1 with committed=False; the rule may only fire "
                                   "at delta >= 0, since tau_c is measured from the anchor",
            }
            answers = C.build_answer_frame(vids, preds, base_probs, delta_s=delta_s,
                                           committed=committed, j_offset=int(js[0]))
            out = C.write_answers(sid, answers, card)
            te_cohort = devsel.cohort_mask(te_pl, args.h_select)
            te_trip = commit_triplet(first[te_rows][te_cohort],
                                     post_preds[te_rows][te_cohort],
                                     te_y[te_cohort], j_end, delta_s)
            print(f"{sid}  dev {trip}  test {te_trip}", flush=True)

    scan_df = pd.DataFrame(scan)
    os.makedirs(C.ANSWERS_DIR, exist_ok=True)
    stride_tag = "" if args.stride == 1 else f"__stride{args.stride}"
    scan_path = os.path.join(C.ANSWERS_DIR, f"_commit_dev_scan__{args.base}{stride_tag}.csv")
    scan_df.to_csv(scan_path, index=False)
    print(f"wrote {scan_path}")
    print(scan_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
