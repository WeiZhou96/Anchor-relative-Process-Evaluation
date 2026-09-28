"""Post-processing arms on top of a base system's probability trajectory.

Four arms, all causal and all exam-takers rather than proposals: exponential
moving average, sliding majority vote, hysteresis and patience(n_p).  Each one
reads the base answer matrix and writes its own CONTRACT 5.4 matrix with
``parent_system_id`` pointing at the base.

Window lengths and thresholds are swept on ``split=dev`` over a small grid and
picked by the smallest dev RMSCD@H subject to window-end macro accuracy not
falling below the base's.  ``split=test`` is not consulted for selection.

Usage:
    python -X utf8 systems/postproc.py --base prefix__gru512__seed20260903
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

GRIDS: Dict[str, List[float]] = {
    "ema": [0.2, 0.35, 0.5, 0.7],          # smoothing weight on the current step
    "majority": [2, 3, 5, 7, 9],           # window length in grid steps
    "hysteresis": [0.05, 0.10, 0.20, 0.30],  # probability margin required to switch away
    "patience": [2, 3, 4, 6],              # consecutive agreeing steps required to switch
}


def _argmax_safe(p: np.ndarray) -> np.ndarray:
    """argmax with NaN rows mapped to the bot code."""
    bad = np.isnan(p).any(axis=-1)
    out = np.where(bad, C.BOT, np.nan_to_num(p, nan=-np.inf).argmax(axis=-1))
    return out.astype(np.int64)


def arm_ema(probs: np.ndarray, alpha: float):
    """Exponential moving average over the probability trajectory.

    A grid cell with no probabilities is missing evidence, not evidence of
    nothing: the smoother holds its previous state instead of averaging a NaN in.
    Blending NaN would poison the whole remaining trajectory of any clip whose
    anchor sits near its head, since those clips have empty prefixes at negative j.
    """
    n, j, k = probs.shape
    q = np.empty_like(probs)
    q[:, 0] = probs[:, 0]
    valid = ~np.isnan(probs).any(axis=-1)
    for t in range(1, j):
        prev = q[:, t - 1]
        have_prev = ~np.isnan(prev).any(axis=-1)
        blended = alpha * probs[:, t] + (1.0 - alpha) * prev
        # fresh evidence with a running state -> blend; fresh evidence without one
        # -> start here; no evidence -> hold
        take = np.where((valid[:, t] & have_prev)[:, None], blended,
                        np.where((valid[:, t] & ~have_prev)[:, None], probs[:, t], prev))
        q[:, t] = take
    return _argmax_safe(q), q


def arm_majority(probs: np.ndarray, w: int):
    base = _argmax_safe(probs)
    n, j = base.shape
    out = np.empty_like(base)
    freq = np.zeros((n, j, C.N_CLASSES), dtype=np.float64)
    for t in range(j):
        lo = max(0, t - w + 1)
        win = base[:, lo:t + 1]
        counts = np.stack([(win == c).sum(axis=1) for c in range(C.N_CLASSES)], axis=1)
        # normalise over the valid votes in the window, not its length: a window
        # holding only bot cells has no distribution at all and is left NaN below
        tot = counts.sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            freq[:, t] = np.where(tot > 0, counts / np.maximum(tot, 1), np.nan)
        best = counts.max(axis=1, keepdims=True)
        tied = counts == best
        # tie break toward the most recent label so the arm stays causal and deterministic
        recent = base[:, t]
        pick = counts.argmax(axis=1)
        keep_recent = tied[np.arange(n), np.clip(recent, 0, C.N_CLASSES - 1)] & (recent >= 0)
        out[:, t] = np.where(keep_recent, recent, pick)
        out[:, t] = np.where(base[:, t] == C.BOT, C.BOT, out[:, t])
    # a cell with no answer carries no distribution either
    freq[out == C.BOT] = np.nan
    return out, freq


def arm_hysteresis(probs: np.ndarray, margin: float):
    base = _argmax_safe(probs)
    n, j = base.shape
    out = np.full_like(base, C.BOT)
    cur = base[:, 0].copy()
    out[:, 0] = cur
    safe = np.nan_to_num(probs, nan=0.0)
    for t in range(1, j):
        k = base[:, t]
        idx = np.arange(n)
        p_k = safe[idx, t, np.clip(k, 0, C.N_CLASSES - 1)]
        p_c = safe[idx, t, np.clip(cur, 0, C.N_CLASSES - 1)]
        switch = (k != cur) & (k != C.BOT) & ((p_k - p_c) > margin)
        # a clip that could not answer yet adopts the first available label
        adopt = (cur == C.BOT) & (k != C.BOT)
        cur = np.where(switch | adopt, k, cur)
        out[:, t] = cur
    return out, probs


def arm_patience(probs: np.ndarray, n_p: int):
    base = _argmax_safe(probs)
    n, j = base.shape
    out = np.full_like(base, C.BOT)
    cur = base[:, 0].copy()
    out[:, 0] = cur
    cand = np.full(n, C.BOT, dtype=np.int64)
    run = np.zeros(n, dtype=np.int64)
    for t in range(1, j):
        k = base[:, t]
        same_cand = k == cand
        run = np.where(same_cand, run + 1, 1)
        cand = k
        switch = (k != cur) & (k != C.BOT) & (run >= n_p)
        adopt = (cur == C.BOT) & (k != C.BOT)
        cur = np.where(switch | adopt, k, cur)
        out[:, t] = cur
    return out, probs


ARMS = {"ema": arm_ema, "majority": arm_majority,
        "hysteresis": arm_hysteresis, "patience": arm_patience}


def find_stride1_system(answers_dir: str, arm: str, base: str):
    """Locate the finest-grid system for this (arm, base) and read back its chosen value.

    A coarse-step build is the *same* system regenerated, not a different one, so it
    must reuse the parameter that was selected on dev at the finest step rather than
    re-running the sweep. Returns (system_id, value).
    """
    import yaml

    prefix, suffix = f"postproc__{arm}", f"__{base}"
    cands = [d for d in os.listdir(answers_dir)
             if d.startswith(prefix) and d.endswith(suffix) and "__stride" not in d
             and os.path.isdir(os.path.join(answers_dir, d))]
    if len(cands) != 1:
        raise SystemExit(
            f"expected exactly one finest-grid system for arm={arm} base={base}, "
            f"found {cands}; run without --stride first")
    sid = cands[0]
    with open(os.path.join(answers_dir, sid, "system_card.yaml"), "r", encoding="utf-8") as fh:
        card = yaml.safe_load(fh) or {}
    return sid, card["dev_tuned_params"]["value"]


def apply_arm(name: str, probs: np.ndarray, value):
    if name == "majority":
        return arm_majority(probs, int(value))
    if name == "patience":
        return arm_patience(probs, int(value))
    if name == "ema":
        return arm_ema(probs, float(value))
    return arm_hysteresis(probs, float(value))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="base system_id whose answers.csv is read")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--h-select", type=float, default=C.H_DEV_SELECT)
    ap.add_argument("--arms", default="ema,majority,hysteresis,patience")
    ap.add_argument("--answers-dir", default=C.ANSWERS_DIR)
    ap.add_argument("--stride", type=int, default=1,
                    help="apply the arm on every k-th base grid point; these arms are stateful, "
                         "so a coarser delta_s needs this rather than a sub-sample of the output")
    args = ap.parse_args()

    man = C.load_manifest(args.manifest)
    base_df = C.read_answers(args.base)
    vids, js, base_preds, base_probs = C.answers_to_arrays(base_df)
    delta_s = float(base_df["delta_s"].iloc[0])
    if args.stride > 1:
        # keep j divisible by the stride so that j = 0 stays on the coarse grid
        keep = (js % args.stride) == 0
        js = js[keep] // args.stride
        base_preds = base_preds[:, keep]
        base_probs = base_probs[:, keep]
        delta_s *= args.stride
    post = C.post_anchor_slice(js)
    idx = {v: i for i, v in enumerate(vids)}

    def split_view(split: str):
        sub = man[man["split"] == split]
        keep = [v for v in sub["video_id"] if v in idx]
        rows = np.array([idx[v] for v in keep])
        y = sub.set_index("video_id").loc[keep, "class_code"].to_numpy().astype(np.int64)
        pl = sub.set_index("video_id").loc[keep, "post_anchor_length_s"].to_numpy().astype(np.float64)
        return rows, y, pl

    dv_rows, dv_y, dv_pl = split_view("dev")
    te_rows, te_y, te_pl = split_view("test")
    base_dev = devsel.window_summary(base_preds[dv_rows][:, post], dv_y, dv_pl,
                                     args.h_select, delta_s)
    print(f"base {args.base} dev@H={args.h_select}: {base_dev}", flush=True)

    for arm in [a.strip() for a in args.arms.split(",") if a.strip()]:
        scan = []
        if args.stride > 1:
            # regenerate the finest-grid system at a coarser step: same rule, same
            # parameter, no fresh sweep
            sid1, value = find_stride1_system(args.answers_dir, arm, args.base)
            preds, probs = apply_arm(arm, base_probs, value)
            dev_s = devsel.window_summary(preds[dv_rows][:, post], dv_y, dv_pl,
                                          args.h_select, delta_s)
            feasible = dev_s["end_macro_acc"] >= base_dev["end_macro_acc"] - 1e-9
            scan.append({"arm": arm, "value": value, "feasible": bool(feasible),
                         "reused_from": sid1, **dev_s})
            sid = f"{sid1}__stride{args.stride}"
        else:
            best = None
            for value in GRIDS[arm]:
                # the arm runs over the whole grid, pre-anchor steps included: they
                # are the warm-up a causal stream processor would really have seen
                preds, probs = apply_arm(arm, base_probs, value)
                s = devsel.window_summary(preds[dv_rows][:, post], dv_y, dv_pl,
                                          args.h_select, delta_s)
                feasible = s["end_macro_acc"] >= base_dev["end_macro_acc"] - 1e-9
                scan.append({"arm": arm, "value": value, "feasible": bool(feasible), **s})
                key = (0 if feasible else 1, s["rmscd"])
                if best is None or key < best[0]:
                    best = (key, value, preds, probs, s, feasible)
            _key, value, preds, probs, dev_s, feasible = best
            tag = f"{value:g}".replace(".", "p")
            sid = f"postproc__{arm}{tag}__{args.base}"
        card = {
            "system_id": sid,
            "family": "postproc",
            "description": f"{arm} post-processing arm on the base system's probability trajectory"
                           + (f", regenerated at delta_s={delta_s:g}" if args.stride > 1 else ""),
            "backbone": "none (operates on cached answer matrix)",
            "trained_on_split": "train (via parent)",
            "delta_s": delta_s,
            "stride_from_finest": args.stride,
            "dev_tuned_params": {"arm": arm, "value": value,
                                 "H_used_for_selection": args.h_select,
                                 "objective": "min dev RMSCD@H s.t. dev window-end macro-acc >= base",
                                 "constraint_satisfied": bool(feasible),
                                 "dev_rmscd": dev_s["rmscd"],
                                 "dev_end_macro_acc": dev_s["end_macro_acc"],
                                 "base_dev_end_macro_acc": base_dev["end_macro_acc"],
                                 "grid": GRIDS[arm],
                                 "selected_at_delta_s": C.FINEST_DELTA_S},
            "train_data_unknown": False,
            "cost_note": "negligible; a causal pass over the base answer matrix",
            "causal": True,
            "parent_system_id": args.base,
            "subsampling_equivalent": False,
            "subsampling_note": ("all four arms carry state across grid steps, so a coarser grid is "
                                 "a different system, not a sub-sample of this one. "
                                 + (f"THIS IS A REGENERATED BUILD at delta_s={delta_s:g}: the rule "
                                    f"and its dev-selected parameter are identical to the finest-grid "
                                    f"system, only the grid it runs on is coarser."
                                    if args.stride > 1 else
                                    "Coarser steps must be regenerated with --stride k.")),
        }
        answers = C.build_answer_frame(vids, preds, probs, delta_s=delta_s,
                                       j_offset=int(js[0]))
        out = C.write_answers(sid, answers, card)
        pd.DataFrame(scan).to_csv(os.path.join(out, "dev_scan.csv"), index=False)
        te_s = devsel.window_summary(preds[te_rows][:, post], te_y, te_pl, args.h_select, delta_s)
        print(f"{sid}\n  dev {dev_s}\n  test {te_s}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
