"""Trivial systems: the extreme anchors of the audit (CONTRACT 6, idea 4.3).

  * ``trivial__majority``  -- always the most frequent class of ``split=train``
  * ``trivial__random``    -- an independent uniform draw at every grid point, fixed seed
  * ``trivial__oracle``    -- the ground-truth class at every post-anchor step
  * ``trivial__constbot``  -- never answers; every step is the bot code

None of them reads a frame.  They exist so that the A track's rank-preservation
and comparability estimates have known degenerate endpoints, and so that a
perturbation grid that cannot separate oracle from random is visibly mis-set.

Usage:
    python -X utf8 systems/trivial.py
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from systems import common as C  # noqa: E402
from systems import devsel  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--delta-s", type=float, default=C.FINEST_DELTA_S)
    ap.add_argument("--j-min", type=int, default=C.J_MIN)
    ap.add_argument("--j-max", type=int, default=C.J_MAX)
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--stride", type=int, default=1,
                    help="regenerate on a coarser grid: delta_s = stride * finest step, and the "
                         "system_id gains a __stride<k> suffix. Only meaningful for the random "
                         "system, whose draws are per grid point; the others are stateless in the "
                         "grid index and sub-sample exactly")
    ap.add_argument("--only", default="",
                    help="comma separated subset of {majority,random,oracle,constbot}; "
                         "empty means all. Use it to regenerate one system without rewriting "
                         "the others")
    ap.add_argument("--restrict-to-features", action="store_true",
                    help="emit only clips that have a cached feature file, so that every "
                         "system in the library covers exactly the same clip set")
    args = ap.parse_args()

    # A coarse build covers the same wall-clock span with fewer points, so the j
    # bounds scale with the stride and j = 0 stays on the grid.
    if args.stride > 1:
        delta_s = C.FINEST_DELTA_S * args.stride
        j_min = -((-C.J_MIN) // args.stride)
        j_max = C.J_MAX // args.stride
    else:
        delta_s, j_min, j_max = args.delta_s, args.j_min, args.j_max
    stride_suffix = "" if args.stride == 1 else f"__stride{args.stride}"
    wanted = {s.strip() for s in args.only.split(",") if s.strip()}

    man = C.load_manifest(args.manifest)
    if args.restrict_to_features:
        have = set(C.available_feature_ids())
        man = man[man["video_id"].isin(have)]
    man = man.sort_values("video_id", kind="stable").reset_index(drop=True)
    vids = man["video_id"].tolist()
    y = man["class_code"].to_numpy().astype(np.int64)
    post_len = man["post_anchor_length_s"].to_numpy().astype(np.float64)
    n = len(vids)
    js = np.arange(j_min, j_max + 1, dtype=np.int64)
    n_j = len(js)
    post = C.post_anchor_slice(js)

    train_counts = np.bincount(man.loc[man["split"] == "train", "class_code"].to_numpy(),
                               minlength=C.N_CLASSES).astype(np.float64)
    prior = train_counts / max(train_counts.sum(), 1.0)
    majority_class = int(train_counts.argmax())
    print(f"train class counts {train_counts.tolist()}  majority={majority_class} "
          f"({C.CLASS_NAMES[majority_class]})", flush=True)

    specs = []

    preds = np.full((n, n_j), majority_class, dtype=np.int64)
    probs = np.broadcast_to(prior, (n, n_j, C.N_CLASSES)).copy()
    specs.append(("trivial__majority", preds, probs, None,
                  f"constant prediction of the split=train majority class "
                  f"({C.CLASS_NAMES[majority_class]}); probabilities are the train prior"))

    rng = np.random.default_rng(args.seed)
    preds = rng.integers(0, C.N_CLASSES, size=(n, n_j)).astype(np.int64)
    probs = np.full((n, n_j, C.N_CLASSES), 1.0 / C.N_CLASSES)
    specs.append(("trivial__random", preds, probs, None,
                  f"independent uniform draw at every grid point, numpy default_rng seed {args.seed}"))

    preds = np.broadcast_to(y[:, None], (n, n_j)).astype(np.int64).copy()
    probs = np.zeros((n, n_j, C.N_CLASSES))
    probs[np.arange(n)[:, None], np.arange(n_j)[None, :], preds] = 1.0
    specs.append(("trivial__oracle", preds, probs, None,
                  "ground-truth class from delta >= 0 onward; the trivial upper bound"))

    preds = np.full((n, n_j), C.BOT, dtype=np.int64)
    specs.append(("trivial__constbot", preds, None, None,
                  "never answers; every post-anchor step is the bot code and therefore counts wrong"))

    for stem, preds, probs, committed, desc in specs:
        short = stem.split("__")[-1]
        if wanted and short not in wanted:
            continue
        is_random = short == "random"
        sid = f"{stem}{stride_suffix}"
        if args.stride > 1:
            desc = f"{desc}; regenerated at delta_s={delta_s:g}"
        card = {
            "system_id": sid,
            "family": "trivial",
            "description": desc,
            "backbone": "none",
            "trained_on_split": "train (class prior only)" if short == "majority" else "none",
            "delta_s": delta_s,
            "stride_from_finest": args.stride,
            "dev_tuned_params": {},
            "train_data_unknown": False,
            "cost_note": "no video is read",
            "causal": True,
            "parent_system_id": None,
            "seed": args.seed if is_random else None,
            "grid": {"delta_s": delta_s, "j_min": int(js[0]), "j_max": int(js[-1]),
                     "n_j": n_j},
            "pre_anchor_rows": "the same rule as everywhere else; the protocol layer masks "
                               "delta < 0, so no system decides that for itself",
            "subsampling_equivalent": not is_random,
            "subsampling_note": (
                "stateless in the grid index" if not is_random
                else ("draws are per grid point, so a coarser grid draws a different sequence "
                      "and cannot be obtained by sub-sampling this one. "
                      + (f"THIS IS A REGENERATED BUILD at delta_s={delta_s:g}: same rule and same "
                         f"seed {args.seed}, drawn directly on the coarser grid."
                         if args.stride > 1 else
                         "Coarser steps are produced with --stride k."))),
            "uses_ground_truth": short == "oracle",
        }
        answers = C.build_answer_frame(vids, preds, probs, delta_s=delta_s,
                                       committed=committed, j_offset=int(js[0]))
        out = C.write_answers(sid, answers, card)
        te = man["split"] == "test"
        # the proxy readout must use this build's own step, or H is read at the
        # wrong grid index and the coarse builds report the wrong window
        s = devsel.summarize(preds[te.to_numpy()][:, post], y[te.to_numpy()],
                             post_len[te.to_numpy()], delta_s=delta_s)
        print(f"{sid} -> {out}")
        print(devsel.fmt(s, prefix="  test "))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
