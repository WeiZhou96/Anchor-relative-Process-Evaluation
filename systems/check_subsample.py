"""Empirical check of CONTRACT 5.4: does sub-sampling the fine grid equal a rerun?

For each trained system, the answer matrix is regenerated from scratch at a
coarser step and compared column by column with the stride-k sub-sample of the
0.25 s matrix.  This is the B-side half of the A track's E0; it is run on the
real feature cache and real checkpoints, not on synthetic trajectories.

Usage:
    python -X utf8 systems/check_subsample.py --ckpt outputs/checkpoints/<id>.pt --stride 2
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from systems import common as C  # noqa: E402
from systems import infer_answers as IA  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--features-dir", default=C.FEATURES_DIR)
    ap.add_argument("--stride", type=int, default=2)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fine = C.FINEST_DELTA_S
    coarse = fine * args.stride
    k = args.stride

    # The coarse rerun uses the j range that the strided fine grid maps onto, so
    # the two runs cover exactly the same set of end times, j = 0 included.
    j_min_c = -((-C.J_MIN) // k)
    j_max_c = C.J_MAX // k

    sid, _kind, _ck, _man, js_f, ans_fine, _nf, _w = IA.run(
        args.ckpt, args.manifest, args.features_dir, None, fine, C.J_MIN, C.J_MAX, device)
    _sid, _k2, _c2, _m2, js_c, ans_coarse, _nf2, _w2 = IA.run(
        args.ckpt, args.manifest, args.features_dir, None, coarse, j_min_c, j_max_c, device)

    v1, js1, p1, q1 = C.answers_to_arrays(ans_fine)
    v2, js2, p2, q2 = C.answers_to_arrays(ans_coarse)
    assert v1 == v2, "clip order differs between the two runs"

    keep = (js1 % k) == 0
    sub_js = js1[keep] // k
    sub_p, sub_q = p1[:, keep], q1[:, keep]
    common = np.intersect1d(sub_js, js2)
    ia = np.searchsorted(sub_js, common)
    ib = np.searchsorted(js2, common)

    a_p, b_p = sub_p[:, ia], p2[:, ib]
    a_q, b_q = sub_q[:, ia], q2[:, ib]
    same_pred = np.array_equal(a_p, b_p)
    max_dp = float(np.nanmax(np.abs(a_q - b_q))) if a_q.size else float("nan")
    mismatch = int((a_p != b_p).sum())

    print(f"system {sid}  stride {k}  fine delta {fine}  coarse delta {coarse}")
    print(f"aligned grid points: {len(common)} (j from {common.min()} to {common.max()})")
    print(f"compared cells: {a_p.size}  pred mismatches: {mismatch}")
    print(f"max abs probability difference: {max_dp:.3e}")
    ok = same_pred and (np.isnan(max_dp) or max_dp < 1e-6)
    print("SUBSAMPLE_EQUIVALENT" if ok else "SUBSAMPLE_NOT_EQUIVALENT")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
