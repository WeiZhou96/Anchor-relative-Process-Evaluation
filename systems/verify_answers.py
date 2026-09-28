"""Acceptance check on every answer matrix this track owns.

Verifies the CONTRACT 5.4 format and the A track's grid requirements:
columns and order, the j range, a single delta_s, prediction codes in
{-1, 0..4}, probability rows that either sum to one or are entirely NaN,
monotone ``committed``, and rectangularity over the manifest's clips.

Usage: python -X utf8 systems/verify_answers.py [--only-mine]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from systems import common as C  # noqa: E402

FAILS = []


def check(sid: str, name: str, cond: bool, detail: str = "") -> None:
    if not cond:
        FAILS.append(f"{sid}: {name} {detail}")
        print(f"  [FAIL] {name} {detail}")


def expected_j_range(delta_s: float):
    """The j range a system must cover at its own step, given the frozen finest grid.

    A coarse build covers the same wall-clock span with fewer points, so the bound
    scales with the stride rather than staying at the finest grid's -11..88.
    """
    k = delta_s / C.FINEST_DELTA_S
    if abs(k - round(k)) > 1e-9:
        return None
    k = int(round(k))
    return -((-C.J_MIN) // k), C.J_MAX // k


def verify(sid: str, path: str, man: pd.DataFrame) -> None:
    df = pd.read_csv(path)
    print(f"- {sid}: rows={len(df)}")
    check(sid, "columns match CONTRACT 5.4", list(df.columns) == C.ANSWER_COLUMNS,
          f"got {list(df.columns)}")
    if list(df.columns) != C.ANSWER_COLUMNS:
        return

    js = sorted(df["j"].unique().tolist())
    check(sid, "j is contiguous", js == list(range(js[0], js[-1] + 1)))
    check(sid, "single delta_s", df["delta_s"].nunique() == 1)
    delta_s = float(df["delta_s"].iloc[0])
    exp = expected_j_range(delta_s)
    check(sid, "delta_s is a multiple of the finest step", exp is not None, f"delta_s={delta_s}")
    if exp is not None:
        check(sid, f"j covers the required range at delta_s={delta_s:g}",
              js[0] <= exp[0] and js[-1] >= exp[1], f"got {js[0]}..{js[-1]}, need {exp[0]}..{exp[1]}")
    check(sid, "j includes 0", 0 in js)
    check(sid, "grid reaches at least anchor + 22 s",
          js[-1] * delta_s >= C.GRID_MAX_S - 1e-9, f"got {js[-1] * delta_s}")

    n_v = df["video_id"].nunique()
    check(sid, "table is rectangular", len(df) == n_v * len(js),
          f"{len(df)} vs {n_v}*{len(js)}")
    check(sid, "clips are a subset of the manifest",
          set(df["video_id"]).issubset(set(man["video_id"])))

    preds = df["pred"].to_numpy()
    check(sid, "pred codes are in {-1, 0..4}",
          bool(np.isin(preds, [-1, 0, 1, 2, 3, 4]).all()),
          f"unexpected {sorted(set(preds.tolist()) - {-1, 0, 1, 2, 3, 4})}")

    P = df[[f"p{k}" for k in range(C.N_CLASSES)]].to_numpy()
    all_nan = np.isnan(P).all(axis=1)
    none_nan = ~np.isnan(P).any(axis=1)
    check(sid, "probability rows are all-NaN or fully populated",
          bool((all_nan | none_nan).all()))
    if none_nan.any():
        s = P[none_nan].sum(axis=1)
        check(sid, "populated probability rows sum to 1",
              bool(np.abs(s - 1.0).max() < 1e-6), f"max dev {np.abs(s - 1.0).max():.2e}")
        check(sid, "probabilities are in [0,1]",
              bool((P[none_nan] >= -1e-9).all() and (P[none_nan] <= 1 + 1e-9).all()))
    # a populated row whose argmax disagrees with pred is only allowed for the
    # commitment arms, whose pred is frozen at the crossing rather than per step
    card_path = os.path.join(os.path.dirname(path), "system_card.yaml")
    card = {}
    if os.path.exists(card_path):
        with open(card_path, "r", encoding="utf-8") as fh:
            card = yaml.safe_load(fh) or {}
    check(sid, "card declares a family", bool(card.get("family")))
    check(sid, "card declares causal", card.get("causal") is True)

    wide = df.sort_values(["video_id", "j"], kind="stable")
    com = wide["committed"].to_numpy().astype(bool).reshape(n_v, len(js))
    viol = int((com[:, :-1] & ~com[:, 1:]).sum())
    check(sid, "committed is monotone non-decreasing", viol == 0, f"{viol} violations")
    if card.get("family") != "commit":
        check(sid, "non-commitment system has committed all False", bool((~com).all()))
    else:
        pre = np.array(js) < 0
        check(sid, "commitment never fires before the anchor", bool((~com[:, pre]).all()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--answers-dir", default=C.ANSWERS_DIR)
    ap.add_argument("--only-mine", action="store_true",
                    help="skip the A track's block__* directories")
    args = ap.parse_args()

    man = C.load_manifest(args.manifest)
    sids = sorted(d for d in os.listdir(args.answers_dir)
                  if os.path.isdir(os.path.join(args.answers_dir, d)))
    n = 0
    for sid in sids:
        if args.only_mine and sid.startswith("block__"):
            continue
        p = os.path.join(args.answers_dir, sid, "answers.csv")
        if not os.path.exists(p):
            continue
        verify(sid, p, man)
        n += 1

    print()
    if FAILS:
        print(f"VERIFY FAILED on {len(FAILS)} check(s):")
        for f in FAILS:
            print("  " + f)
        return 1
    print(f"VERIFY OK: {n} answer matrices conform")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
