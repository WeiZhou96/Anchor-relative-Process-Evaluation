"""Fixed eligibility cohorts and source-video-cluster utilities.

Implements idea section 3.2 and contract section 8.

The cohort is fixed *before* any metric is computed: for a horizon ``H``,
``E_H = {i : L_i^+ >= H}`` where ``L_i^+`` is the post-anchor length under the
*effective* (possibly perturbed) anchor. Every system and every grid point
``delta_j`` uses that same set, which is what makes the points of one ``S_H``
curve comparable to each other. Videos that never stabilise stay in the
denominator; nothing is dropped (contract section 8).
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .classes import CLASS_NAMES, N_CLASSES
from .protocol import TOL, ProtocolVector, perturb_manifest

LENGTH_COL_EFF = "post_anchor_length_s_eff"


def _length_column(manifest: pd.DataFrame) -> str:
    if LENGTH_COL_EFF in manifest.columns:
        return LENGTH_COL_EFF
    return "post_anchor_length_s"


#: The only split an audit number may be computed on.
AUDIT_SPLIT = "test"


def select_split(
    manifest: pd.DataFrame, split: str = AUDIT_SPLIT, split_col: str = "split"
) -> pd.DataFrame:
    """Restrict the manifest to one split before any cohort is formed.

    The audit set is the frozen ``test`` split. ``train`` and ``dev`` were used
    to choose the horizons, the step candidates, the perturbation bounds and the
    system hyper-parameters, so a number computed on them is not an audit
    number -- it is a number about the data the protocol was tuned on. The idea's
    discipline is explicit that the audit set never participates before the
    freeze and that everything before the freeze happens on the development
    subset; this function is the other half of that promise.

    ``split='all'`` is available for diagnostics only and every output stamped
    with it says so.
    """
    if split in (None, "all"):
        return manifest.reset_index(drop=True)
    if split_col not in manifest.columns:
        raise ValueError(
            f"manifest has no {split_col!r} column; cannot restrict to split "
            f"{split!r}. An audit must not silently run on every clip."
        )
    sub = manifest.loc[manifest[split_col].astype(str) == str(split)]
    if len(sub) == 0:
        available = sorted(manifest[split_col].astype(str).unique())
        raise ValueError(f"split {split!r} is empty; available: {available}")
    return sub.reset_index(drop=True)


def split_summary(manifest: pd.DataFrame, split_col: str = "split") -> Dict[str, int]:
    if split_col not in manifest.columns:
        return {}
    return {
        str(k): int(v)
        for k, v in manifest[split_col].astype(str).value_counts().items()
    }


def eligible_mask(manifest: pd.DataFrame, h_s: float) -> np.ndarray:
    """Boolean mask of ``E_H`` over the rows of ``manifest``.

    The comparison carries a ``TOL`` (1e-9 s) slack rather than being a bare
    ``>= H``. This is not laxity, it is what keeps the cohort a property of the
    video rather than of binary floating point. Worked example from manifest v2:
    clip ``KUBbn-T3XYI_00`` has ``anchor_s = 12.08`` and ``duration_s = 22.08``,
    so it holds exactly 10.0 s of post-anchor video and belongs in ``E_10``; but
    ``22.08 - 12.08`` evaluates to ``9.999999999999998``, and a strict ``>=``
    would drop it. One clip either way is immaterial to any result; a cohort that
    changes when the manifest is regenerated with slightly different arithmetic
    is not. The tolerance is far below any real timing resolution (the fastest
    source runs at 50 fps, i.e. 0.02 s per frame), so it can never admit a clip
    that is genuinely short.
    """
    col = _length_column(manifest)
    return manifest[col].to_numpy(dtype=float) >= (float(h_s) - TOL)


def cohort(manifest: pd.DataFrame, h_s: float) -> pd.DataFrame:
    """Rows of ``manifest`` forming ``E_H``, ordered by ``video_id``."""
    sub = manifest.loc[eligible_mask(manifest, h_s)]
    return sub.sort_values("video_id").reset_index(drop=True)


def cohort_for_pi(
    manifest: pd.DataFrame, pv: ProtocolVector
) -> pd.DataFrame:
    """Perturb the anchors per ``pi`` and return the resulting ``E_H``."""
    return cohort(perturb_manifest(manifest, pv), pv.h_s)


def cohort_summary(manifest: pd.DataFrame, h_s: float) -> Dict[str, object]:
    """``N_H``, class distribution and cluster count for one horizon.

    Every reported curve must carry these (idea section 3.2 item 5).
    """
    sub = cohort(manifest, h_s)
    counts = np.zeros(N_CLASSES, dtype=int)
    if len(sub):
        vc = sub["class_code"].value_counts()
        for code, cnt in vc.items():
            code = int(code)
            if 0 <= code < N_CLASSES:
                counts[code] = int(cnt)
    total = int(counts.sum())
    return {
        "h_s": float(h_s),
        "N_H": int(len(sub)),
        "n_clusters": int(sub["source_cluster_id"].nunique()) if len(sub) else 0,
        "class_counts": {CLASS_NAMES[c]: int(counts[c]) for c in range(N_CLASSES)},
        "class_fractions": {
            CLASS_NAMES[c]: (float(counts[c]) / total if total else float("nan"))
            for c in range(N_CLASSES)
        },
        "attrition_from_all": int(len(manifest) - len(sub)),
    }


def cohort_table(manifest: pd.DataFrame, h_list_s: Sequence[float]) -> pd.DataFrame:
    """One row per horizon: the cohort sizes that every table must report."""
    return pd.DataFrame([cohort_summary(manifest, h) for h in h_list_s])


# --------------------------------------------------------------------------
# source video clusters
# --------------------------------------------------------------------------
# Terminology: a cluster groups clips cut from one *source video*, which is not
# the same as grouping them by accident. Measured on manifest v2 (track C's
# clusters.py): 54 of the 1789 clusters carry more than one collision type
# (37 with two, 15 with three, 2 with four; train 31 / dev 6 / test 17), i.e. one
# camera filmed several different crashes. The cluster is therefore the unit of
# *sampling dependence* (shared footage, shared scene, possible near-duplicate
# frames), not a claim that its members are one event, and class-conditional
# curves are computed from each clip's own label rather than a cluster majority.
# Calling it a "source event cluster" would assert the stronger, false thing; the
# column name ``source_cluster_id`` is unchanged.
_TRAILING_NUMERIC = re.compile(r"_[0-9]{1,3}$")


def derive_source_cluster_id(video_id: str) -> str:
    """Source video cluster of a clip id (contract section 3, revised rule).

    Strip trailing ``_<1..3 digits>`` segments **repeatedly** until none remains,
    so that clips cut from one source video land in one cluster whatever their
    nesting depth::

        3Xd6PxEvbNk_11_00 -> 3Xd6PxEvbNk
        3Xd6PxEvbNk_00    -> 3Xd6PxEvbNk

    Stripping only one segment would split those two apart and let the same crash
    appear on both sides of a split, which is the leak this rule exists to close.

    This is a *derivation*, not the authority: the manifest's own
    ``source_cluster_id`` column is what the rest of the package uses. Use
    :func:`check_cluster_rule` to confirm the two agree.
    """
    out = str(video_id)
    if "." in out.rsplit("/", 1)[-1]:
        out = out.rsplit(".", 1)[0]
    while True:
        stripped = _TRAILING_NUMERIC.sub("", out)
        if stripped == out or not stripped:
            return out
        out = stripped


def check_cluster_rule(manifest: pd.DataFrame) -> pd.DataFrame:
    """Rows whose ``source_cluster_id`` disagrees with the derivation rule.

    An empty frame means the manifest and this package agree on what a source
    event is. A non-empty one must be resolved before any bootstrap is reported,
    because the statistical unit would otherwise be undefined.
    """
    derived = manifest["video_id"].astype(str).map(derive_source_cluster_id)
    bad = derived.to_numpy() != manifest["source_cluster_id"].astype(str).to_numpy()
    return pd.DataFrame(
        {
            "video_id": manifest.loc[bad, "video_id"].to_numpy(),
            "source_cluster_id": manifest.loc[bad, "source_cluster_id"].to_numpy(),
            "derived": derived.to_numpy()[bad],
        }
    )


def cluster_codes(manifest: pd.DataFrame) -> np.ndarray:
    """Integer cluster code per row, for the cluster-level bootstrap."""
    return pd.factorize(manifest["source_cluster_id"].astype(str))[0].astype(np.int64)


def cluster_sizes(manifest: pd.DataFrame) -> pd.Series:
    return manifest.groupby("source_cluster_id", observed=True).size().sort_values(
        ascending=False
    )


def split_by_cluster(
    manifest: pd.DataFrame,
    frac: float,
    seed: int,
    from_split: Optional[str] = None,
    split_col: str = "split",
) -> np.ndarray:
    """Carve a development subset out of a split, keeping clusters intact.

    Whole source video clusters move together (idea section 3.2 item 3), so no
    two clips cut from the same source video can straddle the boundary. Returns
    a boolean mask over the rows of ``manifest`` marking the development subset.
    """
    if from_split is not None and split_col in manifest.columns:
        pool = manifest.loc[manifest[split_col].astype(str) == str(from_split)]
    else:
        pool = manifest
    clusters = sorted(pool["source_cluster_id"].astype(str).unique().tolist())
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(clusters))
    target = int(round(float(frac) * len(pool)))
    chosen: List[str] = []
    got = 0
    sizes = pool["source_cluster_id"].astype(str).value_counts().to_dict()
    for idx in order:
        if got >= target:
            break
        c = clusters[int(idx)]
        chosen.append(c)
        got += int(sizes.get(c, 0))
    chosen_set = set(chosen)
    return (
        manifest["source_cluster_id"].astype(str).isin(chosen_set).to_numpy()
        & manifest.index.isin(pool.index)
    )


def check_cluster_disjoint(manifest: pd.DataFrame, split_col: str = "split") -> pd.DataFrame:
    """Report clusters that appear in more than one split (must be empty)."""
    if split_col not in manifest.columns:
        return pd.DataFrame(columns=["source_cluster_id", "splits"])
    g = manifest.groupby("source_cluster_id", observed=True)[split_col].agg(
        lambda s: sorted(set(map(str, s)))
    )
    bad = g[g.map(len) > 1]
    return pd.DataFrame({"source_cluster_id": bad.index, "splits": bad.values})
