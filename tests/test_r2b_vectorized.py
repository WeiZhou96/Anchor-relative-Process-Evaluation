"""Differential tests against the exact pre-vectorization implementation."""
import numpy as np
import pandas as pd
from itertools import combinations
from ape.r2 import reversal_counts
from ape.calib import rank_preservation,_pivot,_pi_meta,_sign
from typing import Sequence,Tuple
def legacy_reversal_counts(scan,ref,metric,pairs):
    mat=scan[scan.metric==metric].pivot(index='system_id',columns='pi_hash',values='value')
    rows=[];seen=set();tie_seen=set()
    for ph in sorted(mat.columns):
        rev=[];ties=[];den=0
        for a,b in pairs:
            d0=mat.at[a,ref]-mat.at[b,ref]
            d=mat.at[a,ph]-mat.at[b,ph]
            if not np.isfinite(d0) or not np.isfinite(d) or d0==0: continue
            den+=1
            if d0*d<0: rev.append([a,b]);seen.add((a,b))
            if d==0: ties.append([a,b]);tie_seen.add((a,b))
        rows.append(dict(pi_hash=ph,n_pairs=den,reversals=len(rev),new_ties=len(ties),reversal_pairs=rev,tie_pairs=ties))
    return dict(reversal_events=sum(r['reversals'] for r in rows),unique_reversed_pairs=len(seen),
                tie_events=sum(r['new_ties'] for r in rows),unique_tied_pairs=len(tie_seen),by_pi=rows)


def legacy_rank_preservation(
    scan: pd.DataFrame,
    pi0_hash: str,
    metric: str,
    pairs: Sequence[Tuple[str, str]],
    blocks: bool = False,
) -> pd.DataFrame:
    """``R_M(pi)``: share of the pairs in ``P`` whose ordering survives ``pi``.

    A pair that becomes an exact tie under ``pi`` counts as **not** preserved:
    the reader's action is a comparison, and a tie does not support one.
    """
    sub = scan if blocks else scan.loc[~scan["is_block"].astype(bool)]
    mat = _pivot(sub, metric)
    meta = _pi_meta(scan)
    if pi0_hash not in mat.columns:
        raise KeyError(f"pi0 hash {pi0_hash} absent from the scan table")
    rows = []
    for pi_hash in mat.columns:
        kept, total = 0, 0
        for a, b in pairs:
            if a not in mat.index or b not in mat.index:
                continue
            s0 = _sign(float(mat.at[a, pi0_hash]) - float(mat.at[b, pi0_hash]))
            s1 = _sign(float(mat.at[a, pi_hash]) - float(mat.at[b, pi_hash]))
            if s0 == 0:
                continue  # not a separated pair at the reference protocol
            total += 1
            kept += int(s1 == s0)
        m = meta.loc[pi_hash]
        rows.append(
            {
                "pi_hash": pi_hash,
                "eps_sys_s": float(m["eps_sys_s"]),
                "eps_jit_sd_s": float(m["eps_jit_sd_s"]),
                "delta_s": float(m["delta_s"]),
                "h_s": float(m["h_s"]),
                "in_plaus": bool(m["in_plaus"]),
                "metric": metric,
                "n_pairs": int(total),
                "R_M": float(kept) / total if total else float("nan"),
            }
        )
    return pd.DataFrame(rows)

def sample_scan():
    rng=np.random.default_rng(20260903);rows=[]
    for j,ph in enumerate(['ref','p1','p2','p3','p4']):
        values=rng.choice([0.,1.,-1.,np.nan,1e-200,-1e-200],size=12)
        for i,v in enumerate(values):
            rows.append(dict(system_id=f's{i}',pi_hash=ph,metric='M',value=v,is_block=i>9,
                eps_sys_s=float(j),eps_jit_sd_s=0.,delta_s=.5,h_s=10.,in_plaus=True))
    return pd.DataFrame(rows)

def test_vectorized_reversals_exactly_match_original_including_nonfinite_and_underflow():
    scan=sample_scan();pairs=list(combinations(sorted(scan.system_id.unique()),2))
    assert reversal_counts(scan,'ref','M',pairs)==legacy_reversal_counts(scan,'ref','M',pairs)

def test_vectorized_rank_matches_original_with_missing_ids_blocks_ties_and_empty_pairs():
    scan=sample_scan();pairs=list(combinations(sorted(scan.system_id.unique()),2))+[('absent','s1')]
    for blocks in [False,True]:
        for ps in [pairs,[]]:
            expected=legacy_rank_preservation(scan,'ref','M',ps,blocks)
            actual=rank_preservation(scan,'ref','M',ps,blocks)
            pd.testing.assert_frame_equal(actual,expected)
