"""R2 adversarial checks: independent statistics, empty strata, identity, and ties."""
from types import SimpleNamespace
from itertools import combinations
import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm
from ape.r2 import (bootstrap_weights, family_bootstrap, holm_intervals, three_strata,
                    consequence, neighbor_grid, reversal_counts, root_classifier,
                    seed_consistency, dynamic_accuracy_summary, corr)
from ape.protocol import ProtocolVector
from ape.metrics import evaluate_system
from ape.cohort import cluster_codes
from conftest import make_manifest, make_answers


def test_fast_bootstrap_matches_original_with_multiclip_multiclass_cluster():
    mf=make_manifest([dict(video_id='a',anchor_s=0,post_s=4,y=0,cluster='shared'),
                      dict(video_id='b',anchor_s=0,post_s=4,y=1,cluster='shared'),
                      dict(video_id='c',anchor_s=0,post_s=4,y=2,cluster='other')])
    at=make_answers({'a':[0,1,0,0,0,0,0],'b':[1,1,0,1,1,1,1],'c':[0,0,0,2,2,2,2]})
    result=evaluate_system(mf,at,ProtocolVector(h_s=3,delta_s=.5),3)
    reps,w=bootstrap_weights(cluster_codes(mf),1000,20260903)
    got=family_bootstrap(result,reps,w)
    expected=[result.frozen_family([1,3],idx) for idx in reps]
    for key in got:
        np.testing.assert_allclose(got[key],[e[key] for e in expected],rtol=0,atol=1e-12)
    assert np.all(w[:,0]==w[:,1])


@pytest.mark.parametrize('own,fixed,mrd,want',[(1,.1,.2,True),(-1,-.1,.2,True),
    (.2,.1,.2,False),(1,.2,.2,False),(.1,.3,.2,False),(None,0,.2,None),(1,None,.2,None)])
def test_g3_absolute_and_strict_thresholds(own,fixed,mrd,want):
    assert consequence(own,fixed,mrd) is want


def test_g3_empty_short_layer_is_not_zero_or_pass():
    x=three_strata([2,3],[12,22],[5,15])
    assert x['counts']==[0,1,1] and x['gap_long_minus_short'] is None


def test_g3_boundary_belongs_to_lower_stratum():
    x=three_strata([1,2,3,4],[5,10,15,20],[5,15])
    assert x['counts']==[1,2,1]
    assert x['gap_long_minus_short']==3


def test_holm_does_not_resurrect_late_narrow_interval():
    rows=[]
    for p in [.03,.04]:
        d=float(norm.isf(p/2))
        rows.append(dict(diff=d,se=1.,p_value=p,ci_lo=d-1.96,ci_hi=d+1.96))
    rows=holm_intervals(rows)
    assert not any(r['significant'] for r in rows)
    assert rows[1]['ci_holm_lo']>0  # local interval alone would falsely reject


def test_holm_strong_effect_and_zero_variance():
    rows=[dict(diff=1.,se=0.,p_value=0.,ci_lo=1.,ci_hi=1.),
          dict(diff=0.,se=0.,p_value=1.,ci_lo=0.,ci_hi=0.)]
    out=holm_intervals(rows)
    assert out[0]['significant'] and not out[1]['significant']


def test_holm_nan_p_never_rejects():
    r=dict(diff=2.,se=np.nan,p_value=np.nan,ci_lo=1.,ci_hi=3.)
    assert not holm_intervals([r])[0]['significant']


def test_neighbor_is_immediate_not_plaus(cfg):
    points=neighbor_grid(cfg,10)
    assert len(points)==54
    assert {p.eps_jit_sd_s for p in points}=={0,.1}
    assert {p.eps_sys_s for p in points}=={-.25,0,.25}
    assert len(neighbor_grid(cfg,4))==36


def test_a1_strict_reversal_new_tie_and_base_tie_separate():
    rows=[]
    for ph,vals in [('ref',[0,1,1]),('rev',[2,1,1]),('tie',[1,1,1])]:
        rows.extend(dict(system_id=s,pi_hash=ph,metric='M',value=v) for s,v in zip('abc',vals))
    x=reversal_counts(pd.DataFrame(rows),'ref','M',list(combinations('abc',2)))
    assert x['reversal_events']==2 and x['unique_reversed_pairs']==2
    assert x['tie_events']==2 and all(r['n_pairs']==2 for r in x['by_pi'])


def cards_and_rows(base_b='prefix__gru512',signs=(1,1,-1)):
    cards={};rows=[]
    for seed,sign in zip(['20260903','20260904','20260905'],signs):
        a=f'clip__r18mean__seed{seed}';base=f'{base_b}__seed{seed}'
        b=f'postproc__ema{seed[-1]}__{base}'
        cards[a]={'family':'clip'};cards[base]={'family':'prefix'}
        cards[b]={'family':'postproc','parent_system_id':base}
        rows.append(dict(system_a=a,system_b=b,diff=sign,significant=True))
    return cards,rows


def test_g4_two_of_three_and_dev_parameter_identity():
    cards,rows=cards_and_rows()
    groups=seed_consistency(rows,cards)
    assert len(groups)==1 and groups[0]['pass'] and groups[0]['n_same_direction']==2


def test_g4_same_base_does_not_pass():
    cards,rows=cards_and_rows(base_b='clip__r18mean')
    assert not any(g['pass'] for g in seed_consistency(rows,cards))


def test_g4_one_significant_seed_not_enough():
    cards,rows=cards_and_rows()
    rows[1]['significant']=False;rows[2]['significant']=False
    assert not any(g['pass'] for g in seed_consistency(rows,cards))


def test_g4_cross_seed_pairs_not_replication():
    cards,rows=cards_and_rows()
    rows[0]['system_b']=rows[1]['system_b'];rows[1]['system_b']=rows[2]['system_b']
    assert not any(g['pass'] for g in seed_consistency(rows,cards))


def test_root_identity_follows_multiple_parents():
    cards={'a':{'parent_system_id':'b'},'b':{'parent_system_id':'prefix__gru512__seed20260903'}}
    assert root_classifier('a',cards)=='prefix__gru512'


def test_root_cycle_rejected():
    with pytest.raises(ValueError): root_classifier('a',{'a':{'parent_system_id':'a'}})


def test_a3_dynamic_risk_and_instantaneous_correctness():
    mf=make_manifest([dict(video_id='a',anchor_s=0,post_s=1),dict(video_id='b',anchor_s=0,post_s=2)])
    table=make_answers({'a':[0,0,1],'b':[0,1,0]},delta_s=1)
    ctx=SimpleNamespace(test=mf,table=lambda s,d:table)
    result=dynamic_accuracy_summary(ctx,'x',ProtocolVector(h_s=2,delta_s=1),np.eye(2))
    np.testing.assert_array_equal(result['risk_set'],[2,2,1])
    np.testing.assert_allclose(result['curve'],[1,.5,1])
    assert result['point']==.75


@pytest.mark.parametrize('kind',['pearson','spearman'])
def test_constant_response_is_undefined(kind):
    assert corr([0,0,0],[1,2,3],kind) is None


def test_signed_response_correlation():
    assert corr([0,1,2],[0,-1,-2])==-1


def test_scan_reference_uses_coarse_rerun_for_pair_differences(tmp_path,monkeypatch):
    """Regression: scan ref values and P must consume the same stride matrix."""
    import json
    from ape import cli
    from ape.protocol import ProtocolConfig
    cfg=ProtocolConfig(dict(delta_s=.5,h_list_s=[3.],grid_max_s=3.,bootstrap_n=20,
         answers={'delta_fine_s':.25},perturb={'delta_s':[.25,.5]},
         metric_family={'s_report_delta_s':[1.,3.]}))
    mf=make_manifest([dict(video_id=v,anchor_s=0,post_s=4) for v in ['a','b']])
    fine=make_answers({v:[1]*13 for v in ['a','b']},delta_s=.25,system_id='A',family='prefix')
    fine.card['subsampling_equivalent']=False
    coarse=make_answers({v:[0]*7 for v in ['a','b']},delta_s=.5,system_id='A__stride2',family='prefix')
    other=make_answers({v:[1]*13 for v in ['a','b']},delta_s=.25,system_id='B',family='prefix')
    monkeypatch.setattr(cli,'load_protocol_checked',lambda p:cfg)
    monkeypatch.setattr(cli,'_load_audit_manifest',lambda args,cfg:(mf,'test'))
    monkeypatch.setattr(cli,'_load_answer_tables',lambda spec:[fine,coarse,other])
    args=SimpleNamespace(pi='unused',manifest='unused',answers='unused',out=str(tmp_path),
                         bootstrap_n=None,plaus=False,mode='axis')
    assert cli.cmd_scan(args)==0
    cal=json.loads((tmp_path/cfg.pi0().pi_hash/'calibration.json').read_text())
    assert cal['metrics']['RMSCD@H']['n_pairs']==1
    scan=pd.read_csv(tmp_path/cfg.pi0().pi_hash/'scan_table.csv')
    ref=scan[(scan.pi_hash==cfg.pi0().pi_hash)&(scan.metric=='RMSCD@H')].set_index('system_id')
    assert ref.at['A','value']==0 and ref.at['B','value']==3
