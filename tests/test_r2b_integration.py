"""v4 adversarial tests for metadata, inferential gates and record-keeping."""
import sys
from pathlib import Path
import numpy as np
import pytest
from ape.r2b_identity import metadata,base_key,group_key
from ape.r2b import g2_metric,revised_consequence,bootstrap_gap,own_visible_delays
from ape.r2 import seed_consistency,bootstrap_weights
from conftest import make_manifest,make_answers

def card(backbone='r18',model='gru512',rule='ema',value=.2,seed=20260903):
    return dict(family='postproc',backbone=backbone,model_kind=model,arm_rule=rule,arm_value=value,
                seed=seed,parent_system_id=None,dev_tuned_params={})

def test_r2_identity_comes_from_card_not_id():
    c=card('clipb16','gru128','teaser_ocsvm_adapt')
    x=metadata('r2__misleading__mean',{'r2__misleading__mean':c})
    assert (x['backbone'],x['model_kind'],x['arm_rule'])==('clipb16','gru128','teaser_ocsvm_adapt')

@pytest.mark.parametrize('missing',['backbone','model_kind','arm_rule','seed'])
def test_missing_r2_identity_fails(missing):
    c=card();del c[missing]
    with pytest.raises(ValueError): metadata('r2__a',{'r2__a':c})

def test_r2_never_falls_back_to_legacy_parser():
    with pytest.raises(ValueError): metadata('r2__foo',{'r2__foo':{'family':'prefix'}})

def test_old_card_inherits_architecture_seed_from_root():
    cards={'base':dict(family='prefix',backbone='resnet18-imagenet',seed=20260903,dev_tuned_params={'hidden':512}),
           'arm':dict(family='postproc',backbone='none',parent_system_id='base',dev_tuned_params={'arm':'ema','value':.7})}
    x=metadata('arm',cards)
    assert (x['backbone'],x['model_kind'],x['arm_rule'],x['seed'])==('r18','gru512','ema','20260903')

def test_dev_parameter_not_a_group_dimension():
    a=metadata('r2__a',{'r2__a':card(value=.2)});b=metadata('r2__b',{'r2__b':card(value=.7,seed=20260904)})
    assert group_key(a)==group_key(b)

def test_threshold_is_a_group_dimension():
    a=card(rule='commit_msp');a['dev_tuned_params']={'threshold':.5}
    b=card(rule='commit_msp');b['dev_tuned_params']={'threshold':.7}
    assert group_key(metadata('r2__a',{'r2__a':a}))!=group_key(metadata('r2__b',{'r2__b':b}))

@pytest.mark.parametrize('dimension,value',[('library_round','R1'),('backbone','clipb16'),('model_kind','gru128'),('model_kind','mean')])
def test_cross_base_round_backbone_model(dimension,value):
    a=metadata('r2__a',{'r2__a':card()});b=dict(a);b[dimension]=value
    assert base_key(a)!=base_key(b)

def test_same_classifier_two_postprocess_arms_not_cross_base():
    a=metadata('r2__a',{'r2__a':card()});b=metadata('r2__b',{'r2__b':card(rule='patience')})
    assert base_key(a)==base_key(b) and group_key(a)!=group_key(b)

def test_r2_seed_replication_ignores_dev_parameters():
    cards={};rows=[]
    for seed in [20260903,20260904,20260905]:
        a=f'r2__a{seed}';b=f'r2__b{seed}'
        cards[a]=card(seed=seed,value=seed%10);cards[b]=card('clipb16',seed=seed,value=seed%3)
        rows.append(dict(system_a=a,system_b=b,diff=1.,significant=seed!=20260904))
    groups=seed_consistency(rows,cards)
    assert len(groups)==1 and groups[0]['pass'] and groups[0]['n_same_direction']==2

@pytest.mark.parametrize('mr,minr,expected',[(.1,.91,False),(.2,.91,True),(.1,.89,True)])
def test_g2_two_arms_inclusive_mrd_strict_rank(mr,minr,expected):
    x=g2_metric('S_H@3',minr,mr,.2)
    assert (x['R_arm'] or x['MRD_arm']) is expected

def test_g2_flips_cannot_carry_gate_with_degenerate_ruler():
    x=g2_metric('median_flips',.1,2.,0.)
    assert x['R_arm'] is None and x['MRD_arm'] is None and x['ruler_degenerate']

def gap(point,lo,hi): return dict(gap_long_minus_short=point,ci_lo=lo,ci_hi=hi)

@pytest.mark.parametrize('own,fixed,want',[
 (gap(1,.1,2),gap(.3,-.1,.7),True),
 (gap(1,.1,2),gap(.1,.05,.15),True),
 (gap(1,-.1,2),gap(.1,-.1,.2),False),
 (gap(.1,.01,.2),gap(0,-.1,.1),False),
 (gap(1,.1,2),gap(.2,.1,.3),False),
 (gap(-1,-2,-.1),gap(-.1,-.15,-.05),True),
 (gap(None,None,None),gap(0,-.1,.1),None),
])
def test_g3_revised_ci_or_ruler_and_strict_inequalities(own,fixed,want):
    assert revised_consequence(own,fixed,.1,.2) is want

def test_paired_cluster_gap_retains_all_members_and_matches_manual():
    values=np.array([1.,2.,3.,8.]);lengths=np.array([1.,1.,2.,3.])
    _,w=bootstrap_weights(np.array([0,0,1,2]),1000,20260903)
    result=bootstrap_gap(values,lengths,[1,2],w)
    assert result['gap_long_minus_short']==6.5
    assert result['ci_lo']==result['ci_hi']==6.5
    assert np.array_equal(w[:,0],w[:,1])

def test_g3_own_censor_is_visible_end_not_H_or_next_grid_point():
    mf=make_manifest([dict(video_id='a',anchor_s=0,post_s=1.2),dict(video_id='b',anchor_s=0,post_s=4.)])
    table=make_answers({'a':[1]*7,'b':[1]*7},delta_s=.5)
    values,coverage=own_visible_delays(mf,table,.5,cap=3.)
    np.testing.assert_allclose(values,[1.2,3.])
    assert coverage['n_missing_visible']==0

def test_g3_missing_visible_cache_is_wrong_and_counted():
    mf=make_manifest([dict(video_id='a',anchor_s=0,post_s=4.)])
    table=make_answers({'a':[0,0,0]},delta_s=.5)
    values,coverage=own_visible_delays(mf,table,.5,cap=3.)
    assert values[0]==3. and coverage['n_missing_visible']==4

def test_report_grouping_requires_three_unique_seeds():
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'report'))
    from make_tables import group_by_rule
    recs=[dict(group_key='R2|clipb16|gru128|ema|None',system_id=str(s),seed=str(s),family='postproc') for s in [20260903,20260904,20260905]]
    complete,partial=group_by_rule(recs)
    assert len(complete)==1 and not partial
    with pytest.raises(ValueError): group_by_rule(recs+[recs[0]])

def test_legacy_task_dispatches_to_full_library(tmp_path,monkeypatch):
    import json
    from ape import r2,r2b
    (tmp_path/'outputs').mkdir()
    (tmp_path/'outputs/report_index.json').write_text(json.dumps({'outputs_root':'r2b'}))
    calls=[]
    monkeypatch.setattr(r2b,'run',lambda root,task:calls.append((root,task)))
    r2.run(tmp_path,'g3')
    assert calls==[(tmp_path,'g3')]

@pytest.mark.parametrize('fields,rule',[
    ({'uses_ground_truth':True},'trivial_oracle'),
    ({'description':'never answers; every cell wrong'},'trivial_constbot')])
def test_trivial_metadata_is_independent_of_renamed_system_id(fields,rule):
    c=dict(family='trivial',seed=None,**fields)
    x=metadata('renamed_without_identity_tokens',{'renamed_without_identity_tokens':c})
    assert x['arm_rule']==rule and x['seed'] is None

def test_seedless_block_does_not_invoke_name_parser(monkeypatch):
    from ape import systems
    monkeypatch.setattr(systems,'seed_of',lambda sid:pytest.fail('name parser invoked'))
    x=metadata('arbitrary',{'arbitrary':dict(family='block',block_family='rand')})
    assert x['seed'] is None and x['arm_rule']=='block_rand'
