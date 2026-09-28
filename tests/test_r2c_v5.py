"""Frozen 22-s cap, oracle invalidity and seedless VLM integration regressions."""
import copy
import numpy as np
import pytest
from ape.r2b import own_visible_delays
from ape.r2b_identity import metadata
from ape.r2 import seed_consistency
from ape.r2c import CAP, VLM, negative_control, checked_gate
from conftest import make_manifest,make_answers


def vlm_card():
    return dict(system_id=VLM,family='vlm',model_kind='qwen25vl7b',backbone='qwen25vl7b',
                arm_rule='vlm_readonly',stateless=True,subsampling_equivalent=True,train_data_unknown=True)


def test_vlm_seed_is_not_invented_or_parsed(monkeypatch):
    from ape import systems
    monkeypatch.setattr(systems,'seed_of',lambda sid:pytest.fail('legacy parser called'))
    card=vlm_card();original=copy.deepcopy(card)
    m=metadata(VLM,{VLM:card})
    assert m['seed'] is None and m['train_data_unknown'] is True
    assert m['library_round']=='R2' and m['arm_rule']=='vlm_readonly'
    assert card==original and 'seed' not in card


@pytest.mark.parametrize('field',['stateless','subsampling_equivalent'])
def test_nonstateless_vlm_does_not_silently_get_seed_exemption(field):
    c=vlm_card();c[field]=False
    with pytest.raises(ValueError,match='seed'):metadata(VLM,{VLM:c})


def test_seedless_vlm_pair_cannot_carry_seed_replication():
    old=dict(family='prefix',backbone='r18',model_kind='gru128',arm_rule='gru128',seed=20260903)
    row=dict(system_a=VLM,system_b='r2__base',diff=1.,significant=True)
    assert seed_consistency([row],{VLM:vlm_card(),'r2__base':old})==[]
    assert row['cross_base'] is True and row['same_seed'] is False


@pytest.mark.parametrize('length,expected',[ (10.,10.),(21.75,21.75),(22.,22.),(22.25,22.),(80.,22.) ])
def test_v5_never_stable_censors_at_min_length_22(length,expected):
    mf=make_manifest([dict(video_id='x',anchor_s=0,post_s=length)])
    table=make_answers({'x':[1]*89},delta_s=.25)
    values,coverage=own_visible_delays(mf,table,.5,cap=CAP)
    assert values.tolist()==[expected] and coverage['n_missing_visible']==0


def test_v5_removes_235_missing_cell_oracle_artifact():
    mf=make_manifest([dict(video_id='short',anchor_s=0,post_s=12.),dict(video_id='long',anchor_s=0,post_s=40.)])
    table=make_answers({'short':[0]*89,'long':[0]*89},delta_s=.25)
    old,oldcov=own_visible_delays(mf,table,.5,cap=23.5)
    new,newcov=own_visible_delays(mf,table,.5,cap=CAP)
    assert old.tolist()==[0.,23.5] and oldcov['n_missing_visible']==3
    assert new.tolist()==[0.,0.] and newcov['n_missing_visible']==0
    assert not negative_control(old[1]-old[0],0.,.4)
    assert negative_control(new[1]-new[0],0.,.4)


def test_v5_22_second_endpoint_remains_visible():
    mf=make_manifest([dict(video_id='x',anchor_s=0,post_s=50.)])
    good=make_answers({'x':[0]*89},delta_s=.25)
    late_wrong=make_answers({'x':[0]*88+[1]},delta_s=.25)
    assert own_visible_delays(mf,good,.5,cap=CAP)[0][0]==0.
    assert own_visible_delays(mf,late_wrong,.5,cap=CAP)[0][0]==22.


@pytest.mark.parametrize('own,fixed,ruler,want',[
    (0.,0.,.4,True),(.4,-.4,.4,True),(-.4,.4,.4,True),(0.,0.,0.,True),
    (.400001,0.,.4,False),(0.,-.400001,.4,False),(None,0.,.4,False),
    (float('nan'),0.,.4,False),(0.,float('inf'),.4,False),(0.,0.,None,False),(0.,0.,-.1,False)])
def test_oracle_requires_both_absolute_gaps_with_inclusive_boundary(own,fixed,ruler,want):
    assert negative_control(own,fixed,ruler) is want


@pytest.mark.parametrize('real_n,consequence,expected',[(168,84,True),(168,83,False),(169,84,False),(169,85,True)])
def test_v5_pool_specific_half_threshold(real_n,consequence,expected):
    g={'random_block':dict(n=2,n_consequence=2,n_undefined=0),'real':dict(n=real_n,n_consequence=consequence,n_undefined=0)}
    result=checked_gate(g,[{'passed':True}])
    assert result['computation_valid'] and result['pass_gate'] is expected


@pytest.mark.parametrize('controls',[[],[{'passed':False}],[{'passed':True},{'passed':False}]])
def test_oracle_failure_invalidates_computation_not_gate(controls):
    g={'random_block':dict(n=2,n_consequence=2,n_undefined=0),'real':dict(n=168,n_consequence=168,n_undefined=0)}
    assert checked_gate(g,controls)==dict(computation_valid=False,status='invalid_computation',pass_gate=None)


def test_undefined_stratum_does_not_become_failed_gate():
    g={'random_block':dict(n=2,n_consequence=2,n_undefined=0),'real':dict(n=168,n_consequence=100,n_undefined=1)}
    assert checked_gate(g,[{'passed':True}])==dict(computation_valid=True,status='undefined',pass_gate=None)
