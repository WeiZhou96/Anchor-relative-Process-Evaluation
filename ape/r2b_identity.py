"""Card-derived merged-library identity; no R2 ID token parsing."""
from . import systems as legacy

def metadata(sid, cards):
    card=cards[sid]; root=sid; seen=set()
    while cards.get(root,{}).get('parent_system_id'):
        if root in seen: raise ValueError('cycle in parent_system_id')
        seen.add(root);root=cards[root]['parent_system_id']
    base=cards.get(root,{})
    is_r2='model_kind' in card or 'freeze_commit' in card
    if str(sid).startswith('r2__') and not is_r2:
        raise ValueError('R2 identity requires an authoritative card')
    dev=card.get('dev_tuned_params') or {}
    if is_r2:
        is_vlm=card.get('family')=='vlm' and card.get('stateless') is True and card.get('subsampling_equivalent') is True
        for key in ['backbone','model_kind','arm_rule']+([] if is_vlm else ['seed']):
            if key not in card: raise ValueError('missing R2 card field: '+key)
        backbone=card['backbone'];model=card['model_kind'];rule=card['arm_rule']
        seed=None if is_vlm else str(card['seed']);value=card.get('arm_value')
    else:
        # Older cards store architecture and rule in structured dev fields or backbone.
        backbone='r18' if 'resnet18' in str(base.get('backbone','')).lower() else base.get('backbone') or 'none'
        hidden=(base.get('dev_tuned_params') or {}).get('hidden')
        model='mean' if base.get('family')=='clip' else ('gru'+str(hidden) if hidden else base.get('family','unknown'))
        fam=card.get('family')
        rule=(dev.get('arm') if fam=='postproc' else 'commit_'+dev['rule'] if fam=='commit' and dev.get('rule') else
              'mean_linear' if fam=='clip' else model if fam=='prefix' else 'block_'+card['block_family'] if fam=='block' else trivial_rule(card) if fam=='trivial' else fam)
        value=dev.get('value',dev.get('threshold'))
        seed=card.get('seed',base.get('seed'))
        seed=str(seed) if seed is not None else None
        # Compatibility for historical minimal unit-test cards only. Production
        # cards carry seed (possibly null) directly or at the parent root.
        if fam in ['clip','prefix','postproc','commit'] and 'seed' not in card and 'seed' not in base: seed=legacy.seed_of(sid)
    threshold=dev.get('threshold') if rule in ['commit_msp','commit_margin'] else None
    return dict(backbone=backbone,model_kind=model,arm_rule=rule,arm_value=value,seed=seed,
                commit_threshold=threshold,library_round='R2' if is_r2 else 'R1',
                train_data_unknown=bool(card.get('train_data_unknown',False)),parent_system_id=card.get('parent_system_id'))

def base_key(meta):
    return '|'.join(str(meta[k]) for k in ['library_round','backbone','model_kind'])

def group_key(meta):
    # Round distinguishes the old unbalanced and newly trained R18 classifiers.
    return '|'.join(str(meta[k]) for k in ['library_round','backbone','model_kind','arm_rule','commit_threshold'])

def trivial_rule(card):
    if card.get('uses_ground_truth'):return 'trivial_oracle'
    if card.get('trained_on_split')=='train (class prior only)':return 'trivial_majority'
    if card.get('subsampling_equivalent') is False:return 'trivial_random'
    if card.get('description','').startswith('never answers;'):return 'trivial_constbot'
    raise ValueError('Unrecognized trivial card semantics')
