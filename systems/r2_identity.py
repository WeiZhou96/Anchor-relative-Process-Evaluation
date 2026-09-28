"""Authoritative metadata adapter for R2 cards; leaves old name parsing unchanged."""
from functools import lru_cache
from pathlib import Path
import yaml
from systems import common as C


@lru_cache(maxsize=None)
def card_for(system_id):
    path=Path(C.ANSWERS_DIR)/system_id/'system_card.yaml'
    card=yaml.safe_load(path.read_text())
    assert card['system_id']==system_id
    for field in ['family','arm_rule','backbone','model_kind','seed','parent_system_id']:
        assert field in card,field
    return card


def grouped_rule(card):
    """Both backbone and base architecture belong in a rule's seed aggregation key."""
    return '__'.join([card['arm_rule'],card['backbone'],card['model_kind']])


def install():
    from ape import systems as old
    originals={name:getattr(old,name) for name in ['family_of','arm_rule','arm_value','parent_of','seed_of']}
    if getattr(old,'_r2_card_adapter_installed',False): return
    def wrap(name):
        def adapted(sid):
            if not str(sid).startswith('r2__'): return originals[name](sid)
            card=card_for(str(sid))
            if name=='family_of': return card['family']
            if name=='arm_rule': return grouped_rule(card)
            if name=='arm_value':
                value=card.get('arm_value')
                return None if value is None else format(value,'g') if isinstance(value,(float,int)) else str(value)
            if name=='parent_of': return card['parent_system_id']
            if name=='seed_of': return str(card['seed'])
        return adapted
    for name in originals: setattr(old,name,wrap(name))
    old._r2_card_adapter_installed=True
