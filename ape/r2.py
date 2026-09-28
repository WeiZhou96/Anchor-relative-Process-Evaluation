"""CPU-only R2 K-a diagnostics; operational rules are in REPORT_R2_K.md.

No training or system generation. Input identity is pinned to report_index.
All bootstrap samples resample source clusters and keep every clip/prefix.
"""
from pathlib import Path
from functools import lru_cache
from itertools import combinations, product
import hashlib
import json
import re
import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr, pearsonr

from . import calib, stats
from .cli import _dump_json, _needs_rerun_per_step, _rulers_by_metric
from .cohort import select_split, cluster_codes
from .metrics import (AnswerTable, evaluate_system, rmscd_per_video,
                      naive_persistence_to_clip_end, trapezoid, flip_counts)
from .protocol import load_protocol_checked, load_manifest, TOL
from .systems import arm_rule, arm_value, seed_of

METRICS = ['RMSCD@H', 'S_H@1', 'S_H@3', 'end_window_macro_acc', 'median_flips']
KNOBS = ['eps_sys_s', 'eps_jit_sd_s', 'delta_s', 'h_s']
SEEDS = ['20260903', '20260904', '20260905']


def read_json(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def dump(obj, p):
    _dump_json(obj, str(p))


def corr(x, y, kind='spearman'):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return float((spearmanr if kind == 'spearman' else pearsonr)(x, y).statistic)


def bootstrap_weights(codes, n_boot, seed):
    reps = stats.cluster_bootstrap_indices(codes, n_boot, seed)
    w = np.asarray([np.bincount(idx, minlength=len(codes)) for idx in reps], float)
    return reps, w


def family_bootstrap(res, reps, w):
    """Exactly the existing res.frozen_family on shared cluster replicates."""
    den = w.sum(axis=1)
    ind = res.indicator
    samples = {'RMSCD@H': w @ rmscd_per_video(ind, res.delta_s) / den}
    for k in [1, 3]:
        samples[f'S_H@{k}'] = w @ ind[:, int(round(k/res.delta_s))].astype(float) / den
    end = res.correct[:, -1]
    accs = []
    for c in range(5):
        mask = res.y == c
        cden = w @ mask.astype(float)
        num = w @ (end & mask).astype(float)
        accs.append(np.divide(num, cden, out=np.full(len(w), np.nan), where=cden > 0))
    samples['end_window_macro_acc'] = np.nanmean(accs, axis=0)
    flips = flip_counts(res.pred)
    samples['median_flips'] = np.array([np.median(flips[idx]) for idx in reps])
    return samples


def holm_intervals(rows, alpha=.05):
    """Holm step-down normal intervals, plus percentile and Bonferroni CIs.

    Ordered intervals require the step-down reject mask: later narrow intervals
    cannot resurrect a hypothesis after the first Holm failure.
    """
    if not rows:
        return rows
    m = len(rows)
    p = np.array([r['p_value'] if np.isfinite(r['p_value']) else 1. for r in rows])
    correction = stats.holm(p, alpha)
    order = np.argsort(p, kind='stable')
    for rank, i in enumerate(order):
        r = rows[i]
        critical = norm.isf(alpha / (2 * (m-rank)))
        radius = critical * r['se']
        r.update(ci_holm_lo=r['diff']-radius, ci_holm_hi=r['diff']+radius,
                 holm_rank=int(rank+1), holm_local_alpha=float(alpha/(m-rank)),
                 p_adjusted=float(correction['adjusted'][i]))
        br = norm.isf(alpha/(2*m)) * r['se']
        r.update(ci_bonferroni_lo=r['diff']-br, ci_bonferroni_hi=r['diff']+br)
        separated = (r['ci_lo'] > 0 or r['ci_hi'] < 0)
        r['significant'] = bool(correction['reject'][i] and separated and
                                (r['ci_holm_lo'] > 0 or r['ci_holm_hi'] < 0))
    return rows


def diff_record(a, b, points, samples):
    d = np.asarray(samples[a])-np.asarray(samples[b])
    d = d[np.isfinite(d)]
    point = float(points[a]-points[b])
    return dict(system_a=a, system_b=b, diff=point,
                ci_lo=float(np.quantile(d, .025)), ci_hi=float(np.quantile(d, .975)),
                se=float(np.std(d, ddof=1)), p_value=stats._normal_p(d, point))


class Context:
    def __init__(self, root, output_root=None, expected_counts=(64,52,48)):
        self.root = Path(root).resolve()
        self.out = Path(output_root) if output_root is not None else self.root/'outputs'
        self.answers = self.root/'outputs/answers'
        self.cfg = load_protocol_checked(str(self.root/'protocol/pi0.yaml'))
        self.full = load_manifest(str(self.root/'data/manifest/manifest_real.csv'))
        self.test = select_split(self.full, 'test').sort_values('video_id').reset_index(drop=True)
        self.dev = select_split(self.full, 'dev')
        index = read_json(self.out/'report_index.json')['rows']
        self.ids = sorted({r['system_id'] for r in index})
        if len(self.ids) != expected_counts[0] or any('__stride' in s for s in self.ids):
            raise ValueError('Library count or stand-in identity mismatch')
        self.cards = {s: self._card(s) for s in self.ids}
        self.blocks = [s for s in self.ids if self.cards[s].get('family') == 'block']
        self.real = [s for s in self.ids if s not in self.blocks]
        self.nontrivial = [s for s in self.real if self.cards[s].get('family') not in ['trivial','standin']]
        assert (len(self.ids),len(self.real),len(self.nontrivial)) == expected_counts
        assert len(self.blocks) == 12

    def _card(self, sid):
        import yaml
        return yaml.safe_load((self.answers/sid/'system_card.yaml').read_text())

    @lru_cache(maxsize=600)
    def table(self, sid, delta):
        card = self.cards[sid]
        fac = int(round(delta/self.cfg.delta_fine_s))
        deny = str(card.get('subsampling_equivalent', True)).lower() in ['false','0','no']
        name = f'{sid}__stride{fac}' if fac > 1 and deny else sid
        t = AnswerTable.from_dir(str(self.answers/name))
        return t.with_identity(sid, card)

    def eval(self, sid, pv):
        r = evaluate_system(self.test, self.table(sid, pv.delta_s), pv, self.cfg.grid_max_s)
        return r

    def provenance(self):
        return dict(audit_split='test', n_test=len(self.test), n_dev=len(self.dev),
                    n_systems=len(self.ids), n_real=len(self.real), n_nontrivial=len(self.nontrivial), n_blocks=len(self.blocks),
                    bootstrap_n=self.cfg.bootstrap_n, seed=self.cfg.seed, alpha=self.cfg.alpha_pairs,
                    cpu_only=True, source_index_sha256=hashlib.sha256((self.out/'report_index.json').read_bytes()).hexdigest(),
                    methods='REPORT_R2_K.md', system_ids=self.ids)

    def save(self, name, obj, folder='gates'):
        dump(dict(**self.provenance(), **obj), self.out/folder/f'{name}.json')


def root_classifier(sid, cards):
    if str(sid).startswith('r2__') or cards.get(sid,{}).get('backbone') is not None:
        from .r2b_identity import metadata, base_key
        return base_key(metadata(sid,cards))
    seen = set()
    while cards.get(sid, {}).get('parent_system_id'):
        if sid in seen:
            raise ValueError('cycle in parent_system_id')
        seen.add(sid)
        sid = cards[sid]['parent_system_id']
    return re.sub(r'__seed\d+', '', sid)


def variant_id(sid, cards):
    if str(sid).startswith('r2__') or cards[sid].get('backbone') is not None:
        from .r2b_identity import metadata, group_key
        return group_key(metadata(sid,cards))
    base = root_classifier(sid, cards)
    rule = arm_rule(sid)
    value = arm_value(sid) if cards[sid].get('family') == 'commit' else None
    return f'{base}|{rule}|{value or "-"}'


def seed_consistency(rows, cards):
    groups = {}
    for r in rows:
        a, b = r['system_a'], r['system_b']
        r['base_a'], r['base_b'] = root_classifier(a, cards), root_classifier(b, cards)
        r['cross_base'] = r['base_a'] != r['base_b']
        from .r2b_identity import metadata
        r['seed_a'], r['seed_b'] = metadata(a,cards)['seed'], metadata(b,cards)['seed']
        r['same_seed'] = r['seed_a'] == r['seed_b'] and r['seed_a'] in SEEDS
        if not r['same_seed']:
            continue
        va, vb = variant_id(a,cards), variant_id(b,cards)
        if va == vb:
            continue
        key = tuple(sorted([va,vb]))
        orient = 1 if (va,vb) == key else -1
        group = groups.setdefault(key, dict(variant_a=key[0], variant_b=key[1],
                                             cross_base=r['cross_base'], seeds=[]))
        group['seeds'].append(dict(seed=r['seed_a'], system_a=a, system_b=b,
                                  oriented_diff=orient*r['diff'], significant=r['significant']))
    for g in groups.values():
        positive = {r['seed'] for r in g['seeds'] if r['significant'] and r['oriented_diff'] > 0}
        negative = {r['seed'] for r in g['seeds'] if r['significant'] and r['oriented_diff'] < 0}
        g['n_same_direction'] = max(len(positive), len(negative))
        g['pass'] = g['cross_base'] and g['n_same_direction'] >= 2
        g['missing_tied_seeds'] = sorted(set(SEEDS)-{r['seed'] for r in g['seeds']})
    return list(groups.values())


def neighbor_grid(cfg, h):
    ref = cfg.pi0(h)
    axes = cfg._axis_grid(False)
    vals = []
    for k in KNOBS:
        levels = axes[k]; i = levels.index(getattr(ref,k))
        vals.append(levels[max(0,i-1):i+2])
    return [ref.replace(**dict(zip(KNOBS,p))) for p in product(*vals)]


def make_scan(ctx, grid):
    rows=[]
    for i,pv in enumerate(grid):
        for sid in ctx.ids:
            result=ctx.eval(sid,pv)
            for m,v in result.frozen_family(ctx.cfg.s_report_delta_s).items():
                rows.append(dict(system_id=sid,is_block=sid in ctx.blocks,pi_hash=pv.pi_hash,
                                 **{k:getattr(pv,k) for k in KNOBS},in_plaus=True,metric=m,value=v,
                                 missing_lookup_rate=result.missing_lookup_rate,
                                 n_missing_lookup=result.n_missing_lookup,n_cells=result.pred.size))
        print(f'[scan] {i+1}/{len(grid)} {pv.label()}', flush=True)
    return pd.DataFrame(rows)


def calibrate(ctx, scan=None):
    cfg=ctx.cfg
    grids={'plaus':cfg.scan_grid(plaus=True,mode='full')}
    for h in cfg.h_list_s:
        ref=cfg.pi0(h)
        grids[f'axis{h}']=list({p.pi_hash:p for k,vals in cfg._axis_grid(False).items()
                                      for p in [ref.replace(**{k:v}) for v in vals]}.values())
        grids[f'neighbor{h}']=neighbor_grid(cfg,h)
    union={p.pi_hash:p for grid in grids.values() for p in grid}
    if scan is None:
        scan=make_scan(ctx,list(union.values()))
    else:
        expected={(sid,ph,m) for sid in ctx.ids for ph in union for m in METRICS}
        observed=list(zip(scan.system_id,scan.pi_hash,scan.metric))
        if len(observed)!=len(expected) or set(observed)!=expected:
            raise ValueError("Cached scan must exactly cover the frozen systems, grid and metrics")
    (ctx.out/'r2').mkdir(exist_ok=True)
    scan.to_csv(ctx.out/'r2/scan_all.csv',index=False)
    plaus_hashes={p.pi_hash for p in grids['plaus']}
    for h in cfg.h_list_s:
        ref=cfg.pi0(h)
        results={s:ctx.eval(s,ref) for s in ctx.ids}
        first=results[ctx.ids[0]]
        codes=cluster_codes(pd.DataFrame({'source_cluster_id':first.source_cluster_id}))
        reps,w=bootstrap_weights(codes,cfg.bootstrap_n,cfg.seed)
        points={s:results[s].frozen_family(cfg.s_report_delta_s) for s in ctx.ids}
        samples={s:family_bootstrap(results[s],reps,w) for s in ctx.real}
        payload={}
        for m in METRICS:
            pairs,tests=stats.significant_pairs_from_samples({s:points[s][m] for s in ctx.real},
                         {s:samples[s][m] for s in ctx.real},m,cfg.alpha_pairs,'holm','normal')
            payload[m]={'significant_pairs':pairs,'tests':[t.__dict__ for t in tests]}
        end_ci={s:tuple(np.quantile(samples[s]['end_window_macro_acc'],[.025,.975])) for s in ctx.real}
        tied=stats.tied_at_window_end(end_ci)
        rulers=_rulers_by_metric({s:points[s] for s in ctx.real},tied)
        payload.update(pi_hash=ref.pi_hash,pi=ref.as_dict(),n_boot=cfg.bootstrap_n,
                       alpha=cfg.alpha_pairs,correction='holm',p_method='normal',audit_split='test',
                       systems_real=ctx.real,systems_block=ctx.blocks,window_end_tied_pairs=tied,
                       rulers_by_metric=rulers,correction_resolution=stats.correction_resolution_ok(
                           cfg.bootstrap_n,len(ctx.real)*(len(ctx.real)-1)//2,cfg.alpha_pairs))
        dump(payload,ctx.out/'metrics'/ref.pi_hash/'_pairs.json')
        for s in ctx.ids:
            old=read_json(ctx.out/'metrics'/ref.pi_hash/f'{s}.json')
            # The old eval stage already selected stride reruns. Verify equality.
            for m,v in points[s].items():
                if not np.isclose(v,old['frozen_family'][m],rtol=0,atol=1e-12):
                    raise ValueError(f'eval/scan mismatch: {h} {s} {m}')
        cache={f'{s}::{m}':samples[s][m] for s in ctx.real for m in METRICS}
        np.savez_compressed(ctx.out/'r2'/f'bootstrap_{ref.pi_hash}.npz',**cache)
        for mode,folder,key in [('full','calib_plaus','plaus'),('axis','calib',f'axis{h}')]:
            phs={p.pi_hash for p in grids[key]}
            sub=scan[scan.pi_hash.isin(phs)].copy()
            sub['in_plaus']=sub.pi_hash.isin(plaus_hashes)
            out=ctx.out/folder/ref.pi_hash;out.mkdir(parents=True,exist_ok=True)
            sub.to_csv(out/'scan_table.csv',index=False)
            cal=dict(pi0_hash=ref.pi_hash,pi0=ref.as_dict(),audit_split='test',r0=cfg.r0,
                     scan_mode=mode,n_pi=len(phs),n_systems=len(ctx.ids),n_clips_scored=len(ctx.test),
                     p_method='normal',correction='holm',rulers_by_metric=rulers,
                     correction_resolution=payload['correction_resolution'],
                     stride_notes=[],reference_stride_reruns_applied=True,metrics={},delta_star={})
            for m in METRICS:
                for blocks in [False,True]:
                    x=calib.calibrate_metric(sub,ref.pi_hash,m,payload[m]['significant_pairs'],cfg.r0,rulers[m],blocks)
                    bs=x.pop('b_s_table');rt=x.pop('R_table')
                    tag=m.replace('@','at')+('_withblocks' if blocks else '')
                    bs.to_csv(out/f'b_s_{tag}.csv',index=False)
                    rt.to_csv(out/f'R_{tag}.csv',index=False)
                    cal['metrics'][m+(' [with blocks]' if blocks else '')]=x
            for s in ctx.ids:
                fbd={d:ctx.eval(s,ref.replace(delta_s=d)).frozen_family(cfg.s_report_delta_s)['median_flips']
                     for d in cfg._axis_grid(False)['delta_s']}
                cal['delta_star'][s]=calib.delta_star(fbd)
            dump(cal,out/'calibration.json')
        ns=scan[scan.pi_hash.isin({p.pi_hash for p in grids[f'neighbor{h}']})]
        ns.to_csv(ctx.out/'r2'/f'neighbor_{ref.pi_hash}.csv',index=False)
        print(f'[calibration] H={h} complete',flush=True)
    ctx.save('calibration_run',dict(scan_union_n=len(union),n_plaus=54,all_eval_reference_values_verified=True),folder='r2')


def three_strata(values,lengths,cuts):
    labels=np.searchsorted(cuts,np.asarray(lengths),side='left')
    values=np.asarray(values,float)
    means=[float(values[labels==i].mean()) if np.any(labels==i) else None for i in range(3)]
    counts=[int(np.sum(labels==i)) for i in range(3)]
    gap=means[2]-means[0] if counts[0] and counts[2] else None
    return dict(means=means,counts=counts,gap_long_minus_short=gap)


def consequence(own,fixed,mrd):
    if own is None or fixed is None or mrd is None:
        return None
    return bool(abs(own)>mrd and abs(fixed)<mrd)


def g3(ctx):
    cuts=np.quantile(ctx.dev.post_anchor_length_s,[1/3,2/3])
    lengths=ctx.test.post_anchor_length_s.to_numpy()
    allh=[]
    for h in ctx.cfg.h_list_s:
        ref=ctx.cfg.pi0(h)
        cal=read_json(ctx.out/'calib_plaus'/ref.pi_hash/'calibration.json')
        mrd=cal['metrics']['RMSCD@H']['MRD_plaus']
        rows=[]
        for s in ctx.ids:
            res=ctx.eval(s,ref)
            naive=naive_persistence_to_clip_end(ctx.test,ctx.table(s,ref.delta_s),ref,ctx.cfg.grid_max_s)
            t=naive['time_to_stable_per_video']
            own=three_strata(np.where(np.isfinite(t),np.minimum(t,h),h),lengths,cuts)
            fixed_len=ctx.test.set_index('video_id').loc[res.video_ids,'post_anchor_length_s'].to_numpy()
            fixed=three_strata(rmscd_per_video(res.indicator,res.delta_s),fixed_len,cuts)
            role=('random_block' if ctx.cards[s].get('block_family')=='rand' else
                  'other_block' if s in ctx.blocks else 'real' if s in ctx.nontrivial else 'trivial')
            rows.append(dict(system_id=s,role=role,own=own,fixed=fixed,MRD_plaus=mrd,
                             consequence=consequence(own['gap_long_minus_short'],fixed['gap_long_minus_short'],mrd)))
        groups={role:dict(n=sum(r['role']==role for r in rows),
                         n_consequence=sum(r['role']==role and r['consequence'] is True for r in rows),
                         n_undefined=sum(r['role']==role and r['consequence'] is None for r in rows))
                for role in ['random_block','real','trivial','other_block']}
        available=all(groups[k]['n_undefined']==0 for k in ['random_block','real'])
        passed=all(groups[k]['n_consequence']>0 for k in ['random_block','real']) if available else None
        allh.append(dict(h_s=h,pi_hash=ref.pi_hash,MRD_plaus=mrd,groups=groups,rows=rows,pass_gate=passed))
    primary=next(x for x in allh if x['h_s']==ctx.cfg.h_ref_s)
    ctx.save('g3',dict(dev_tercile_cuts_s=cuts,by_h=allh,pass_gate=primary['pass_gate'],
                       primary_h=ctx.cfg.h_ref_s,n_clips_beyond_cached_D2_horizon=int(np.sum(lengths>ctx.cfg.grid_max_s+TOL)),
                       D2_visible_end_cap_s=ctx.cfg.grid_max_s,
                       limitation='Empty strata remain undefined; D2 follows the existing cache-capped convention.'))


def g4(ctx):
    out=[]
    for h in ctx.cfg.h_list_s:
        ref=ctx.cfg.pi0(h)
        pairs=read_json(ctx.out/'metrics'/ref.pi_hash/'_pairs.json')
        boot=np.load(ctx.out/'r2'/f'bootstrap_{ref.pi_hash}.npz')
        points={s:ctx.eval(s,ref).frozen_family(ctx.cfg.s_report_delta_s)['RMSCD@H'] for s in ctx.nontrivial}
        samples={s:boot[f'{s}::RMSCD@H'] for s in ctx.nontrivial}
        tied=[(a,b) for a,b in pairs['window_end_tied_pairs'] if a in points and b in points]
        rows=holm_intervals([diff_record(a,b,points,samples) for a,b in tied])
        for r in rows:
            for side in ['a','b']:
                sid=r[f'system_{side}']
                r[f'end_window_ci_{side}']=np.quantile(boot[f'{sid}::end_window_macro_acc'],[.025,.975]).tolist()
        groups=seed_consistency(rows,ctx.cards)
        out.append(dict(h_s=h,pi_hash=ref.pi_hash,n_tied=len(tied),
                        n_heterogeneous=sum(r['significant'] for r in rows),
                        n_cross_base_heterogeneous=sum(r['significant'] and r['cross_base'] for r in rows),
                        n_replicated_cross_base_groups=sum(g['pass'] for g in groups),
                        pass_gate=any(g['pass'] for g in groups),pairs=rows,seed_groups=groups))
    ctx.save('g4',dict(by_h=out,primary_h=ctx.cfg.h_ref_s,
                       pass_gate=next(x['pass_gate'] for x in out if x['h_s']==ctx.cfg.h_ref_s),
                       CI_method='paired cluster percentile 95%; bootstrap-SE normal Holm step-down intervals and reject mask; Bonferroni companion'))


def reversal_counts(scan,ref,metric,pairs):
    mat=scan[scan.metric==metric].pivot(index='system_id',columns='pi_hash',values='value')
    pairs=list(pairs)
    ia=mat.index.get_indexer([a for a,b in pairs]);ib=mat.index.get_indexer([b for a,b in pairs])
    if np.any(ia<0) or np.any(ib<0): raise KeyError('pair identity absent from scan')
    values=mat.to_numpy(dtype=float)
    base=values[:,mat.columns.get_loc(ref)]
    d0=base[ia]-base[ib]
    rows=[];seen=set();tie_seen=set()
    for j,ph in enumerate(mat.columns):
        d=values[ia,j]-values[ib,j]
        valid=np.isfinite(d0)&np.isfinite(d)&(d0!=0)
        rev=[list(pairs[i]) for i in np.flatnonzero(valid&(d0*d<0))]
        ties=[list(pairs[i]) for i in np.flatnonzero(valid&(d==0))]
        seen.update(map(tuple,rev));tie_seen.update(map(tuple,ties))
        rows.append(dict(pi_hash=ph,n_pairs=int(valid.sum()),reversals=len(rev),new_ties=len(ties),reversal_pairs=rev,tie_pairs=ties))
    return dict(reversal_events=sum(r['reversals'] for r in rows),unique_reversed_pairs=len(seen),
                tie_events=sum(r['new_ties'] for r in rows),unique_tied_pairs=len(tie_seen),by_pi=rows)


def a1(ctx):
    out=[]
    for h in ctx.cfg.h_list_s:
        ref=ctx.cfg.pi0(h);scan=pd.read_csv(ctx.out/'r2'/f'neighbor_{ref.pi_hash}.csv')
        ps=read_json(ctx.out/'metrics'/ref.pi_hash/'_pairs.json')
        for m in METRICS:
            pairs=ps[m]['significant_pairs']
            out.append(dict(h_s=h,metric=m,n_pi=int(scan.pi_hash.nunique()),
                      significant=reversal_counts(scan,ref.pi_hash,m,pairs),
                      nontrivial_significant=reversal_counts(scan,ref.pi_hash,m,[(a,b) for a,b in pairs if a in ctx.nontrivial and b in ctx.nontrivial]),
                      all_pairs=reversal_counts(scan,ref.pi_hash,m,list(combinations(ctx.real,2)))))
    ctx.save('a1',dict(rows=out,definition='strict sign reversals in immediate-neighbor product; new ties separate'),folder='ablations')


def dynamic_accuracy_summary(ctx,sid,ref,w):
    """Instantaneous micro accuracy with a varying risk set; no E_H or suffix AND."""
    t=ctx.table(sid,ref.delta_s)
    offsets=np.arange(int(round(ref.h_s/ref.delta_s))+1)*ref.delta_s
    js=np.broadcast_to(np.rint(offsets/t.delta_s).astype(int),(len(ctx.test),len(offsets)))
    pred,present,_,_=t.lookup_2d(ctx.test.video_id,js)
    alive=offsets[None,:]<=ctx.test.post_anchor_length_s.to_numpy()[:,None]+TOL
    if np.any(alive & ~present): raise ValueError('dynamic accuracy has missing cache cells')
    correct=(pred==ctx.test.class_code.to_numpy()[:,None]) & alive
    den=alive.sum(axis=0)
    curve=np.divide(correct.sum(axis=0),den,out=np.full(len(den),np.nan),where=den>0)
    bd=w@alive.astype(float);bn=w@correct.astype(float)
    curves=np.divide(bn,bd,out=np.full_like(bn,np.nan),where=bd>0)
    return dict(curve=curve,risk_set=den,point=float(trapezoid(curve,ref.delta_s)/ref.h_s),
                samples=trapezoid(curves,ref.delta_s,axis=1)/ref.h_s)


def a3(ctx):
    g=read_json(ctx.out/'gates/g4.json')
    codes=cluster_codes(ctx.test)
    _,w=bootstrap_weights(codes,ctx.cfg.bootstrap_n,ctx.cfg.seed)
    out=[]
    for gh in g['by_h']:
        h=gh['h_s'];ref=ctx.cfg.pi0(h)
        values={s:dynamic_accuracy_summary(ctx,s,ref,w) for s in ctx.nontrivial}
        points={s:d['point'] for s,d in values.items()};samples={s:d['samples'] for s,d in values.items()}
        rows=holm_intervals([diff_record(r['system_a'],r['system_b'],points,samples) for r in gh['pairs']])
        for r,old in zip(rows,gh['pairs']):
            r['g4_heterogeneous']=old['significant'];r['cross_base']=old['cross_base']
            r['auc_a']=points[r['system_a']];r['auc_b']=points[r['system_b']]
        n=sum(r['g4_heterogeneous'] for r in rows)
        detected=sum(r['g4_heterogeneous'] and r['significant'] for r in rows)
        out.append(dict(h_s=h,n_tied=len(rows),n_g4_heterogeneous=n,n_detected=detected,n_missed=n-detected,
                        detection_fraction=detected/n if n else None,pairs=rows,
                        systems={s:{k:v for k,v in d.items() if k!='samples'} for s,d in values.items()}))
    ctx.save('a3',dict(by_h=out,statistic='AUC of dynamic-denominator instantaneous micro accuracy / H',
                       limitation='Failure to separate this predeclared scalar does not establish whole-curve equivalence.'),folder='ablations')


def rank_ablation(ctx,name):
    out=[]
    for h in ctx.cfg.h_list_s:
        ref=ctx.cfg.pi0(h);d=ctx.out/'calib_plaus'/ref.pi_hash
        scan=pd.read_csv(d/'scan_table.csv');scan=scan[scan.system_id.isin(ctx.real)]
        for m in METRICS:
            rt=pd.read_csv(d/f'R_{m.replace("@","at")}.csv')
            if name=='a4':
                mat=scan[scan.metric==m].pivot(index='system_id',columns='pi_hash',values='value')
                alternate={ph:corr(mat[ref.pi_hash],mat[ph]) for ph in mat.columns}
            else:
                alt=calib.rank_preservation(scan,ref.pi_hash,m,list(combinations(ctx.real,2)))
                alternate=dict(zip(alt.pi_hash,alt.R_M))
            rows=[]
            for r in rt.to_dict('records'):
                alt=alternate[r['pi_hash']]
                rows.append(dict(pi_hash=r['pi_hash'],R_M=r['R_M'],alternative=alt,
                                 in_R=bool(r['R_M']>=ctx.cfg.r0),
                                 in_alternative=bool(alt is not None and alt>=ctx.cfg.r0)))
            finite=[r['alternative'] for r in rows if r['alternative'] is not None and np.isfinite(r['alternative'])]
            out.append(dict(h_s=h,metric=m,n_pi=len(rows),min_R=min(r['R_M'] for r in rows),
                        min_alternative=min(finite) if finite else None,
                        n_R=sum(r['in_R'] for r in rows),n_alternative=sum(r['in_alternative'] for r in rows),
                        added=sum(r['in_alternative'] and not r['in_R'] for r in rows),
                        removed=sum(r['in_R'] and not r['in_alternative'] for r in rows),
                        intersection=sum(r['in_R'] and r['in_alternative'] for r in rows),
                        ruler_degenerate=m=='median_flips',by_pi=rows))
    ctx.save(name,dict(rows=out,alternative='Spearman average ranks' if name=='a4' else 'all reference-nontied pairs without significance screening'),folder='ablations')


def axis_sub(scan,ref,knob):
    mask=np.ones(len(scan),bool)
    for k in KNOBS:
        if k!=knob: mask &= np.isclose(scan[k].to_numpy(),getattr(ref,k),atol=1e-9,rtol=0)
    return scan[mask]


def mechanisms(ctx):
    scan=pd.read_csv(ctx.out/'r2/scan_all.csv')
    pb=[];pc=[]
    for h in ctx.cfg.h_list_s:
        ref=ctx.cfg.pi0(h)
        jitter=axis_sub(scan,ref,'eps_jit_sd_s')
        for m in METRICS:
            bs=calib.offsets_and_spread(jitter[jitter.system_id.isin(ctx.nontrivial)],ref.pi_hash,m).sort_values('eps_jit_sd_s')
            pb.append(dict(h_s=h,metric=m,T_b=corr(bs.eps_jit_sd_s,bs.s_M),
                           points=bs[['eps_jit_sd_s','b_M','s_M']].to_dict('records')))
        for s in ctx.ids:
            per={}
            for d in [.25,.5,1.]:
                r=ctx.eval(s,ref.replace(delta_s=d));f=flip_counts(r.pred)
                per[d]={'median':float(np.median(f)),'mean':float(np.mean(f))}
            fbd={d:v['median'] for d,v in per.items()}
            pc.append(dict(h_s=h,system_id=s,role='real' if s in ctx.nontrivial else 'block' if s in ctx.blocks else 'trivial',
                           T_c=(fbd[.25]-fbd[.5])/max(abs(fbd[.25]),1),
                           finest_mean_increment=per[.25]['mean']-per[.5]['mean'],
                           points=per,**calib.delta_star(fbd)))
    ctx.save('p_b',dict(rows=pb,statistic='Spearman(jitter SD, across-nontrivial-system SD of metric shifts)'),folder='mechanisms')
    ctx.save('p_c',dict(rows=pc,statistic='(median flips at .25 - at .5)/max(abs(finest median),1)',
                       tolerance_from_existing_delta_star=.05),folder='mechanisms')


def g5(ctx):
    scan=pd.read_csv(ctx.out/'r2/scan_all.csv')
    rows=[]
    families=sorted({ctx.cards[s]['block_family'] for s in ctx.blocks})
    for h in ctx.cfg.h_list_s:
        ref=ctx.cfg.pi0(h)
        for m in METRICS:
            for knob in KNOBS:
                sub=axis_sub(scan,ref,knob)
                real=calib.offsets_and_spread(sub[sub.system_id.isin(ctx.nontrivial)],ref.pi_hash,m).set_index('pi_hash')
                for family in ['all_blocks']+families:
                    ids=ctx.blocks if family=='all_blocks' else [s for s in ctx.blocks if ctx.cards[s]['block_family']==family]
                    block=calib.offsets_and_spread(sub[sub.system_id.isin(ids)],ref.pi_hash,m,blocks=True).set_index('pi_hash')
                    joined=real[[knob,'b_M']].join(block[['b_M']],rsuffix='_block').sort_values(knob)
                    nonref=joined.index!=ref.pi_hash
                    a=joined.loc[nonref,'b_M'].to_numpy();b=joined.loc[nonref,'b_M_block'].to_numpy()
                    # 1e-12 is a numerical-zero tolerance, not an acceptance threshold.
                    sa=np.where(np.abs(a)<=1e-12,0,np.sign(a));sb=np.where(np.abs(b)<=1e-12,0,np.sign(b))
                    active=(sa!=0)|(sb!=0)
                    rows.append(dict(h_s=h,metric=m,axis=knob,block_family=family,
                        pearson=corr(joined.b_M,joined.b_M_block,'pearson'),spearman=corr(joined.b_M,joined.b_M_block),
                        sign_agreement=float(np.mean(sa==sb)),n_nonreference=len(a),
                        n_both_zero=int(np.sum((sa==0)&(sb==0))),
                        active_sign_agreement=float(np.mean(sa[active]==sb[active])) if np.any(active) else None,
                        curve=joined.reset_index().to_dict('records')))
    ctx.save('g5',dict(rows=rows,pass_gate=None,verdict='descriptive_only_no_preregistered_threshold',
                       n_family_axis_metric_h_rows=len(rows)))


def run(root,task):
    if (Path(root)/"outputs/r2c/report_index.json").exists():
        from .r2c import run as run_c
        return run_c(root,task)
    index=Path(root)/'outputs/report_index.json'
    if index.exists() and read_json(index).get('outputs_root')=='r2b':
        from .r2b import run as run_b
        return run_b(root,task)
    ctx=Context(root)
    tasks={'calibrate':calibrate,'g3':g3,'g4':g4,'a1':a1,'a3':a3,
           'a4':lambda c:rank_ablation(c,'a4'),'a5':lambda c:rank_ablation(c,'a5'),
           'mechanisms':mechanisms,'g5':g5}
    if task=='all':
        for name,fn in tasks.items():
            print(f'[stage] {name}',flush=True)
            fn(ctx)
    else:
        tasks[task](ctx)
