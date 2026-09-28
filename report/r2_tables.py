"""Render R2 gate and obligation-control evidence without recomputing decisions."""
import json
from pathlib import Path


def load(root,folder,name):
    p=Path(root)/folder/f'{name}.json'
    return json.loads(p.read_text()) if p.exists() else {}


def g3_rows(root,h):
    data=load(root,'gates','g3')
    for d in data.get('by_h',[]):
        if d['h_s']==h:
            revised=next((x for x in data.get('revised_rule',{}).get('by_h',[]) if x['h_s']==h),{})
            lookup={r['system_id']:r for r in revised.get('rows',[])}
            return {r['system_id']:dict(r,revised=lookup.get(r['system_id'],{})) for r in d['rows']}
    return {}


def render(root,out,write,fmt):
    """The caller's CSV/Markdown writer is shared with the five original tables."""
    def table(name,header,rows,notes=None):
        return write(out,name,header,rows,name.replace('_',' '),
                     ['统计单位为源视频簇；test仅计算；冻结bootstrap=1000。完整库规则见 REPORT_R2_Kb.md；旧库定义见 REPORT_R2_K.md。']+(notes or []))
    data=load(root,'gates','g3')
    if data:
        rows=[]
        for h in data['by_h']:
            for r in h['rows']:
                rows.append([h['h_s'],r['system_id'],r['role'],*r['own']['counts'],
                             *[fmt(x,6) for x in r['own']['means']],*r['fixed']['counts'],
                             *[fmt(x,6) for x in r['fixed']['means']],
                             fmt(r['own']['gap_long_minus_short'],6),fmt(r['fixed']['gap_long_minus_short'],6),
                             fmt(r['MRD_plaus'],6),fmt(r['consequence'])])
        table('tableS_G3_strata',['H','System','Role','Own N1','Own N2','Own N3','Own mean1','Own mean2','Own mean3',
              'Fixed N1','Fixed N2','Fixed N3','Fixed mean1','Fixed mean2','Fixed mean3','Own gap','Fixed gap','MRD','Consequence'],rows,
              [f"dev三分位切点（秒）：{data['dev_tercile_cuts_s']}。空层记n/a，不改分层。",
               'D2沿用缓存21.5秒可见片尾，H处删失；未观测更晚作答，不作外推。'])
    data=load(root,'gates','g4')
    if data:
        table('tableS_G4_summary',['H','Tied pairs','Heterogeneous','Cross-base heterogeneous','Replicated groups','Pass'],
              [[h['h_s'],h['n_tied'],h['n_heterogeneous'],h['n_cross_base_heterogeneous'],h['n_replicated_cross_base_groups'],fmt(h['pass_gate'])] for h in data['by_h']])
        rows=[]
        for h in data['by_h']:
            for r in h['pairs']:
                rows.append([h['h_s'],r['system_a'],r['system_b'],fmt(r['diff'],6),fmt(r['ci_lo'],6),fmt(r['ci_hi'],6),
                             fmt(r['ci_holm_lo'],6),fmt(r['ci_holm_hi'],6),r['p_adjusted'],r['significant'],r['cross_base'],r['same_seed']])
        table('tableS_G4_pairs',['H','System a','System b','Diff','95% lo','95% hi','Holm lo','Holm hi','Holm p','Heterogeneous','Cross-base','Same seed'],rows,
              ['Holm区间与完整step-down拒绝标志联合读取；不将后序单个区间排除0单独视为拒绝。',
               '窗末区间重叠是操作化同分标准，不是统计等价性证明。'])
    data=load(root,'ablations','a1')
    if data:
        table('tableS_A1_reversals',['H','Metric','Neighbor points','P reversal events','P unique pairs','P new ties','Nontrivial events','All-pair events'],
              [[r['h_s'],r['metric'],r['n_pi'],r['significant']['reversal_events'],r['significant']['unique_reversed_pairs'],
                r['significant']['tie_events'],r['nontrivial_significant']['reversal_events'],r['all_pairs']['reversal_events']] for r in data['rows']],
              ['相邻档指完整冻结轴直接邻居；抖动为0/0.1，非plaus的0/0.25。严格符号翻转与新增同分分开。'])
    data=load(root,'ablations','a3')
    if data:
        table('tableS_A3_prefix_accuracy',['H','G4 heterogeneous','AUC detected','AUC missed','Detection fraction'],
              [[r['h_s'],r['n_g4_heterogeneous'],r['n_detected'],r['n_missed'],fmt(r['detection_fraction'],6)] for r in data['by_h']],
              ['瞬时准确率使用动态风险集，统计量为AUC/H；该标量未显著不代表整条曲线等价。'])
        table('tableS_A3_pairs',['H','System a','System b','AUC a','AUC b','AUC diff','95% lo','95% hi','Holm p','G4 heterogeneous','AUC distinguishable'],
              [[h['h_s'],r['system_a'],r['system_b'],fmt(r['auc_a'],6),fmt(r['auc_b'],6),fmt(r['diff'],6),
                fmt(r['ci_lo'],6),fmt(r['ci_hi'],6),r['p_adjusted'],r['g4_heterogeneous'],r['significant']] for h in data['by_h'] for r in h['pairs']])
    for name in ['a4','a5']:
        data=load(root,'ablations',name)
        if data:
            table(f'tableS_{name.upper()}_region',['H','Metric','Min R','Min alternative','R domain','Alternative domain','Added','Removed','Intersection','Ruler degenerate'],
                  [[r['h_s'],r['metric'],fmt(r['min_R'],6),fmt(r['min_alternative'],6),r['n_R'],r['n_alternative'],r['added'],r['removed'],r['intersection'],r['ruler_degenerate']] for r in data['rows']],
                  [data['alternative'],'r0保持0.9；所有域仅指有限54点扫描，不外推连续参数域。'])
    data=load(root,'mechanisms','p_b')
    if data:
        table('tableS_Pb_jitter',['H','Metric','T_b','SD at 0','SD at .1','SD at .25','SD at .5'],
              [[r['h_s'],r['metric'],fmt(r['T_b'],6),*[fmt(p['s_M'],6) for p in r['points']]] for r in data['rows']],
              ['T_b=Spearman(抖动标准差,真实非平凡系统偏移SD)，常数响应记n/a。'])
    data=load(root,'mechanisms','p_c')
    if data:
        table('tableS_Pc_stride',['H','System','Role','T_c','Median F .25','Median F .5','Median F 1','Finest mean increment','Delta star','Saturated'],
              [[r['h_s'],r['system_id'],r['role'],fmt(r['T_c'],6),*[r['points'][k]['median'] for k in ['0.25','0.5','1.0']],
                fmt(r['finest_mean_increment'],6),r['delta_star'],r['saturated']] for r in data['rows']],
              ['5%判据沿用已有delta_star；中位数零可能掩盖少数片段仍有变化，均值增量同时报告。'])
    data=load(root,'gates','g5')
    if data:
        table('tableS_G5_response_shape',['H','Metric','Axis','Block family','Pearson','Spearman','Sign agreement','Active sign agreement','Both-zero points'],
              [[r['h_s'],r['metric'],r['axis'],r['block_family'],fmt(r['pearson'],6),fmt(r['spearman'],6),fmt(r['sign_agreement'],6),
                fmt(r['active_sign_agreement'],6),r['n_both_zero']] for r in data['rows']],
              ['无预注册G5通过阈值；描述性报告。符号一致率排除ref，常数曲线相关记n/a。'])

    data=load(root,'gates','g2')
    if data:
        table('tableS_G2_two_arms',['H','Metric','Min R','MRD plaus','Ruler','R arm','MRD arm','Excluded trivial dependence'],
            [[h['h_s'],x['metric'],fmt(x['min_R'],6),fmt(x['MRD_plaus'],6),fmt(x['ruler'],6),fmt(x['R_arm']),fmt(x['MRD_arm']),x['excluded_trivial_dependence']] for h in data['by_h'] for x in h['rows']],
            ['仅H=10决定G2；F(Delta)不计入门，RMSCD的MRD不跨H。'])
    data=load(root,'gates','g3')
    if data.get('revised_rule'):
        by=data['revised_rule']['by_h']
        table('tableS_G3_revised',['H','System','Role','N short','N middle','N long','Own gap','Own CI lo','Own CI hi','Fixed gap','Fixed CI lo','Fixed CI hi','MRD','Ruler','Consequence','Missing visible cells'],
            [[h['h_s'],x['system_id'],x['role'],*x['own']['counts'],*[fmt(x['own'][k],6) for k in ['gap_long_minus_short','ci_lo','ci_hi']],*[fmt(x['fixed'][k],6) for k in ['gap_long_minus_short','ci_lo','ci_hi']],fmt(x['MRD_plaus'],6),fmt(x['ruler'],6),fmt(x['consequence']),x['coverage']['n_missing_visible']] for h in by for x in h['rows']],
            ['同一E_H、dev合格集合切点；own在min(L+,23.5)处删失，fixed为RMSCD积分。缺失可见作答计错；两口径共用簇bootstrap。'])
        table('tableS_G3_two_rules',['H','Original random / 2','Original real / 168','Original pass','Revised random / 2','Revised real / 168','Revised pass','Dev cuts'],
            [[a['h_s'],a['groups']['random_block']['n_consequence'],a['groups']['real']['n_consequence'],fmt(a['pass_gate']),b['groups']['random_block']['n_consequence'],b['groups']['real']['n_consequence'],fmt(b['pass_gate']),str(b['dev_tercile_cuts_s'])] for a,b in zip(data['by_h'],by)])
