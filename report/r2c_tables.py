"""Final K-c tables: immutable primary numbers and explicit VLM secondary columns."""
from pathlib import Path
import csv
import json
import shutil
import ape_io as io
import r2_tables

VLM='r2__qwen25vl7b__readonly'
RULES=['original_rule','v4_23p5','v5_22p0']
LABELS=['R2-8','A4-2 23.5 s (historical)','A5-1 22.0 s (record)']


def available_root(root):
    root=Path(root)
    candidate=root if root.name=='r2c' else root/'r2c'
    return candidate if (candidate/'gates/g3_v5.json').is_file() else None


def read(p):return json.loads(Path(p).read_text())


def horizon(rule,h):return next(x for x in rule['by_h'] if x['h_s']==h)


def calibration_rows(primary,secondary,fmt):
    headers=['Metric']
    fields=[('min_R','Min R'),('MRD_plaus','MRD plaus'),('ruler','Ruler'),('n_pairs','P'),
            ('max_b','Max b'),('max_s','Max s'),('eps_max','eps max'),('reference_range','Reference range'),('verdict','Verdict')]
    for _,name in fields:headers.extend([name+' [primary -VLM]',name+' [secondary +VLM]'])
    rows=[]
    for metric in ['RMSCD@H','S_H@1','S_H@3','end_window_macro_acc','median_flips']:
        row=[metric]
        for key,_ in fields:
            for cal in [primary,secondary]:
                val=cal['metrics'][metric].get(key)
                row.append(json.dumps(val,sort_keys=True) if isinstance(val,(dict,list)) else fmt(val,6))
        rows.append(row)
    return headers,rows


def render(root,args,mt):
    root=Path(root);primary=root.parent/'r2b';out=Path(args.out_dir) if args.out_dir else root/'tables'
    out.mkdir(parents=True,exist_ok=True)
    protocols=io.reference_protocols(str(root),str(root/'metrics'))
    if args.pi_hash:protocols=[x for x in protocols if x[0]==args.pi_hash]
    write=mt.write_table;fmt=mt.fmt
    notes=['主列不含VLM，逐值沿用K-b；副列含只读VLM，池级统计重新计算。',
           'test仅用于冻结审计；VLM为确定性无种子系统，train_data_unknown=true，无粗步长目录。',
           'G3以v5的22.0秒口径为记录值；23.5秒历史计算含缓存伪效应，不能作为证据。']
    # Capture the existing table builders without changing any primary input tree.
    def capture(fn,*a,**kw):
        items=[];old=mt.write_table
        mt.write_table=lambda *x:items.append(x)
        try:fn(*a,**kw)
        finally:mt.write_table=old
        assert len(items)==1
        return items[0]
    g3=read(root/'gates/g3_v5.json')
    for ph,h in protocols:
        base=io.load_metrics(str(primary/'metrics'),ph);full=io.load_metrics(str(root/'metrics'),ph)
        pairs=io.load_pairs(str(root/'metrics'),ph)
        vlm=next(x for x in full if x['system_id']==VLM)
        for suffix in ([''] if h==10. else [])+['_'+mt.htag(h)]:
            a=capture(mt.table1,base,io.load_pairs(str(primary/'metrics'),ph),h,ph,out,suffix)
            b=capture(mt.table1,full,pairs,h,ph,out,suffix)
            lookup={r[1]:r for r in a[3]}
            header=b[2][:-2]+[x+' [primary -VLM]' for x in b[2][-2:]]+[x+' [secondary +VLM]' for x in b[2][-2:]]
            rows=[r[:-2]+(lookup[r[1]][-2:] if r[1] in lookup else ['n/a','n/a'])+r[-2:] for r in b[3]]
            write(out,b[1],header,rows,b[4],b[5]+notes)
            b=capture(mt.table1_by_rule,base,h,ph,out,suffix)
            rows=list(b[3]);commit=vlm['commit']
            rows.append([vlm['group_key'],*[vlm.get(k,'-') for k in ['backbone','model_kind','arm_rule','commit_threshold','library_round']],
                'vlm','n/a (deterministic)','n/a',*[fmt(vlm.get(k),6) for k in ['full_clip_macro_acc','end_window_macro_acc','RMSCD','flips_median']],
                *[fmt(commit.get(k),6) for k in ['rho','tau_c','e_c']]])
            write(out,b[1],b[2],rows,b[4],b[5]+notes+['VLM单独列点估计，无种子标准差；不把它记为缺失种子的训练组。'])
            # Per-system phenomenon values do not depend on the pool.
            mt.table5(full,h,ph,out,suffix)
            mt.table5_by_rule(base,h,ph,out,suffix)
            # Retain the old accounting columns, explicitly naming the rank pool.
            a=capture(mt.table2,base,h,ph,out,suffix,use_rule=args.conclusion_rule)
            b=capture(mt.table2,full,h,ph,out,suffix,use_rule=args.conclusion_rule)
            baseline={r[0]:r for r in a[3]}
            header=b[2][:7]+['Conclusion stable? [primary -VLM]','Conclusion stable? [secondary +VLM]']
            keys=['own gap','fixed gap','MRD','consequence']
            for label in LABELS:
                for key in keys:header.extend([label+' '+key+' [-VLM]',label+' '+key+' [+VLM]'])
            header+=['22 s computation valid [-VLM]','22 s computation valid [+VLM]']
            pools={p:{rule:horizon(g3[p][rule],h) for rule in RULES} for p in ['primary','secondary']}
            look={p:{rule:{r['system_id']:r for r in d['rows']} for rule,d in rules.items()} for p,rules in pools.items()}
            rows=[]
            for row in b[3]:
                sid=row[0];new=row[:7]+[baseline[sid][7] if sid in baseline else 'n/a',row[7]]
                for rule in RULES:
                    for key in keys:
                        for pool in ['primary','secondary']:
                            r=look[pool][rule].get(sid,{})
                            value=r.get('MRD_plaus') if key=='MRD' else r.get('consequence') if key=='consequence' else r.get(key.split()[0],{}).get('gap_long_minus_short')
                            new.append(fmt(value,6))
                new.extend(fmt(pools[p]['v5_22p0']['computation_valid']) for p in ['primary','secondary']);rows.append(new)
            write(out,b[1],header,rows,b[4],b[5]+notes+['末端23.5秒列仅追溯；oracle双阴性对照的数值和有效性见tableS_G3_oracle_controls。'])
            cals=[read(p/'calib_plaus'/ph/'calibration.json') for p in [primary,root]]
            header,rows=calibration_rows(*cals,fmt)
            write(out,'table3_characterization'+suffix,header,rows,f'表3 三档乘积标定 H={h:g} s',notes+[
                '54点冻结乘积网格；RMSCD的MRD只比较同H；F的零尺子与步长平凡依赖不承载G2。',
                'Max b/s沿用既有标定定义；跨H的RMSCD偏移只能作诊断，不能冒充MRD。'])
            # Complete original-format calibrations, including with-block and axis columns.
            for pool,p in [('primary',primary),('secondary',root)]:
                for variant,folder in [('plaus','calib_plaus'),('axis','calib')]:
                    cal=read(p/folder/ph/'calibration.json')
                    mt.table3(cal,cal['rulers_by_metric'],h,ph,out/pool,suffix,notes,variant=variant)
    mt.table4(args.manifest,protocols,str(root/'metrics'),out)
    for pool,p in [('primary',primary),('secondary',root)]:
        r2_tables.render(str(p),str(out/pool),write,fmt)
    # Pair all predeclared controls on their natural row keys; no partial
    # averaging or implicit pooling of VLM and seeded systems.
    names={
        'tableS_A1_reversals':['H','Metric'],
        'tableS_A3_prefix_accuracy':['H'],
        'tableS_A4_region':['H','Metric'],
        'tableS_A5_region':['H','Metric'],
        'tableS_Pb_jitter':['H','Metric'],
        'tableS_Pc_stride':['H','System'],
        'tableS_G5_response_shape':['H','Metric','Axis','Block family'],
    }
    for name,keys in names.items():
        def csv_rows(pool):
            with (out/pool/(name+'.csv')).open() as f:
                reader=csv.DictReader(f);return reader.fieldnames,list(reader)
        header,a=csv_rows('primary');other,b=csv_rows('secondary');assert header==other
        index=[{tuple(r[k] for k in keys):r for r in rows} for rows in [a,b]]
        fields=[k for k in header if k not in keys]
        paired=[]
        for key in sorted(set(index[0])|set(index[1])):
            paired.append([*key,*[idx.get(key,{}).get(field,'n/a') for field in fields for idx in index]])
        write(out,name+'_two_pools',keys+[field+' '+pool for field in fields for pool in ['[-VLM]','[+VLM]']],
              paired,name+' 两系统池',notes)
    summary=[];detailed=[];controls=[]
    for h in [4.,10.,21.5]:
        for rule,label in zip(RULES,LABELS):
            row=[h,label]
            for pool in ['primary','secondary']:
                d=horizon(g3[pool][rule],h);groups=d['groups']
                row.extend(['n/a (empty stratum)' if groups['random_block']['n_undefined'] else f"{groups['random_block']['n_consequence']}/{groups['random_block']['n']}",
                            'n/a (empty stratum)' if groups['real']['n_undefined'] else f"{groups['real']['n_consequence']}/{groups['real']['n']}",fmt(d['pass_gate'])])
                for r in d['rows']:
                    detailed.append([h,pool,label,r['system_id'],r['role'],*[fmt(r[key].get(k),6) for key in ['own','fixed'] for k in ['gap_long_minus_short','ci_lo','ci_hi']],
                                     fmt(r['MRD_plaus'],6),fmt(r.get('ruler'),6),fmt(r['consequence']),r.get('coverage',{}).get('n_missing_visible','n/a')])
            summary.append(row)
        for pool in ['primary','secondary']:
            d=horizon(g3[pool]['v5_22p0'],h)
            for c in d['oracle_negative_controls']:
                controls.append([h,pool,c['system_id'],c['control_kind'],fmt(c['own_gap'],6),fmt(c['fixed_gap'],6),fmt(c['ruler'],6),fmt(c['passed']),d['status']])
    write(out,'tableS_G3_three_rules',['H','Rule','Random [-VLM]','Real [-VLM]','Pass [-VLM]','Random [+VLM]','Real [+VLM]','Pass [+VLM]'],summary,'G3 三口径对照',notes)
    write(out,'tableS_G3_three_rules_details',['H','Pool','Rule','System','Role','Own gap','Own CI lo','Own CI hi','Fixed gap','Fixed CI lo','Fixed CI hi','MRD','Ruler','Consequence','Missing cells'],detailed,'G3 三口径逐系统证据',notes)
    write(out,'tableS_G3_oracle_controls',['H','Pool','System','Control kind','Own gap','Fixed gap','Ruler','Passed','Computation status'],controls,'G3 22秒 oracle 阴性对照',notes)
    g2=[read(p/'gates/g2.json') for p in [primary,root]];g4=[read(p/'gates/g4.json') for p in [primary,root]]
    rows=[]
    for h in [4.,10.,21.5]:
        for a,b in zip(horizon(g2[0],h)['rows'],horizon(g2[1],h)['rows']):
            rows.append([h,a['metric'],*[fmt(x[k],6) for k in ['min_R','MRD_plaus','ruler','R_arm','MRD_arm'] for x in [a,b]]])
    write(out,'tableS_G2_two_pools',['H','Metric']+[k+' '+pool for k in ['min R','MRD','ruler','R arm','MRD arm'] for pool in ['[-VLM]','[+VLM]']],rows,'G2 两系统池、两承载臂',notes)
    fields=['n_tied','n_heterogeneous','n_cross_base_heterogeneous','n_replicated_cross_base_groups','pass_gate']
    write(out,'tableS_G4_two_pools',['H']+[k+' '+p for k in fields for p in ['[-VLM]','[+VLM]']],
          [[h,*[horizon(g,h)[k] for k in fields for g in g4]] for h in [4.,10.,21.5]],'G4 两系统池',notes+['VLM配对可参与Holm家族，但无种子，不能承载至少两个种子复现条件。'])
    print('[K-c tables] '+str(out),flush=True)
