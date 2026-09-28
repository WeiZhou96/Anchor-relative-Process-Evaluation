"""K-c figures: K-b primary evidence, plus three-rule v5 G3 comparison."""
from pathlib import Path
import numpy as np
import ape_io as io
import r2_figs
from r2c_tables import read,horizon,VLM,RULES


def fig8(plt,root,out):
    data=read(Path(root)/'gates/g3_v5.json')
    fig,axes=plt.subplots(3,3,figsize=(13,10),constrained_layout=True)
    colors={'real':'#42657E','random_block':'#A76450'}
    labels=['R2-8 original','A4-2: 23.5 s (cache invalid)','A5-1: 22.0 s (record)']
    for i,h in enumerate([4.,10.,21.5]):
        for j,rule in enumerate(RULES):
            ax=axes[i,j];p=horizon(data['primary'][rule],h);s=horizon(data['secondary'][rule],h)
            for role in ['real','random_block']:
                rows=[r for r in p['rows'] if r['role']==role and r['fixed']['gap_long_minus_short'] is not None]
                ax.scatter([r['own']['gap_long_minus_short'] for r in rows],[r['fixed']['gap_long_minus_short'] for r in rows],
                    s=12 if role=='real' else 32,color=colors[role],alpha=.65,label='Primary real' if role=='real' else 'Random blocks',zorder=3)
            vlm=next(r for r in s['rows'] if r['system_id']==VLM)
            if vlm['fixed']['gap_long_minus_short'] is not None:
                ax.scatter([vlm['own']['gap_long_minus_short']],[vlm['fixed']['gap_long_minus_short']],marker='*',s=90,color='#659583',edgecolors='white',lw=.5,label='VLM (secondary)',zorder=5)
            else:
                ax.set_ylim(-1.,1.);ax.set_yticks([])
                vals=[r['own']['gap_long_minus_short'] for r in p['rows'] if r['role']=='real']
                ax.plot(vals,np.zeros(len(vals)),ls='none',marker='|',markersize=7,color='#42657E',alpha=.45)
                ax.text(.03,.49,'Fixed contrast undefined:\nshort stratum empty in E_H\nRug: own gaps only',transform=ax.transAxes,fontsize=8,color='#596873')
            if j:
                ax.axhspan(-p['ruler'],p['ruler'],color='#42657E',alpha=.08,label='Primary ruler')
                for k,y in enumerate([-s['ruler'],s['ruler']]):ax.axhline(y,color='#659583',lw=.7,ls=':',label='Secondary ruler' if k==0 else None)
                for x in [-p['MRD_plaus'],p['MRD_plaus']]:ax.axvline(x,color='#929BA3',lw=.7,ls='--')
                oracle=next(r for r in p['rows'] if r['system_id']=='trivial__oracle')
                ax.scatter([oracle['own']['gap_long_minus_short']],[oracle['fixed']['gap_long_minus_short']],marker='D',s=34,
                    facecolors='none',edgecolors='#AC6B59',lw=1.,label='Cache oracle',zorder=6)
            ax.axhline(0,color='#929BA3',lw=.6);ax.axvline(0,color='#929BA3',lw=.6)
            def count(d):
                g=d['groups']['real']
                return "undefined" if g['n_undefined'] else f"{g['n_consequence']}/{g['n']}"
            valid='; oracle OK' if j==2 and p['computation_valid'] else '; INVALID' if j==2 else ''
            ax.set_title(f"H={h:g} s | {labels[j]}\nConsequences: -VLM {count(p)}, +VLM {count(s)}{valid}",loc='left',fontsize=8.3)
            ax.set_xlabel('Own: long - short (s)',fontsize=8)
            ax.set_ylabel('Fixed contrast undefined' if not j and h!=4. else 'Fixed H: long - short (s)',fontsize=8)
            ax.tick_params(labelsize=7);ax.margins(.15)
            if i==0:ax.legend(fontsize=6.2,loc='upper left',ncol=1)
    r2_figs.save(fig,out,'fig8_r2c_g3_three_rules',plt)


def render(root,args,mf):
    root=Path(root);primary=root.parent/'r2b';out=Path(args.out_dir) if args.out_dir else root/'figs'
    out.mkdir(parents=True,exist_ok=True)
    protocols=io.reference_protocols(str(primary),str(primary/'metrics'))
    ph=args.pi_hash or '8ac32aae418b';h=dict(protocols)[ph]
    recs=io.load_metrics(str(primary/'metrics'),ph);pairs=io.load_pairs(str(primary/'metrics'),ph)
    plt=mf.setup_mpl();g4=read(primary/'gates/g4.json')
    proofs=sorted((r['system_a'],r['system_b']) for g in horizon(g4,h)['seed_groups'] if g['pass'] for r in g['seeds'] if r['significant'])
    want=[s.strip() for s in args.fig1_systems.split(',') if s.strip()] or list(proofs[0])
    mf.fig1(plt,recs,pairs,h,ph,out,want,'K-b primary: first lexical replicated cross-base same-seed pair')
    mf.fig2(plt,recs,h,out,args.fig2_system or None)
    _,cal=io.load_calibration(str(primary/'calib_plaus'),ph)
    mf.fig3(plt,cal,out,False)
    mf.fig4(plt,protocols,str(primary/'metrics'),out)
    # Figures 5-7 preserve primary history. The old 23.5-s figure 8 is
    # suppressed at render time rather than written then removed.
    r2_figs.render(plt,str(primary),out,include_legacy_fig8=False)
    fig8(plt,root,out)
    readme='Figures 1-7 use the immutable K-b primary pool (without VLM).\nFigure 5 preserves R2-8 for traceability. Figure 8 compares all three G3 rules and both pools; A5-1 22 s is the record.\nAt 23.5 s the ground-truth cache sentinel fails: that historical pass is not evidence.\n'
    (out/'README.txt').write_text(readme)
    print('[K-c figures] '+str(out),flush=True)
