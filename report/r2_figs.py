"""Quantitative R2 diagnostics, with complete source arrays saved in JSON."""
from pathlib import Path
import json
import numpy as np


def read(root,folder,name):
    p=Path(root)/folder/f'{name}.json'
    return json.loads(p.read_text()) if p.exists() else {}


def save(fig,out,name,plt):
    for ext in ['png','pdf','svg']:
        fig.savefig(Path(out)/f'{name}.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)


def render(plt,root,out,include_legacy_fig8=True):
    g3=read(root,'gates','g3')
    if g3:
        fig,axes=plt.subplots(3,1,figsize=(9,7.5),constrained_layout=True)
        for ax,h in zip(axes,g3['by_h']):
            rows=[r for r in h['rows'] if r['role'] in ['real','random_block']]
            xs=np.arange(len(rows))
            for key,color,marker,label in [('own','#718CA7','o','Own visible end'),('fixed','#315B78','x','Fixed cohort')]:
                ys=[abs(r[key]['gap_long_minus_short']) if r[key]['gap_long_minus_short'] is not None else np.nan for r in rows]
                ax.scatter(xs,ys,s=13,c=color,marker=marker,label=label)
            ax.axhline(h['MRD_plaus'],color='#AC6B59',linestyle='--',linewidth=1,label='MRD plaus')
            ax.set_title(f"H = {h['h_s']:g} s; {len(rows)} systems; fixed contrast missing in {sum(r['consequence'] is None for r in rows)}",loc='left',fontsize=9)
            ax.set_ylabel('|Long - short| (s)');ax.set_ylim(bottom=0);ax.set_xlim(-1,len(rows))
            ax.set_xlabel('System index (JSON order; first two are random gauge blocks)')
            ax.legend(loc='upper right',ncol=3,fontsize=7)
        save(fig,out,'fig5_r2_g3_stratified_gaps',plt)
    g4=read(root,'gates','g4');a3=read(root,'ablations','a3')
    if g4 and a3:
        fig,axes=plt.subplots(1,3,figsize=(10.8,3.6),constrained_layout=True)
        for ax,gh,ah in zip(axes,g4['by_h'],a3['by_h']):
            rows=[r for r in ah['pairs'] if r['g4_heterogeneous']]
            diffs={(r['system_a'],r['system_b']):r['diff'] for r in gh['pairs']}
            for detected,color,label in [(False,'#AC6B59','AUC misses'),(True,'#315B78','AUC separates')]:
                rs=[r for r in rows if r['significant']==detected]
                ax.scatter([diffs[(r['system_a'],r['system_b'])] for r in rs],[r['diff'] for r in rs],
                           s=9,alpha=.5,color=color,label=f'{label}: {len(rs)}')
            ax.axhline(0,color='#929BA3',lw=.7);ax.axvline(0,color='#929BA3',lw=.7)
            ax.set_xlabel('RMSCD paired difference (s)');ax.set_ylabel('Dynamic accuracy AUC/H difference')
            ax.set_title(f"H = {gh['h_s']:g} s; {gh['n_heterogeneous']} heterogeneous pairs",fontsize=9)
            ax.legend(loc='best',fontsize=7)
        save(fig,out,'fig6_r2_g4_vs_accuracy_control',plt)
    g5=read(root,'gates','g5')
    if g5:
        rows=[r for r in g5['rows'] if r['h_s']==10 and r['block_family']=='all_blocks']
        metrics=sorted({r['metric'] for r in rows});knobs=['eps_sys_s','eps_jit_sd_s','delta_s','h_s']
        fig,axes=plt.subplots(2,1,figsize=(8.5,4.5),constrained_layout=True)
        for ax,key,label,vmin,vmax in [(axes[0],'pearson','Pearson response correlation',-1,1),(axes[1],'sign_agreement','Non-reference sign agreement',0,1)]:
            mat=np.array([[next(r[key] for r in rows if r['metric']==m and r['axis']==k) for m in metrics] for k in knobs],float)
            im=ax.imshow(np.ma.masked_invalid(mat),vmin=vmin,vmax=vmax,cmap='BrBG' if key=='pearson' else 'Blues',aspect='auto')
            ax.grid(False);ax.set_xticks(range(len(metrics)),metrics,fontsize=8);ax.set_yticks(range(len(knobs)),knobs,fontsize=8)
            for i in range(len(knobs)):
                for j in range(len(metrics)):
                    v=mat[i,j];ax.text(j,i,'n/a' if np.isnan(v) else f'{v:.2f}',ha='center',va='center',fontsize=8,
                        color='white' if np.isfinite(v) and abs(v)>.8 else '#172C38')
            ax.set_title(label+'; H = 10 s',loc='left',fontsize=9)
            fig.colorbar(im,ax=ax,fraction=.025,pad=.02)
        save(fig,out,'fig7_r2_g5_response_shape',plt)

    if include_legacy_fig8 and g3.get('revised_rule'):
        fig,axes=plt.subplots(1,3,figsize=(11,3.8),constrained_layout=True)
        for ax,h in zip(axes,g3['revised_rule']['by_h']):
            rows=[r for r in h['rows'] if r['role'] in ['real','random_block']]
            for role,c,label in [('real','#315B78','Real systems'),('random_block','#AC6B59','Random blocks')]:
                rs=[r for r in rows if r['role']==role]
                ax.scatter([r['own']['gap_long_minus_short'] for r in rs],[r['fixed']['gap_long_minus_short'] for r in rs],s=22 if role=='random_block' else 10,alpha=.7,color=c,label=label)
            ax.axhline(0,color='#929BA3',lw=.7);ax.axvline(0,color='#929BA3',lw=.7)
            ax.axhspan(-h['ruler'],h['ruler'],color='#315B78',alpha=.08)
            ax.set_xlabel('Own visible end: long - short (s)');ax.set_ylabel('Fixed H: long - short (s)')
            ax.set_title(f"H={h['h_s']:g}s; real consequences {h['groups']['real']['n_consequence']}/168",fontsize=9)
            ax.legend(fontsize=7)
        fig.text(.5,-.015,'Visible end capped at 23.5 s; real-system caches end at 22 s, with missing visible cells scored wrong.',ha='center',fontsize=8,color='#596873')
        save(fig,out,'fig8_r2b_g3_revised',plt)
