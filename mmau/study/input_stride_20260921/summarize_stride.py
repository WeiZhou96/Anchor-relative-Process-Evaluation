"""Aggregate seeds and retain paired differences for each cadence contrast."""
import json
import statistics
from pathlib import Path

base=Path(__file__).resolve().parent
runs=json.loads((base/'input/ape_evaluation.json').read_text())['runs']
arms=sorted({r['arm'] for r in runs})
conditions=['stride2','stride4','stride8','legacy4plus39']
cells=[]
for arm in arms:
    for condition in conditions:
        rr=[r for r in runs if r['arm']==arm and r['condition']==condition]
        assert len(rr)==3
        cells.append(dict(arm=arm,condition=condition,input_frames=rr[0]['input_frames'],metrics={k:statistics.mean(r['metrics'][k] for r in rr) for k in rr[0]['metrics']},class_recalls={k:statistics.mean(r['native_class_recall'][k] for r in rr) for k in rr[0]['native_class_recall']}))
paired=[]
for arm in arms:
    for condition in ['stride2','stride8','legacy4plus39']:
        baseline={r['seed']:r for r in runs if r['arm']==arm and r['condition']=='stride4'}
        changed={r['seed']:r for r in runs if r['arm']==arm and r['condition']==condition}
        values={key:[changed[s]['metrics'][key]-baseline[s]['metrics'][key] for s in sorted(baseline)] for key in ('end_window_macro_acc','end_window_micro_acc','RMSCD@H_norm_macro','mean_flips')}
        paired.append(dict(arm=arm,condition=condition,reference='stride4',seed_differences=values,means={k:statistics.mean(v) for k,v in values.items()}))
report=dict(cells=cells,paired=paired,formal=False,source_identity_certified=False,no_retraining=True,common_n=269,horizon=64,output_delta=8)
(base/'RESULTS.json').write_text(json.dumps(report,indent=2)+'\n')
for cell in cells:
    print(cell['arm'],cell['condition'],{k:round(cell['metrics'][k],5) for k in ('end_window_macro_acc','end_window_micro_acc','RMSCD@H_norm_macro','mean_flips')})
for pair in paired:
    print('PAIRED',pair['arm'],pair['condition'],{k:[round(x,5) for x in v] for k,v in pair['seed_differences'].items()})
