"""Summarize paired shifts at a fixed horizon without checkpoint reselection."""
import json
import statistics
from pathlib import Path

base=Path(__file__).resolve().parent
runs=json.loads((base/'anchor/ape_evaluation.json').read_text())['runs']
arms=sorted({r['arm'] for r in runs})
cells=[]
for arm in arms:
    for shift in (-4,0,4):
        rr=[r for r in runs if r['arm']==arm and r['shift']==shift]
        cells.append(dict(arm=arm,shift=shift,metrics={k:statistics.mean(r['metrics'][k] for r in rr) for k in rr[0]['metrics']},class_recalls={k:statistics.mean(r['native_class_recall'][k] for r in rr) for k in rr[0]['native_class_recall']}))
paired=[]
for arm in arms:
    for shift in (-4,4):
        base_runs={r['seed']:r for r in runs if r['arm']==arm and r['shift']==0}
        changed={r['seed']:r for r in runs if r['arm']==arm and r['shift']==shift}
        values={key:[changed[s]['metrics'][key]-base_runs[s]['metrics'][key] for s in sorted(base_runs)] for key in ('end_window_macro_acc','end_window_micro_acc','RMSCD@H_norm_macro')}
        paired.append(dict(arm=arm,shift=shift,seed_differences=values,means={k:statistics.mean(v) for k,v in values.items()}))
report=dict(cells=cells,paired=paired,formal=False,source_identity_certified=False,no_retraining=True,common_n=269,horizon=60)
(base/'RESULTS.json').write_text(json.dumps(report,indent=2)+'\n')
for cell in cells:
    print(cell['arm'],cell['shift'],{k:round(cell['metrics'][k],4) for k in ('end_window_macro_acc','end_window_micro_acc','RMSCD@H_norm_macro','mean_flips')})
