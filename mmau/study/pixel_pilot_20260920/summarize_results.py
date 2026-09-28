"""Aggregate the fixed pilot runs without selecting a winning seed."""
import json
import statistics
from pathlib import Path

base = Path(__file__).resolve().parent
model = json.loads((base / 'model_run/summary.json').read_text())
evaluation = json.loads((base / 'model_run/ape_evaluation.json').read_text())
result = {'models': {}, 'baseline': [r for r in evaluation['runs'] if r['system'] == 'train_majority' and r['split'] == 'dev']}
for kind in ('prefix_mean', 'gru'):
    item = {'endpoints': {}, 'stability': {}}
    records = [r for r in model['runs'] if r['model'] == kind]
    for split in ('train', 'dev'):
        item['endpoints'][split] = {}
        for offset in (0, 39, 67):
            values = [r['splits'][split][str(offset)]['macro_accuracy'] for r in records]
            item['endpoints'][split][str(offset)] = {'mean': statistics.mean(values), 'min': min(values), 'max': max(values)}
    for horizon in (39, 67):
        records_eval = [r for r in evaluation['runs'] if r['system'].startswith(kind + '_') and r['split'] == 'dev' and r['h_f'] == horizon]
        item['stability'][str(horizon)] = {k: statistics.mean([r['metrics'][k] for r in records_eval]) for k in records_eval[0]['metrics']}
        item['stability'][str(horizon)]['mean_ever_correct_then_wrong_count'] = statistics.mean(r['ever_correct_then_wrong_at_end'] for r in records_eval)
        item['stability'][str(horizon)]['mean_native_class_recall'] = {k: statistics.mean(r['native_class_recall'][k] for r in records_eval) for k in records_eval[0]['native_class_recall']}
    result['models'][kind] = item
result['elapsed_seconds'] = model['elapsed_seconds']
(base / 'RESULTS.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
