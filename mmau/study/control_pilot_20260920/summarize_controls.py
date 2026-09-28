"""Compare every prespecified arm and paired seed, including epoch-40 sensitivity."""
import json
import statistics
from pathlib import Path

base = Path(__file__).resolve().parent
old = base.parent / "pixel_pilot_20260920"
summary = json.loads((base / "model_run/summary.json").read_text())
old_summary = json.loads((old / "model_run/summary.json").read_text())
evaluation = json.loads((base / "model_run/ape_evaluation.json").read_text())
old_evaluation = json.loads((old / "model_run/ape_evaluation.json").read_text())
runs = summary["runs"] + [dict(r, arm={"prefix_mean": "mean_weighted", "gru": "gru_weighted"}[r["model"]], system=f'{r["model"]}_{r["seed"]}') for r in old_summary["runs"]]
metrics = evaluation["runs"] + [r for r in old_evaluation["runs"] if r["system"] != "train_majority"]
arms = ["anchor_weighted", "anchor_unweighted", "mean_weighted", "mean_unweighted", "gru_weighted", "gru_unweighted", "shuffle_weighted", "shuffle_unweighted"]
result = {"arms": {}, "paired_differences": {}, "formal": False, "dev_selected": True}
for arm in arms:
    rr = [r for r in runs if r["arm"] == arm]
    item = {}
    for offset in (0, 39, 67):
        values = [r["splits"]["dev"][str(offset)]["macro_accuracy"] for r in rr]
        item[f"macro_{offset}"] = statistics.mean(values)
    item["macro67_range"] = [min(r['splits']['dev']['67']['macro_accuracy'] for r in rr), max(r['splits']['dev']['67']['macro_accuracy'] for r in rr)]
    item["micro67"] = statistics.mean(r['splits']['dev']['67']['micro_accuracy'] for r in rr)
    item["train_macro67"] = statistics.mean(r['splits']['train']['67']['macro_accuracy'] for r in rr)
    mm = [r for r in metrics if r["system"] in {x["system"] for x in rr} and r["split"] == "dev" and r["h_f"] == 67]
    for key in ("RMSCD@H_norm_macro", "RMSCD@H_norm", "mean_flips", "flip_rate"):
        item[key] = statistics.mean(r["metrics"][key] for r in mm)
    item["class_recalls"] = {k: statistics.mean(r['native_class_recall'][k] for r in mm) for k in mm[0]['native_class_recall']}
    item["selected_epochs"] = [r['selected_epoch'] for r in rr]
    folder = old / "model_run" if arm in ("mean_weighted", "gru_weighted") else base / "model_run"
    item["epoch40_macro67"] = statistics.mean(json.loads((folder / (r['system'] + '_history.json')).read_text())[-1]['dev_macro_67'] for r in rr)
    result['arms'][arm] = item
for a, b in [("anchor_weighted", "mean_weighted"), ("shuffle_weighted", "gru_weighted"), ("shuffle_unweighted", "gru_unweighted"), ("anchor_unweighted", "anchor_weighted"), ("mean_unweighted", "mean_weighted"), ("gru_unweighted", "gru_weighted")]:
    aa = {r['seed']: r['splits']['dev']['67']['macro_accuracy'] for r in runs if r['arm'] == a}
    bb = {r['seed']: r['splits']['dev']['67']['macro_accuracy'] for r in runs if r['arm'] == b}
    differences = [aa[s] - bb[s] for s in sorted(aa)]
    result['paired_differences'][a + '_minus_' + b] = {'by_seed': differences, 'mean': statistics.mean(differences)}
(base / 'RESULTS.json').write_text(json.dumps(result, indent=2) + '\n')
for arm, row in result['arms'].items():
    print(arm, {k: round(row[k], 4) for k in ('macro_0', 'macro_39', 'macro_67', 'micro67', 'RMSCD@H_norm_macro', 'mean_flips', 'epoch40_macro67')})
print(json.dumps(result['paired_differences'], indent=2))
