"""One exclusive final batch: only H=10 endpoint macro-Acc on test."""
from pathlib import Path
import datetime,json,os,hashlib
import numpy as np,pandas as pd
from systems import common as C
from systems.devsel import macro_acc
from systems.r2_models import STATE,write_json,check_freeze
from systems.r2_generate import assert_seal


def main():
    check_freeze();plan=json.loads((STATE/'selection_sealed.json').read_text());assert_seal(plan)
    verified=json.loads((STATE/'verification_complete.json').read_text())
    assert verified['new_directories']==324 and verified['checks_failed']==0
    registry=json.loads((STATE/'new_registry.json').read_text())
    target=STATE/'final_endpoint_summary.csv';done=STATE/'final_evaluation_complete.json'
    if done.exists():
        print('FINAL_EVALUATION_ALREADY_COMPLETE; returning stored table, no recalculation')
        print(done.read_text());return
    lock=STATE/'test_evaluation_started.json';ledger=STATE/'test_evaluation_ledger.jsonl'
    with lock.open('x') as f:
        json.dump({'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'batch_number':1,'metric':'H10 window-end macro-Acc','scope':'324 new fine/coarse directories','selection_sealed_utc':plan['sealed_utc']},f)
        f.flush();os.fsync(f.fileno())
    man=C.load_manifest().set_index('video_id');results=[]
    cohorts={s:man[(man.split==s)&(man.post_anchor_length_s+1e-9>=10)] for s in ['dev','test']}
    counts={s:len(d) for s,d in cohorts.items()};assert counts=={'dev':52,'test':1113}
    digest={x['system_id']:x['sha256'] for x in verified['files']}
    with ledger.open('x') as f:
        for item in registry:
            path=Path(item['answers_dir'])/'answers.csv'
            assert hashlib.sha256(path.read_bytes()).hexdigest()==digest[item['system_id']]
            df=pd.read_csv(path,usecols=['video_id','j','pred'])
            end=df[df.j==int(round(10/item['delta_s']))].set_index('video_id')
            row=dict(item)
            for split,cohort in cohorts.items():
                pred=end.loc[cohort.index,'pred'].to_numpy();label=cohort.class_code.to_numpy()
                row[split+'_end_macro_acc']=macro_acc(pred,label)
            row['H_s']=10.0;row['dev_N']=52;row['test_N']=1113;row['test_metric_evaluations']=1
            results.append(row)
            f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
    pd.DataFrame(results).to_csv(target,index=False)
    write_json(done,{'batch_count':1,'systems_scored':len(results),'test_metric_evaluations_per_system':1,'metric':'window-end macro-Acc','H_s':10.0,'cohort_sizes':counts,'other_new_test_metrics_computed':0,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'ledger':str(ledger),'summary_csv':str(target)})
    print('FINAL_EVALUATION_DONE',done.read_text(),flush=True)


if __name__=='__main__': main()
