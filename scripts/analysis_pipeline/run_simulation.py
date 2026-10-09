#!/usr/bin/env python3
"""Generate all independent known-boundary realizations from their fixed seeds."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import argparse,json,hashlib,time
import numpy as np,pandas as pd,scipy
import simulation_model as model

def one_case(case, manifest):
    structure=model.build_structure(case,manifest)
    records=[]
    for replicate in range(manifest['replication']['independent_replicates_per_case']):
        records.extend(model.evaluate_replicate(case,structure,replicate,manifest))
    return records

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=1);p.add_argument('--resume',action='store_true');a=p.parse_args()
    identity={'manifest_sha256':hashlib.sha256(a.manifest.read_bytes()).hexdigest(),
              'model_sha256':hashlib.sha256(Path(model.__file__).read_bytes()).hexdigest()}
    record=a.output/'started_inputs.json'
    if a.output.exists():
        if not a.resume or not record.exists() or json.loads(record.read_text())!=identity:
            raise ValueError('Use a new simulation directory, or resume the same inputs')
    a.output.mkdir(parents=True,exist_ok=a.resume); start=time.monotonic()
    record.write_text(json.dumps(identity,indent=2)+'\n')
    manifest=model.load_manifest(a.manifest);cases=model.enumerate_cases(manifest)
    records=[]
    with ProcessPoolExecutor(a.workers) as pool:
        jobs={pool.submit(one_case,c,manifest):c['case_id'] for c in cases}
        for job in as_completed(jobs):
            records.extend(job.result()); print(jobs[job],len(records)//6,'realizations',flush=True)
    table=pd.DataFrame(records).sort_values(['case_id','replicate','map_name','estimator']).reset_index(drop=True)
    if len(table)!=31200 or table.duplicated(['case_id','replicate','map_name','estimator']).any():
        raise ValueError('Simulation realization grid incomplete')
    table['manifest_sha256']=hashlib.sha256(a.manifest.read_bytes()).hexdigest()
    table['numpy_version']=np.__version__;table['scipy_version']=scipy.__version__
    summary,thresholds=model.summarize(table,manifest)
    calibration=thresholds.groupby(['estimator','map_name','metric','margin_mad'],sort=True)[['numerator','denominator']].sum().reset_index()
    calibration['rate']=calibration.numerator/calibration.denominator.replace(0,np.nan)
    for name,frame in [('simulation_cell_metrics',table),('simulation_summary',summary),('simulation_threshold_sensitivity',thresholds),('simulation_calibration_by_map',calibration)]:
        frame.to_csv(a.output/(name+'.tsv'),sep='\t',index=False)
    (a.output/'analysis_record.json').write_text(json.dumps({'status':'complete','independent_realizations':5200,'rows':len(table),'seconds':time.monotonic()-start,'source':'fresh deterministic simulation'},indent=2)+'\n')
if __name__=='__main__':main()
