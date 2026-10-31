#!/usr/bin/env python3
"""Prepare and fit the RCTD model sequentially for the selected spatial sections."""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def completed(output,sid):
    folder=output/'all'/sid;status=folder/'RUN_STATUS.json'
    if not status.exists():return False
    record=json.loads(status.read_text())
    if record.get('status')!='fit_completed':return False
    manifest=output/'inputs/all'/sid/'INPUT_MANIFEST.json'
    if record.get('section')!=sid or record.get('mode')!='all' or record.get('protocol_sha256')!=sha(ROOT/'config/rctd_model.json') or record.get('runner_sha256')!=sha(ROOT/'scripts/run_rctd.R') or record.get('input_manifest_sha256')!=sha(manifest):raise ValueError('Completed fit identity changed for '+sid)
    for item in record['outputs']:
        if sha(folder/item['file'])!=item['sha256']:raise ValueError('Completed fit output changed for '+sid)
    return True

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--counts',type=Path,required=True,help='Verified public count files by resource and sample')
    p.add_argument('--output',type=Path,required=True,help='RCTD outputs and regenerated count staging')
    p.add_argument('--section',action='append',help='Optional sample identifier; repeat to select sections')
    p.add_argument('--rscript',default='Rscript')
    p.add_argument('--remove-generated-counts',action='store_true',help='Remove temporary count matrices after successful fit')
    a=p.parse_args();counts=a.counts.resolve();output=a.output.resolve();output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((ROOT/'config/source_pipeline/source_counts_manifest.json').read_text())
    sections=[x['sample_id'] for x in manifest['sections']]
    if a.section:
        missing=set(a.section)-set(sections)
        if missing:raise ValueError('Unknown section(s): '+', '.join(sorted(missing)))
        wanted=set(a.section);sections=[s for s in sections if s in wanted]
    env=os.environ.copy()
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):env[key]='2'
    for sid in sections:
        if completed(output,sid):
            print('Reusing verified completed fit:',sid,flush=True);continue
        print('Preparing:',sid,flush=True)
        subprocess.run([sys.executable,str(ROOT/'scripts/prepare_rctd_inputs.py'),'--all','--section',sid,'--counts',str(counts),'--output',str(output)],check=True,env=env)
        print('Fitting:',sid,flush=True)
        subprocess.run([a.rscript,'scripts/run_rctd.R','--mode','all','--section',sid,'--output',str(output)],check=True,env=env,cwd=ROOT)
        status=json.loads((output/'all'/sid/'RUN_STATUS.json').read_text())
        if status.get('status')!='fit_completed':raise RuntimeError('RCTD fit did not complete for '+sid)
        if a.remove_generated_counts:(output/'inputs/all'/sid/'counts.mtx.gz').unlink()
        print('Completed:',sid,flush=True)
    print('Completed sections:',len(sections),flush=True)
if __name__=='__main__':main()
