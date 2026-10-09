#!/usr/bin/env python3
"""Register newly fitted maps for the same downstream regional analysis."""
import argparse,json,shutil
from pathlib import Path
import numpy as np
import pandas as pd
from source_inputs import digest

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fits',type=Path,required=True);p.add_argument('--workspace',type=Path,required=True)
    p.add_argument('--repository',type=Path,default=Path(__file__).resolve().parents[2])
    a=p.parse_args();record=json.loads((a.fits/'run_manifest.json').read_text())
    if a.workspace.exists():raise FileExistsError('Use a new workspace for the new source fits')
    config=json.loads((a.repository/'config/analysis_pipeline/regional.json').read_text())
    expected={(s['resource'],s['sample_id']) for s in config['source_sections']}
    got={(s['resource'],s['sample_id']) for s in record['section_sources']}
    if got!=expected:raise ValueError('The complete downstream analysis requires all 22 source sections')
    if set(record['methods_requested'])!=set(['M0_expr_kmeans','M1_spatial_concat_kmeans','M2_spatial_ward','M3_spatial_leiden','M4_spagcn_official','M5_stagate_official','M6_bayesspace_official']):
        raise ValueError('The complete comparison requires all seven configured methods')
    shutil.copytree(a.repository/'data',a.workspace/'data');shutil.copytree(a.repository/'config',a.workspace/'config')
    for section in config['source_sections']:
        resource,sample=section['resource'],section['sample_id'];new=a.fits/'sections'/resource/sample
        manifest=json.loads((new/'manifest.json').read_text())
        for name,r in manifest['files'].items():
            f=new/name
            if digest(f)!=r['sha256'] or f.stat().st_size!=r['bytes']:raise ValueError('Changed source-fit artifact: '+name)
        old=pd.read_csv(a.workspace/section['source_root']/'spot_inputs.tsv.gz',sep='\t',float_precision='round_trip').set_index('barcode')
        fresh=pd.read_csv(new/'spot_inputs.tsv.gz',sep='\t',float_precision='round_trip').set_index('barcode')
        if not old.index.equals(fresh.index):raise ValueError('Registered barcode order differs: '+sample)
        for name in ['array_row','array_col','coarse_label','raw_label','x_fullres','y_fullres']:
            equal = old[name].fillna('').equals(fresh[name].fillna('')) if name in ['coarse_label','raw_label'] else np.array_equal(old[name].to_numpy(),fresh[name].to_numpy())
            if not equal:raise ValueError('Original registration differs: '+sample+'/'+name)
        destination=a.workspace/section['source_root']
        for name in ['input_maps.tsv.gz','spot_scores.tsv.gz']:shutil.copy2(new/name,destination/name)
        shutil.copy2(new/'spot_components.tsv.gz',a.workspace/'data/reference_inputs/components'/resource/sample/'spot_components.tsv.gz')
    for name in config['bound_files']:
        f=a.workspace/name;config['bound_files'][name]={'bytes':f.stat().st_size,'sha256':digest(f)}
    config['study_run_type']='new fits from original public counts, compared with the saved publication fit grid'
    config['source_fit_manifest_sha256']=digest(a.fits/'run_manifest.json')
    (a.workspace/'config/analysis_pipeline/regional.json').write_text(json.dumps(config,indent=2)+'\n')
    print('Registered 22 newly fitted sections for regional analysis')

if __name__=='__main__':main()
