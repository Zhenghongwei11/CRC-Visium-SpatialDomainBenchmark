#!/usr/bin/env python3
"""Recover normalized gene measurements and describe deposited label coverage."""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source_pipeline'))
from fixed_scores import read_counts, fixed_scores
from source_inputs import digest

GENES=['TGFB1','CXCL12','ACTA2','TAGLN']
def read(path):
    return pd.read_csv(path,sep='\t',float_precision='round_trip',low_memory=False)

def recover(workspace,count_root,manifest,output):
    records=[]
    for section in json.loads(manifest.read_text())['sections']:
        resource,sample=section['resource'],section['sample_id']
        folder=count_root/resource/sample/'members'
        for member in section['source_files']:
            p=folder/member['name']
            if p.stat().st_size!=member['size_bytes'] or digest(p)!=member['sha256']:
                raise ValueError('Original count identity changed: '+sample)
        with threadpool_limits(limits=1):
            counts,barcodes,genes,_=read_counts(folder,sample)
            recovered=fixed_scores(counts,barcodes,genes,section['dataset_id'],sample)
        source=workspace/section['source_root']
        spots=read(source/'spot_inputs.tsv.gz')
        saved=read(source/'spot_scores.tsv.gz').query("score_id == 'primary_barrier'").set_index('barcode')
        calculated=recovered.query("score_id == 'primary_barrier'").set_index('barcode').loc[spots.barcode,'score_value'].to_numpy()
        recorded=saved.loc[spots.barcode,'score_value'].to_numpy()
        if not np.allclose(calculated,recorded,atol=2e-7,rtol=2e-6):
            raise ValueError('Original count normalization differs: '+sample)
        order=pd.Index(barcodes).get_indexer(spots.barcode)
        if min(order)<0 or len(set(barcodes))!=len(barcodes):raise ValueError('Invalid barcode registration')
        x=counts[order].astype(float);totals=np.asarray(x.sum(axis=1)).ravel()
        values=np.log1p(x[:,[genes.index(g) for g in GENES]].toarray()*np.divide(10000.,totals,out=np.zeros_like(totals),where=totals>0)[:,None])
        mean=values.mean(axis=1);error=abs(mean-calculated)
        row={k:section[k] for k in ['resource','patient_id','sample_id','section_id']}
        row.update(score_genes=';'.join(GENES),normalization='float64 per-spot CP10K over the complete historical feature universe; natural log1p',
                   feature_count_expected=section['feature_count'],feature_count_recovered=len(genes),feature_count_status='pass' if len(genes)==section['feature_count'] else 'mismatch',
                   matrix_barcode_count_expected=section['matrix_barcode_count'],matrix_barcode_count_recovered=len(barcodes),matrix_barcode_count_status='pass' if len(barcodes)==section['matrix_barcode_count'] else 'mismatch',
                   registered_barcode_count=len(spots),n_barcodes_compared=len(spots),
                   registered_barcode_order_sha256=hashlib.sha256(('\n'.join(spots.barcode)+'\n').encode()).hexdigest(),
                   registered_barcodes_source_sha256=digest(source/'spot_inputs.tsv.gz'),
                   source_package_sha256=section['source_package_sha256'],source_archive_sha256=section['source_package_sha256'],
                   source_file_names=';'.join(m['name'] for m in section['source_files']),
                   source_file_sha256_json=json.dumps({m['name']:m['sha256'] for m in section['source_files']},separators=(',',':')),
                   source_file_sizes_json=json.dumps({m['name']:m['size_bytes'] for m in section['source_files']},separators=(',',':')),
                   source_counts_manifest_sha256=digest(manifest),recovery_script_sha256=digest(__file__),
                   historical_result_manifest_sha256=None,historical_spot_inputs_sha256=digest(source/'spot_inputs.tsv.gz'),
                   historical_spot_scores_sha256=digest(source/'spot_scores.tsv.gz'),historical_spot_scores_gzip_sha256_expected=digest(source/'spot_scores.tsv.gz'),
                   historical_fixed_score_output_sha256=None,component_manifest_sha256=None,
                   spot_components_sha256=digest(workspace/'data/reference_inputs/components'/resource/sample/'spot_components.tsv.gz'),
                   score_comparison_status='pass',mismatches_vs_historical_recomputed=int((~np.isclose(mean,calculated,atol=2e-7,rtol=2e-6)).sum()),
                   max_absolute_error_vs_historical_recomputed=float(error.max()),max_absolute_error_vs_historical_recorded=float(abs(mean-recorded).max()),
                   max_absolute_error_vs_fixed_scores_recorded=float(abs(calculated-recorded).max()),score_atol=2e-7,score_rtol=2e-6)
        records.append(row)
    pd.DataFrame(records).to_csv(output/'component_source_recovery.tsv',sep='\t',index=False,na_rep='NA')

def coverage(workspace,output):
    protocol=json.loads((workspace/'config/analysis_pipeline/regional.json').read_text())
    ann=workspace/'data/reference_inputs/annotations/GSE294385'
    raw=read(ann/'GSE294385_raw_label_layers.tsv')
    qc=read(ann/'GSE294385_label_layers.tsv')
    rows=[]
    for section in protocol['source_sections']:
        spots=read(workspace/section['source_root']/'spot_inputs.tsv.gz');sample=section['sample_id']
        present=spots.raw_label.fillna('').astype(str).str.strip().ne('')
        row={k:section[k] for k in ['resource','patient_id','sample_id','section_id']}
        row.update(registered_spots=len(spots),source_label_present_spots=int(present.sum()),source_label_absent_spots=int((~present).sum()),source_label_absent_fraction=float((~present).mean()))
        for label in ['tumor','stroma','normal epithelium','unresolved','mixed','exclude']:
            row['n_'+label.replace(' ','_')]=int(spots.coarse_label.eq(label).sum())
        row.update(public_raw_registered_label_spots=None,public_qc_registered_label_spots=None,public_raw_qc_registered_sets_equal=None)
        if section['resource']=='GSE294385':
            rset=set(raw.loc[raw.sample_id.eq(sample),'Barcode']);qset=set(qc.loc[qc.sample_id.eq(sample),'Barcode'])
            r=spots.barcode.isin(rset);q=spots.barcode.isin(qset)
            row.update(public_raw_registered_label_spots=int(r.sum()),public_qc_registered_label_spots=int(q.sum()),public_raw_qc_registered_sets_equal=bool(np.array_equal(r,q)))
            if sample=='M-ST-15':
                base=workspace/protocol['output_dir']/'sections'/section['resource']/sample
                with np.load(base/'geometry_arrays.npz') as arrays:
                    spots['spatial_block']=arrays['historical_raw_6nn|baseline|blocks']
                    spots['public_raw_present']=r;spots['public_qc_present']=q
                    spots['primary_anchor_near']=arrays['historical_raw_6nn|baseline|near'];spots['primary_anchor_far']=arrays['historical_raw_6nn|baseline|far']
                (output/'figure_source_data').mkdir(exist_ok=True)
                spots.to_csv(output/'figure_source_data/figureS7_spots.tsv.gz',sep='\t',index=False,compression={'method':'gzip','mtime':0})
                hierarchy=raw[raw.sample_id.eq(sample)].set_index('Barcode').reindex(spots.loc[r,'barcode'])
                hierarchy.index.name='barcode'
                hierarchy[['Layer1','Layer2','Layer3']].reset_index().to_csv(output/'additional_source_data/patient6_original_annotation_hierarchy.tsv.gz',sep='\t',index=False,compression={'method':'gzip','mtime':0})
        rows.append(row)
    pd.DataFrame(rows).to_csv(output/'source_label_coverage.tsv',sep='\t',index=False)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['workspace','source-manifest','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--count-root',type=Path)
    p.add_argument('--stage',choices=['all','counts','coverage'],default='all')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);(a.output/'additional_source_data').mkdir(exist_ok=True)
    if a.stage in ['all','counts']:
        if a.count_root is None:p.error('--count-root is required for count recovery')
        recover(a.workspace,a.count_root,a.source_manifest,a.output)
    if a.stage in ['all','coverage']:coverage(a.workspace,a.output)
