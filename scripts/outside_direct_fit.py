#!/usr/bin/env python3
"""Frozen outside-direct-fit gene-universe sensitivity of saved regional vectors only."""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_key]='1'
import argparse,json,sys,time,traceback
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from rctd_profiles import compare_profiles
from regional_profiles import profile

ROOT=Path(__file__).resolve().parents[1]
DOC=ROOT/'config'
NATIVE=ROOT/'results/rctd'
SUPPORT=ROOT/'results/rctd_calibration'
PRIMARY=ROOT/'results/rctd_profiles'
OLD=ROOT/'results/regional_profiles'
DEFAULT_OUT=ROOT/'results/outside_direct_fit'
METRICS=('spearman','scaled_abs_difference','unscaled_abs_difference')
CONTROLS=('frozen_mean_calibrated_weights','unweighted_full_regions')
VALUES=('near_value','far_value','far_minus_near')

import hashlib

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def readj(p):return json.loads(Path(p).read_text())
def readt(p):return pd.read_csv(p,sep='\t',float_precision='round_trip')
def check(x,message):
    if not x:raise ValueError(message)
def writej(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name('.'+p.name+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def save(p,x):
    p=Path(p);tmp=p.with_name('.'+p.name+'.tmp');x.to_csv(tmp,sep='\t',index=False,na_rep='NA',compression={'method':'gzip','mtime':0} if p.name.endswith('.gz') else None);tmp.replace(p)
def rel(p):
    p=Path(p).resolve()
    try:return str(p.relative_to(ROOT))
    except ValueError:return str(p)

def protocol():
    p=DOC/'direct_fit_exclusion.json';j=readj(p)
    check(j['controls']==list(CONTROLS),'Control definition differs')
    return j,{rel(p):sha(p)}

def make_membership(panel,bulk,regression):
    # Preserve every original gene-ID row and exact literal symbol; never collapse duplicate symbols.
    x=panel.copy();symbols=x.gene_symbol.fillna('').astype(str);union=set(bulk)|set(regression)
    x['in_any_native_bulk_symbol']=symbols.isin(set(bulk));x['in_any_native_regression_symbol']=symbols.isin(set(regression));x['in_direct_fit_symbol_union']=symbols.isin(union)
    x['retain_outside_direct_fit']=x.broad & ~x.in_direct_fit_symbol_union
    x['reason']=np.where(~x.broad,'not_original_broad_geneID_row',np.where(x.in_direct_fit_symbol_union,'excluded_exact_direct_fit_symbol_union','retained_original_broad_outside_direct_fit_union'))
    x['symbol_missing_in_source_panel']=x.gene_symbol.isna()
    return x

def native_complete():
    sections=readj(ROOT/'config/source_pipeline/source_counts_manifest.json')['sections'];entries=[]
    for section in sections:
        p=NATIVE/'all'/section['sample_id']/'RUN_STATUS.json'
        if not p.exists():return False,None
        r=readj(p)
        if r.get('status')!='fit_completed':return False,None
        entries.append({'section':section['sample_id'],'RUN_STATUS_sha256':sha(p)})
    return True,{'status':'all_native_fits_completed','sections':entries}

def build_or_verify_panel(out,amendment_hashes):
    lock_path=out/'PANEL_LOCK.json'
    if lock_path.exists():
        lock=readj(lock_path);check(lock['script_sha256']==sha(__file__),'Panel builder script changed after lock')
        for name,digest in lock['source_hashes'].items():check(sha(ROOT/name)==digest,'Frozen panel source changed: '+name)
        for name,digest in lock['panel_files'].items():check(sha(out/name)==digest,'Frozen panel artifact changed: '+name)
        check(lock['weighted_profile_vectors_opened_before_lock'] is False and lock['new_mask_RNA_outcomes_calculated_before_lock'] is False,'Panel sequence provenance invalid')
        return lock
    ready,batch=native_complete();check(ready,'All22 native fits must complete before panel freeze')
    source=dict(amendment_hashes)
    for helper in ['rctd_profiles.py','regional_profiles.py','rctd_calibration.py']:
        helper_path=ROOT/'scripts'/helper;source[rel(helper_path)]=sha(helper_path)
    manifest_path=ROOT/'config/source_pipeline/source_counts_manifest.json';sections=readj(manifest_path)['sections'];source[rel(manifest_path)]=sha(manifest_path)
    entries={x['section']:x for x in batch['sections']};check(set(entries)=={x['sample_id'] for x in sections},'Native batch section universe differs')
    oldrun_path=OLD/'RUN_MANIFEST.json';oldrun=readj(oldrun_path);panel_path=OLD/'gene_panels.tsv';check(sha(panel_path)==oldrun['outputs']['gene_panels.tsv']['sha256'],'Original full gene panel hash differs');source[rel(panel_path)]=sha(panel_path);source[rel(oldrun_path)]=sha(oldrun_path)
    model_path=DOC/'rctd_model.json';model_sha=sha(model_path);check(model_sha==readj(DOC/'rctd_model_lock.json')['sha256'],'Native protocol lock differs');source[rel(model_path)]=model_sha
    panel=readt(panel_path)[['resource','gene_id','gene_symbol','broad','fixed_gene_scale']]
    check(not panel[['resource','gene_id']].duplicated().any() and panel.gene_id.notna().all(),'Original gene-ID rows invalid')
    check(np.isfinite(panel.fixed_gene_scale).all() and (panel.fixed_gene_scale>0).all(),'Original scales invalid')
    unions={r:{'bulk':set(),'regression':set(),'by_section':[]} for r in panel.resource.unique()};inputs=[]
    for section in sections:
        sid=section['sample_id'];p=NATIVE/'all'/sid;status_path=p/'RUN_STATUS.json';r=readj(status_path)
        check(r['status']=='fit_completed' and r['mode']=='all' and r['section']==sid and r['protocol_sha256']==model_sha,'Source native fit not actually complete: '+sid)
        check(sha(status_path)==entries[sid]['RUN_STATUS_sha256'],'Native batch/status identity differs: '+sid)
        hashes={x['file']:x['sha256'] for x in r['outputs']};sets={};source[rel(status_path)]=sha(status_path)
        for role,file in [('bulk','NATIVE_BULK_GENES.tsv.gz'),('regression','NATIVE_REGRESSION_GENES.tsv.gz')]:
            f=p/file;check(sha(f)==hashes[file],'Actual native gene-list hash differs: '+sid+':'+role);genes=readt(f).gene.astype(str);check(genes.is_unique,'Native gene list duplicates');sets[role]=set(genes);source[rel(f)]=sha(f)
        u=unions[section['resource']];u['bulk']|=sets['bulk'];u['regression']|=sets['regression'];u['by_section'].append(sets)
        inputs.append({'resource':section['resource'],'section':sid,'patient_id':section['patient_id'],'RUN_STATUS_path':rel(status_path),'RUN_STATUS_sha256':sha(status_path),'bulk_gene_list_path':rel(p/'NATIVE_BULK_GENES.tsv.gz'),'bulk_gene_list_sha256':hashes['NATIVE_BULK_GENES.tsv.gz'],'n_bulk_symbols':len(sets['bulk']),'regression_gene_list_path':rel(p/'NATIVE_REGRESSION_GENES.tsv.gz'),'regression_gene_list_sha256':hashes['NATIVE_REGRESSION_GENES.tsv.gz'],'n_regression_symbols':len(sets['regression']),'n_section_direct_union':len(sets['bulk']|sets['regression'])})
    frames=[];counts=[]
    for resource,p in panel.groupby('resource',sort=False):
        u=unions[resource];x=make_membership(p,u['bulk'],u['regression']);frames.append(x);symbols=set(x.loc[x.broad,'gene_symbol'].dropna());direct=u['bulk']|u['regression'];kept=x[x.retain_outside_direct_fit]
        counts.append({'resource':resource,'native_sections_used':len(u['by_section']),'full_geneID_rows':len(x),'original_broad_geneID_rows':int(x.broad.sum()),'original_broad_unique_nonmissing_symbols':len(symbols),'native_bulk_union_symbols':len(u['bulk']),'native_regression_union_symbols':len(u['regression']),'direct_fit_union_symbols':len(direct),'excluded_broad_geneID_rows':int((x.broad & x.in_direct_fit_symbol_union).sum()),'retained_geneID_rows':len(kept),'retained_unique_nonmissing_symbols':int(kept.gene_symbol.nunique()),'retained_missing_symbol_rows':int(kept.symbol_missing_in_source_panel.sum())})
        gene_rows=[{'gene_symbol':g,'in_bulk_union':g in u['bulk'],'in_regression_union':g in u['regression'],'native_sections_using_directly':sum(g in a['bulk'] or g in a['regression'] for a in u['by_section'])} for g in sorted(direct)]
        save(out/f'EXACT_DIRECT_FIT_SYMBOL_UNION_{resource}.tsv',pd.DataFrame(gene_rows))
    save(out/'FULL_PANEL_MEMBERSHIP.tsv.gz',pd.concat(frames,ignore_index=True));save(out/'PANEL_COUNTS.tsv',pd.DataFrame(counts));save(out/'NATIVE_GENE_SOURCE_PROVENANCE.tsv',pd.DataFrame(inputs))
    names=['FULL_PANEL_MEMBERSHIP.tsv.gz','PANEL_COUNTS.tsv','NATIVE_GENE_SOURCE_PROVENANCE.tsv']+[f'EXACT_DIRECT_FIT_SYMBOL_UNION_{r}.tsv' for r in unions]
    lock={'locked_utc':now(),'status':'complete_resource_direct_fit_union_panel_fixed_before_new_mask_outcomes','script_sha256':sha(__file__),'source_hashes':source,'panel_files':{name:sha(out/name) for name in names},'native_sections_used':len(sections),'resource_counts':counts,'rule':'Retain original broad gene-ID rows with exact literal symbol absent from resource all-section native bulk OR regression union; all gene-ID rows sharing an excluded symbol are removed. No new alias/variance/effect selection.','weighted_profile_vectors_opened_before_lock':False,'new_mask_RNA_outcomes_calculated_before_lock':False,'raw_count_matrices_read':False,'new_model_or_weights':False,'interpretation':'Gene-universe sensitivity only; shared offsets/screening and biological covariance remain. Not independent held-out validation.'}
    writej(lock_path,lock);return lock

def primary_complete():
    p=PRIMARY/'RUN_MANIFEST.json'
    if not p.exists():return False,None
    r=readj(p);check(r.get('status')=='completed_same_RNA_model_conditional_profiles_not_cell_composition_validation','Primary manifest is not actually completed')
    return True,r

def masked_metrics(logs,mask,scale):
    if not mask.any():return {m:dict.fromkeys(VALUES,np.nan) for m in METRICS}
    return compare_profiles(logs,mask,scale)

def summarize_two(comparison,cases):
    comparison=comparison.copy();comparison['included_in_same_finite_patient_summary']=False;comparison['included_in_strict_same_finite_patient_summary']=False;main=[];strict=[]
    for (resource,tier),c in cases.groupby(['resource','tier'],sort=True):
        check(not c.patient_id.duplicated().any(),'More than one pair per resource patient')
        for metric in METRICS:
            subset=comparison[comparison.resource.eq(resource)&comparison.tier.eq(tier)&comparison.metric.eq(metric)];common=[];sc=[]
            for case in c.itertuples(index=False):
                row=subset[subset.case_id.eq(case.case_id)];check(len(row)==2 and set(row.control)==set(CONTROLS),'Two controls not retained per attempt')
                good=bool(case.eligible and np.isfinite(row[list(VALUES)].to_numpy(float)).all())
                if good:common.append(case.case_id)
                if good and case.stricter_distribution_flag:sc.append(case.case_id)
            base=comparison.resource.eq(resource)&comparison.tier.eq(tier)&comparison.metric.eq(metric)
            comparison.loc[base & comparison.case_id.isin(common),'included_in_same_finite_patient_summary']=True;comparison.loc[base & comparison.case_id.isin(sc),'included_in_strict_same_finite_patient_summary']=True
            for ids,scope,dest in [(common,'mean_calibration_common_finite_patients',main),(sc,'strict_KS_common_finite_patients',strict)]:
                for control in CONTROLS:
                    g=subset[subset.case_id.isin(ids)&subset.control.eq(control)];n=g.patient_id.nunique();check(len(g)==n,'Independent patient denominator duplicated')
                    dest.append({'resource':resource,'tier':tier,'metric':metric,'control':control,'summary_scope':scope,'attempted_patients':int(c.patient_id.nunique()),'n_same_finite_patients':int(n),'same_finite_patient_ids':','.join(sorted(g.patient_id.unique())),'near_median':float(g.near_value.median()) if n else np.nan,'far_median':float(g.far_value.median()) if n else np.nan,'median_patient_far_minus_near':float(g.far_minus_near.median()) if n else np.nan,'patient_delta_min':float(g.far_minus_near.min()) if n else np.nan,'patient_delta_max':float(g.far_minus_near.max()) if n else np.nan,'n_delta_positive':int((g.far_minus_near>0).sum()),'n_delta_negative':int((g.far_minus_near<0).sum()),'status':'descriptive_gene_universe_sensitivity' if n else 'unavailable_no_common_finite_patients'})
    return comparison,pd.DataFrame(main),pd.DataFrame(strict)

def execute_saved_vector_sensitivity(out,lock,primary):
    check(sha(__file__)==lock['script_sha256'],'Script differs from panel freeze')
    for name,digest in lock['panel_files'].items():check(sha(out/name)==digest,'Panel artifact changed')
    for name,digest in lock['source_hashes'].items():check(sha(ROOT/name)==digest,'Panel source changed')
    check(datetime.fromisoformat(now())>datetime.fromisoformat(lock['locked_utc']),'Panel must precede outcome calculation')
    for name,digest in primary['outputs'].items():check(sha(PRIMARY/name)==digest,'Completed primary output hash differs: '+name)
    weights_lock_path=SUPPORT/'WEIGHTS_LOCK.json';weights_lock=readj(weights_lock_path);check(primary['weights_lock_sha256']==sha(weights_lock_path),'Primary support lock differs')
    for name,digest in weights_lock['tables'].items():check(sha(SUPPORT/name)==digest,'Support metadata/weights changed')
    cases=readt(SUPPORT/'case_eligibility.tsv');pairs=readt(SUPPORT/'SOURCE_PAIR_MANIFEST.tsv');check(len(cases)==primary['n_attempted_cases']==len(pairs)*len(cases.tier.unique()),'Attempted case universe differs')
    panel=readt(out/'FULL_PANEL_MEMBERSHIP.tsv.gz');panels={r:g.reset_index(drop=True) for r,g in panel.groupby('resource',sort=False)}
    oldrun=readj(OLD/'RUN_MANIFEST.json');oldcache={};source_hashes={rel(PRIMARY/'RUN_MANIFEST.json'):sha(PRIMARY/'RUN_MANIFEST.json'),rel(weights_lock_path):sha(weights_lock_path),rel(SUPPORT/'case_eligibility.tsv'):sha(SUPPORT/'case_eligibility.tsv'),rel(SUPPORT/'SOURCE_PAIR_MANIFEST.tsv'):sha(SUPPORT/'SOURCE_PAIR_MANIFEST.tsv'),rel(OLD/'RUN_MANIFEST.json'):sha(OLD/'RUN_MANIFEST.json')}
    vector_path=PRIMARY/'weighted_region_profiles.npz';check(sha(vector_path)==primary['outputs']['weighted_region_profiles.npz'],'Primary full-feature saved-vector hash differs');source_hashes[rel(vector_path)]=sha(vector_path)
    run={'started_utc':now(),'status':'running_saved_vector_gene_universe_sensitivity','analysis_class':'outcome_informed_gene_universe_sensitivity_not_independent_validation','script_sha256':sha(__file__),'PANEL_LOCK_sha256':sha(out/'PANEL_LOCK.json'),'weights_lock_sha256':sha(weights_lock_path),'primary_manifest_sha256':sha(PRIMARY/'RUN_MANIFEST.json'),'no_raw_counts':True,'no_new_fit_weights_spot_sampling_or_thinning':True,'primary_outputs_modified':False,'source_hashes':source_hashes};writej(out/'STARTED_OUTCOMES.json',run)
    rows=[];regions=[];t0=time.monotonic()
    # First weighted NPZ access is deliberately after PANEL_LOCK and all identity checks.
    with np.load(vector_path,allow_pickle=False) as vectors:
        for resource,p in panels.items():
            check(np.array_equal(vectors[resource+'|gene_ids'].astype(str),p.gene_id.to_numpy(str)),'Resource full-feature gene-ID order differs')
            check(np.array_equal(vectors[resource+'|broad_mask'],p.broad.to_numpy(bool)),'Original broad mask differs')
            check(np.array_equal(vectors[resource+'|fixed_gene_scale'],p.fixed_gene_scale.to_numpy(float)),'Original scales differ')
        for case in cases.itertuples(index=False):
            p=panels[case.resource];mask=p.retain_outside_direct_fit.to_numpy(bool);scale=p.fixed_gene_scale.to_numpy(float);order=[(case.section_a,'near'),(case.section_a,'far'),(case.section_b,'near'),(case.section_b,'far')];weighted=[];full=[]
            for sid,band in order:
                if sid not in oldcache:
                    oldpath=OLD/'section_profiles'/(sid+'.npz');check(sha(oldpath)==oldrun['outputs']['section_profiles/'+sid+'.npz']['sha256'],'Original whole-band NPZ hash differs');source_hashes[rel(oldpath)]=sha(oldpath)
                    with np.load(oldpath,allow_pickle=False) as old:
                        check(np.array_equal(old['gene_ids'].astype(str),p.gene_id.to_numpy(str)),'Original full-region gene-ID order differs')
                        oldcache[sid]={b:old['exacthex__stroma__'+b].copy() for b in ['near','far']}
                vector=np.asarray(oldcache[sid][band]);check(vector.shape==(len(p),) and np.isfinite(vector).all() and (vector>=0).all(),'Invalid saved original full-band vector');full.append(profile(vector))
                key=case.case_id+'|'+sid+'|'+band+'|weighted_counts'
                if case.eligible:
                    check(key in vectors,'Eligible primary weighted vector missing');v=np.asarray(vectors[key]);check(v.shape==(len(p),) and np.isfinite(v).all() and (v>=0).all(),'Invalid saved full-feature weighted vector');weighted.append(profile(v));mass=float(v.sum())
                else:
                    check(key not in vectors,'Failed support unexpectedly has a weighted RNA vector');mass=np.nan
                regions.append({'resource':case.resource,'patient_id':case.patient_id,'case_id':case.case_id,'tier':case.tier,'sample_id':sid,'band':band,'n_retained_genes':int(mask.sum()),'weighted_full_feature_mass_used':mass,'original_full_feature_mass_used':float(vector.sum()),'denominator_rule':'full feature mass, before gene mask; never retained-gene sum'})
            metric_map={CONTROLS[0]:masked_metrics(weighted,mask,scale) if case.eligible else {m:dict.fromkeys(VALUES,np.nan) for m in METRICS},CONTROLS[1]:masked_metrics(full,mask,scale)}
            for metric in METRICS:
                for control in CONTROLS:
                    values=metric_map[control][metric];reason='empty_retained_panel' if not mask.any() else 'failed_primary_support' if control==CONTROLS[0] and not case.eligible else 'nonfinite_metric' if not np.isfinite(list(values.values())).all() else 'available'
                    rows.append({k:getattr(case,k) for k in ['resource','patient_id','section_a','section_b','pair_id','case_id','tier']}|{'panel':'v1_broad_outside_resource_all_section_native_bulk_or_regression_symbols','control':control,'metric':metric,'n_genes':int(mask.sum()),'eligible_mean_calibration':bool(case.eligible),'stricter_distribution_flag':bool(case.stricter_distribution_flag),'max_weighted_fraction_KS':float(case.max_weighted_fraction_KS),'calibration_failure_reason':case.failure_reason,'result_status':reason}|values)
    comparison,summary,strict=summarize_two(pd.DataFrame(rows),cases)
    check(len(comparison)==len(cases)*len(METRICS)*len(CONTROLS),'Attempted outcome rows lost')
    check(comparison[~comparison.eligible_mean_calibration & comparison.control.eq(CONTROLS[0])][list(VALUES)].isna().all().all(),'Failed weighted outcomes not NA')
    save(out/'patient_control_results.tsv',comparison);save(out/'same_finite_patient_resource_summary.tsv',summary);save(out/'strict_KS_same_finite_patient_summary.tsv',strict);save(out/'REGION_VECTOR_DENOMINATORS.tsv',pd.DataFrame(regions))
    for name,digest in lock['panel_files'].items():check(sha(out/name)==digest,'Panel changed during new-mask calculation')
    check(sha(weights_lock_path)==primary['weights_lock_sha256'],'Primary weights changed')
    output_names=list(lock['panel_files'])+['PANEL_LOCK.json','patient_control_results.tsv','same_finite_patient_resource_summary.tsv','strict_KS_same_finite_patient_summary.tsv','REGION_VECTOR_DENOMINATORS.tsv']
    run.update(completed_utc=now(),elapsed_seconds=time.monotonic()-t0,status='completed_outside_direct_fit_gene_universe_sensitivity',n_attempted_pairs=len(pairs),n_attempted_cases=len(cases),n_patient_control_metric_rows=len(comparison),source_hashes=source_hashes,empty_panel_has_no_new_minimum_gene_threshold=True,primary_results_unchanged=True,limits='Not independent held-out validation: shared full-feature offsets, screening/source availability and biological covariance remain.',outputs={name:sha(out/name) for name in output_names});writej(out/'RUN_MANIFEST.json',run)

def toy(path):
    fixture=pd.DataFrame({'resource':['TOY']*5,'gene_id':['g1','g2','g3','g4','g5'],'gene_symbol':['A','X','X','B','Z'],'broad':[True,True,True,False,True],'fixed_gene_scale':[.25,.5,.75,1.,2.]})
    m=make_membership(fixture,{'X'},{'B'});check(m.loc[m.retain_outside_direct_fit,'gene_id'].tolist()==['g1','g5'],'Duplicate-symbol exclusion fixture failed')
    vectors=[np.array(v,float) for v in [[10,20,30,40,900],[20,10,40,30,900],[12,18,32,38,900],[18,12,38,32,900]]];logs=[profile(v) for v in vectors];mask=m.retain_outside_direct_fit.to_numpy();scale=m.fixed_gene_scale.to_numpy()
    check(np.allclose(logs[0],np.log1p(vectors[0]/vectors[0].sum()*1e6)),'Full-feature normalization differs')
    check(not np.allclose(logs[0][mask],np.log1p(vectors[0][mask]/vectors[0][mask].sum()*1e6)),'Retained-gene denominator wrongly used')
    all_empty=masked_metrics(logs,np.zeros(5,bool),scale);check(all(np.isnan(v) for d in all_empty.values() for v in d.values()),'Empty panel not unavailable')
    one=masked_metrics(logs,np.array([True,False,False,False,False]),scale);check(np.isnan(one['spearman']['near_value']) and np.isfinite(one['scaled_abs_difference']['near_value']),'A new minimum-gene threshold was introduced')
    cases=[];rows=[]
    for resource,n in [('TOY_A',2),('TOY_B',1)]:
        for patient in range(n):
            for tier in ['composition_means','composition_and_qc_means','composition_second_moments_and_qc']:
                cid=f'{resource}|{patient}|{tier}';good=resource=='TOY_A' and patient==0;cases.append(dict(resource=resource,patient_id=str(patient),case_id=cid,tier=tier,eligible=good,stricter_distribution_flag=good and tier=='composition_means'))
                for metric in METRICS:
                    for control in CONTROLS:
                        x=np.nan if control==CONTROLS[0] and not good else 1.;rows.append(dict(resource=resource,patient_id=str(patient),case_id=cid,tier=tier,metric=metric,control=control,near_value=x,far_value=x+1,far_minus_near=1. if np.isfinite(x) else np.nan))
    c,s,t=summarize_two(pd.DataFrame(rows),pd.DataFrame(cases));check(len(s)==len(t)==2*3*3*2,'Summary failed to retain all resource/tier/metric/control rows');check(s[s.resource.eq('TOY_B')].near_median.isna().all(),'No common patient converted to a zero effect');check(t[~t.tier.eq('composition_means')].n_same_finite_patients.eq(0).all(),'Strict subset leaked failed cases')
    writej(path,{'checked_utc':now(),'status':'toy_duplicate_symbol_full_denominator_empty_and_single_gene_common_N_NA_passed','script_sha256':sha(__file__),'source_RNA_or_weighted_vector_files_opened':False,'native_models_or_weights_recomputed':False,'duplicate_symbol_geneID_rows_all_excluded':True,'full_feature_denominator_preserved':True,'empty_panel_all_metrics_NA':True,'single_gene_abs_metric_not_discarded_by_new_threshold':True,'same_finite_patient_summary_rows':len(s),'strict_summary_rows':len(t)})

def main():
    global NATIVE,SUPPORT,PRIMARY,OLD
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,default=DEFAULT_OUT);ap.add_argument('--native',type=Path,default=NATIVE);ap.add_argument('--support',type=Path,default=SUPPORT);ap.add_argument('--primary',type=Path,default=PRIMARY);ap.add_argument('--profiles',type=Path,default=OLD);ap.add_argument('--watch',action='store_true');ap.add_argument('--interval',type=int,default=60);ap.add_argument('--toy',action='store_true');ap.add_argument('--toy-output',type=Path,default=ROOT/'results/outside_direct_fit_toy.json');args=ap.parse_args()
    NATIVE=args.native.resolve();SUPPORT=args.support.resolve();PRIMARY=args.primary.resolve();OLD=args.profiles.resolve()
    if args.toy:toy(args.toy_output);return
    check(args.interval>=60,'Watch interval must be at least60seconds')
    for p in [PRIMARY,SUPPORT,OLD,NATIVE]:check(args.output.resolve()!=p.resolve() and p.resolve() not in args.output.resolve().parents,'Protected primary/model root cannot be destination')
    amendment,hashes=protocol();args.output.mkdir(parents=True,exist_ok=True);phase='waiting_native_completion'
    try:
        while True:
            if (args.output/'RUN_MANIFEST.json').exists():
                existing=readj(args.output/'RUN_MANIFEST.json');check(existing['script_sha256']==sha(__file__),'Completed sensitivity code identity differs')
                check(existing['status']=='completed_outside_direct_fit_gene_universe_sensitivity','Incomplete sensitivity cannot be reused as completed')
                build_or_verify_panel(args.output,hashes)
                for name,digest in existing['source_hashes'].items():check(sha(ROOT/name)==digest,'Completed sensitivity source differs: '+name)
                for name,digest in existing['outputs'].items():check(sha(args.output/name)==digest,'Completed sensitivity output differs')
                return
            ready,batch=native_complete()
            if not ready:
                state={'status':'waiting_all22_native_before_panel_freeze','checked_utc':now(),'last_batch_status':batch.get('status') if batch else None,'weighted_vectors_opened':False,'new_mask_outcomes_calculated':False};writej(args.output/'CURRENT_STATUS.json',state)
                if batch and batch.get('status')=='stopped_without_parameter_relaxation':raise RuntimeError('Native batch truly stopped before panel qualification')
                if not args.watch:return
                time.sleep(args.interval);continue
            phase='freezing_metadata_panel';lock=build_or_verify_panel(args.output,hashes)
            ready,primary=primary_complete()
            if ready:
                phase='computing_saved_vector_sensitivity';execute_saved_vector_sensitivity(args.output,lock,primary);writej(args.output/'CURRENT_STATUS.json',{'status':'completed','completed_utc':now(),'RUN_MANIFEST_sha256':sha(args.output/'RUN_MANIFEST.json')});return
            writej(args.output/'CURRENT_STATUS.json',{'status':'panel_frozen_waiting_primary_profiles','checked_utc':now(),'PANEL_LOCK_sha256':sha(args.output/'PANEL_LOCK.json'),'weighted_vectors_opened':False,'new_mask_outcomes_calculated':False})
            follow=NATIVE/'POSTFIT_FOLLOWON_STATUS.json'
            if follow.exists() and readj(follow).get('status')=='stopped_without_threshold_or_parameter_relaxation':raise RuntimeError('Primary follow-on stopped; no masked outcomes calculated')
            if not args.watch:return
            time.sleep(args.interval)
    except BaseException as e:
        writej(args.output/'CURRENT_STATUS.json',{'status':'stopped_without_fit_weight_mask_or_threshold_relaxation','stopped_utc':now(),'phase':phase,'error':type(e).__name__+': '+str(e),'traceback':traceback.format_exc()});raise

if __name__=='__main__':main()
