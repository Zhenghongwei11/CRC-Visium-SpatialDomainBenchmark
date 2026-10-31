#!/usr/bin/env python3
"""Conditional RNA profiles from previously frozen continuous-support weights."""
from __future__ import annotations
import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_name]='2'
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import scipy
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits
from regional_profiles import counts_for,geometry_for,profile,measures

ROOT=Path(__file__).resolve().parents[1]
SUPPORT=ROOT/'results/author_calibration'
OLD=ROOT/'results/regional_profiles'
DEFAULT_OUT=ROOT/'results/author_profiles'
METRICS=('spearman','scaled_abs_difference','unscaled_abs_difference')
CONTROLS=('frozen_mean_calibrated_weights','unweighted_full_regions','unweighted_model_available_regions','old_spot_matched','old_spot_and_depth_matched')


def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def now():return datetime.now(timezone.utc).isoformat()


def jsave(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def verify_all_controls(w,cov,cases,targets,groups):
    counts={'attempted_cases':len(cases),'successful_cases':0,'infeasible_cases_with_missing_weights':0,'region_constraint_checks':0,'feature_mean_checks':0};maxdiff=0.
    mergekeys=['patient_id','sample_id','band','spot_index','barcode','block'];assert not cov[mergekeys].duplicated().any()
    for _,case in cases.iterrows():
        local=w[w.case_id==case.case_id];target=targets[targets.case_id==case.case_id]
        assert not local[['sample_id','band','spot_index']].duplicated().any()
        if not case.eligible:
            assert not case.solver_success and local.weight.isna().all() and target.shared_target.isna().all();counts['infeasible_cases_with_missing_weights']+=1;continue
        counts['successful_cases']+=1
        data=local.merge(cov,on=mergekeys,validate='many_to_one',suffixes=('','_cov'));assert len(data)==len(local)
        regions=[g for _,g in data.groupby(['sample_id','band'],sort=False)];assert len(regions)==4
        abundances=[r[groups].to_numpy(float) for r in regions];rms=np.maximum(np.sqrt(np.mean([np.mean(x*x,0) for x in abundances],0)),1e-6)
        names=['group_mean:'+g for g in groups];features=[a/rms for a in abundances]
        if case.tier!='composition_means':
            names+=['qc:log1p_total_umis','qc:log1p_detected_genes','qc:mitochondrial_fraction'];features=[np.column_stack([x,r[['log1p_total_umis','log1p_detected_genes','mitochondrial_umi_fraction']].to_numpy()]) for x,r in zip(features,regions)]
        if case.tier=='composition_second_moments_and_qc':
            names+=['group_second_moment:'+g for g in groups];features=[np.column_stack([x,(a/rms)**2]) for x,a in zip(features,abundances)]
        ct=target.set_index('feature').loc[names];achieved=[]
        for region,x in zip(regions,features):
            v=region.weight.to_numpy(float);assert np.isfinite(v).all() and(v>=0).all() and v.max()<=.05+1e-10 and abs(v.sum()-1)<1e-8
            bm=region.groupby('block').weight.sum().to_numpy();assert bm.max()<=1/3+1e-10 and v.sum()**2/np.dot(v,v)>=20-1e-8 and bm.sum()**2/np.dot(bm,bm)>=3-1e-8
            mean=v@x;assert np.allclose(mean,ct.shared_target,atol=1e-6,rtol=0);achieved.append(mean);counts['feature_mean_checks']+=len(names);counts['region_constraint_checks']+=1
        meanrange=float(np.ptp(achieved,axis=0).max());assert meanrange<=1e-6;maxdiff=max(maxdiff,meanrange)
    assert counts['attempted_cases']==12 and counts['successful_cases']==8 and counts['infeasible_cases_with_missing_weights']==4
    counts['max_recomputed_scaled_feature_range']=maxdiff
    return counts


def compare(logs,broad,scale):
    n=measures(logs[0][broad],logs[2][broad],scale[broad]);f=measures(logs[1][broad],logs[3][broad],scale[broad])
    for a,b,rec in [(logs[0][broad],logs[2][broad],n),(logs[1][broad],logs[3][broad],f)]:assert np.isclose(spearmanr(a,b).statistic,rec['spearman'],atol=1e-12,rtol=0)
    return {metric:{'near_value':n[metric],'far_value':f[metric],'far_minus_near':f[metric]-n[metric]} for metric in METRICS}


def main():
    global SUPPORT,OLD
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=DEFAULT_OUT);parser.add_argument('--counts',type=Path,required=True);parser.add_argument('--profiles',type=Path,required=True);parser.add_argument('--support',type=Path,required=True);args=parser.parse_args();out=args.output;SUPPORT=args.support.resolve();OLD=args.profiles.resolve();os.environ['CRC_COUNTS_ROOT']=str(args.counts.resolve())
    if(out/'RUN_MANIFEST.json').exists():raise ValueError('Completed conditional-profile run exists; preserve and version any rerun')
    out.mkdir(parents=True,exist_ok=True)
    slock_path=SUPPORT/'WEIGHTS_LOCK.json';slock=json.loads(slock_path.read_text());srun=json.loads((SUPPORT/'RUN_MANIFEST.json').read_text())
    assert slock['weights_sha256']==sha(SUPPORT/'weights.tsv.gz') and slock['case_eligibility_sha256']==sha(SUPPORT/'case_eligibility.tsv') and not slock['new_weighted_RNA_calculated']
    for name,digest in srun['outputs'].items():assert sha(SUPPORT/name)==digest
    oldrun=json.loads((OLD/'RUN_MANIFEST.json').read_text());panel_path=OLD/'gene_panels.tsv';assert sha(panel_path)==oldrun['outputs']['gene_panels.tsv']['sha256']
    panel=pd.read_csv(panel_path,sep='\t',float_precision='round_trip');panel=panel[panel.resource=='Valdeolivas'];broad=panel.broad.to_numpy(bool);scale=panel.fixed_gene_scale.to_numpy(float)
    assert int(broad.sum())==10198
    w=pd.read_csv(SUPPORT/'weights.tsv.gz',sep='\t',float_precision='round_trip');cov=pd.read_csv(SUPPORT/'spot_covariates.tsv.gz',sep='\t',float_precision='round_trip');cases=pd.read_csv(SUPPORT/'case_eligibility.tsv',sep='\t',float_precision='round_trip');targets=pd.read_csv(SUPPORT/'shared_targets.tsv',sep='\t',float_precision='round_trip')
    verified=verify_all_controls(w,cov,cases,targets,srun['groups']);verified.update(weights_lock_sha256=sha(slock_path),support_run_manifest_sha256=sha(SUPPORT/'RUN_MANIFEST.json'),weights_sha256=sha(SUPPORT/'weights.tsv.gz'),case_eligibility_sha256=sha(SUPPORT/'case_eligibility.tsv'),checked_utc=now(),new_RNA_outcomes_seen=False);jsave(out/'PRE_RNA_INPUT_VERIFICATION.json',verified)
    started=now();assert datetime.fromisoformat(started)>datetime.fromisoformat(slock['locked_utc']);t0=time.monotonic()
    run={'started_utc':started,'status':'running_conditional_profiles_on_frozen_weights','analysis_class':'explicitly_outcome_informed_followup','weights_lock_sha256':sha(slock_path),'weights_sha256':sha(SUPPORT/'weights.tsv.gz'),'case_eligibility_sha256':sha(SUPPORT/'case_eligibility.tsv'),'shared_targets_sha256':sha(SUPPORT/'shared_targets.tsv'),'script_sha256':sha(__file__),'helper_sha256':sha(ROOT/'scripts/regional_profiles.py'),'gene_panel_sha256':sha(panel_path),'broad_genes':int(broad.sum()),'new_gene_selection':False,'integer_thinning':False,'weighted_counts_interpretation':'finite nonnegative fractional weighted sums of raw spot counts; not integer UMIs','distribution_gate_interpretation':'Mean calibration qualifies separately from the stricter marginal-distribution flag; all current successful cases fail maxKS<=0.20','source_sections':[],'software':{'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__}}
    jsave(out/'STARTED.json',run)
    source_manifest_path=ROOT/'config/source_pipeline/source_counts_manifest.json';source_manifest=json.loads(source_manifest_path.read_text());byid={s['sample_id']:s for s in source_manifest['sections']}
    oldfull=pd.read_csv(OLD/'within_patient_summaries.tsv',sep='\t',float_precision='round_trip');oldfull=oldfull[(oldfull.scope=='all_eligible')&(oldfull.resource=='Valdeolivas')&(oldfull.geometry=='exacthex')&(oldfull.compartment=='stroma')&(oldfull.panel=='broad')]
    oldmatch=pd.read_csv(OLD/'matched_sampling_patient_summaries.tsv',sep='\t',float_precision='round_trip');oldmatch=oldmatch[(oldmatch.resource=='Valdeolivas')&(oldmatch.compartment=='stroma')&(oldmatch.panel=='broad')]
    for name in ['within_patient_summaries.tsv','matched_sampling_patient_summaries.tsv']:assert sha(OLD/name)==oldrun['outputs'][name]['sha256']
    run['source_manifest_sha256']=sha(source_manifest_path);run['old_full_patient_summary_sha256']=sha(OLD/'within_patient_summaries.tsv');run['old_sampled_patient_summary_sha256']=sha(OLD/'matched_sampling_patient_summaries.tsv')
    patient_rows=[];comparison_rows=[];region_rows=[];arrays={'gene_ids':panel.gene_id.to_numpy(str),'broad_mask':broad,'fixed_gene_scale':scale}
    with threadpool_limits(limits=2):
        for patient,pcases in cases.groupby('patient_id',sort=True):
            first=pcases.iloc[0];sids=[first.section_a,first.section_b];data={};regions=[]
            for sid in sids:
                s=byid[sid];counts,spots,ids,symbols,qc,members,checks=counts_for(s);assert np.array_equal(ids,panel.gene_id.to_numpy(str));geo,blocks,_=geometry_for(s,spots)
                # Model-available support has the same registered indices for each tier.
                local=w[(w.patient_id==patient)&(w.sample_id==sid)&(w.tier=='composition_means')]
                with np.load(OLD/'section_profiles'/(sid+'.npz')) as old:
                    for band,mask in zip(['near','far'],geo['exacthex:stroma']):assert np.array_equal(np.asarray(counts[mask].sum(0)).ravel(),old['exacthex__stroma__'+band])
                data[sid]=(counts,spots,ids,qc,geo,blocks)
                run['source_sections'].append({'patient_id':patient,'sample_id':sid,'source_files':members,'loader_checks':checks,'spot_input_sha256':sha(ROOT/s['source_root']/'spot_inputs.tsv.gz')})
                for band in ['near','far']:regions.append((sid,band))
            model_logs=[]
            for sid,band in regions:
                counts,spots,ids,qc,geo,blocks=data[sid];rr=w[(w.patient_id==patient)&(w.sample_id==sid)&(w.band==band)&(w.tier=='composition_means')].sort_values('spot_index');ix=rr.spot_index.to_numpy(int)
                assert np.array_equal(rr.barcode.to_numpy(str),spots.barcode.to_numpy(str)[ix]) and geo['exacthex:stroma'][0 if band=='near' else 1][ix].all()
                vector=np.asarray(counts[ix].sum(0)).ravel();model_logs.append(profile(vector))
            model_metrics=compare(model_logs,broad,scale)
            for _,case in pcases.iterrows():
                base={name:case[name] for name in ['resource','patient_id','section_a','section_b','case_id','tier']};base.update(eligible_mean_calibration=bool(case.eligible),stricter_distribution_flag=bool(case.stricter_distribution_flag),max_weighted_fraction_KS=float(case.max_weighted_fraction_KS),calibration_failure_reason=case.failure_reason,profile_status='mean_calibrated_distribution_diagnostic_failed' if case.eligible and not case.stricter_distribution_flag else 'mean_calibrated_distribution_diagnostic_passed' if case.eligible else 'unavailable_calibration_failed')
                logs=[]
                for sid,band in regions:
                    counts,spots,ids,qc,geo,blocks=data[sid];rr=w[(w.case_id==case.case_id)&(w.sample_id==sid)&(w.band==band)].sort_values('spot_index');ix=rr.spot_index.to_numpy(int)
                    assert len(set(ix))==len(ix) and np.array_equal(rr.barcode.to_numpy(str),spots.barcode.to_numpy(str)[ix]);assert geo['exacthex:stroma'][0 if band=='near' else 1][ix].all()
                    suffix=case.case_id+'|'+sid+'|'+band;npz_key='';mass=np.nan;check=np.nan
                    if case.eligible:
                        weight=rr.weight.to_numpy(float);assert np.isfinite(weight).all() and(weight>=0).all();vector=np.asarray(counts[ix].multiply(weight[:,None]).sum(axis=0)).ravel();assert np.isfinite(vector).all() and(vector>=0).all();mass=float(vector.sum());check=float(np.dot(weight,qc['total'][ix]));assert np.isclose(mass,check,rtol=1e-12,atol=1e-8)
                        transformed=profile(vector);logs.append(transformed);npz_key=suffix+'|weighted_counts';arrays[npz_key]=vector;arrays[suffix+'|broad_log1pCPM']=transformed[broad]
                    else:assert rr.weight.isna().all()
                    region_rows.append({**base,'sample_id':sid,'band':band,'n_model_available_spots':len(ix),'n_positive_weight_spots':int((rr.weight>1e-12).sum()) if case.eligible else 0,'fractional_weighted_library_mass':mass,'independent_weighted_spot_library_sum':check,'npz_weighted_counts_key':npz_key})
                metrics=compare(logs,broad,scale) if case.eligible else {m:{'near_value':np.nan,'far_value':np.nan,'far_minus_near':np.nan} for m in METRICS}
                for metric in METRICS:
                    patient_rows.append({**base,'panel':'broad','n_genes':int(broad.sum()),'metric':metric,**metrics[metric]})
                    full=oldfull[(oldfull.patient_id==patient)&(oldfull.metric==metric)];assert len(full)==1
                    controls={'frozen_mean_calibrated_weights':metrics[metric],'unweighted_model_available_regions':model_metrics[metric],'unweighted_full_regions':{k:float(full.iloc[0][k]) for k in ['near_value','far_value','far_minus_near']}}
                    for label,oldcontrol in [('old_spot_matched','spot_matched'),('old_spot_and_depth_matched','spot_and_depth_matched')]:
                        value=oldmatch[(oldmatch.patient_id==patient)&(oldmatch.metric==metric)&(oldmatch.control==oldcontrol)];assert len(value)==1;controls[label]={k:float(value.iloc[0][k]) for k in ['near_value','far_value','far_minus_near']}
                    for control,values in controls.items():comparison_rows.append({**base,'panel':'broad','metric':metric,'control':control,'included_in_same_patient_summary':bool(case.eligible),**values})
            del data,counts
            print('CONDITIONAL_PROFILES_DONE',patient,'eligible tiers',int(pcases.eligible.sum()),flush=True)
    def save(rows,name):
        frame=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows);frame.to_csv(out/name,sep='\t',index=False,na_rep='NA');return frame
    result=save(patient_rows,'weighted_profile_patient_results.tsv');comparison=save(comparison_rows,'same_patient_control_comparisons.tsv');save(region_rows,'region_weighted_libraries.tsv');np.savez_compressed(out/'weighted_region_profiles.npz',**arrays)
    summaries=[];strict=[]
    for tier,tg in comparison.groupby('tier',sort=False):
        good=tg[tg.included_in_same_patient_summary];n=int(good.patient_id.nunique());ids=','.join(sorted(good.patient_id.unique()))
        for (control,metric),g in good.groupby(['control','metric'],sort=True):
            assert g.patient_id.nunique()==len(g)==n
            summaries.append({'resource':'Valdeolivas','tier':tier,'control':control,'metric':metric,'attempted_patients':4,'n_same_common_patients':n,'same_common_patient_ids':ids,'near_median':g.near_value.median(),'far_median':g.far_value.median(),'median_patient_far_minus_near':g.far_minus_near.median(),'patient_delta_min':g.far_minus_near.min(),'patient_delta_max':g.far_minus_near.max(),'n_delta_positive':int((g.far_minus_near>0).sum()),'n_delta_negative':int((g.far_minus_near<0).sum()),'interpretation':'descriptive_common_patient_comparison_of_different_weighted_targets_not_causal_composition_attribution'})
        for metric in METRICS:
            g=result[(result.tier==tier)&(result.metric==metric)&result.eligible_mean_calibration&result.stricter_distribution_flag]
            strict.append({'resource':'Valdeolivas','tier':tier,'metric':metric,'n_strict_distribution_common_patients':g.patient_id.nunique(),'median_patient_far_minus_near':g.far_minus_near.median() if len(g) else np.nan,'status':'unavailable_no_case_passes_stricter_distribution_diagnostic' if not len(g) else 'descriptive_stricter_distribution_subset'})
    save(summaries,'same_common_patient_resource_summary.tsv');save(strict,'stricter_distribution_support_summary.tsv')
    assert len(result)==36 and len(comparison)==180 and result[~result.eligible_mean_calibration][['near_value','far_value','far_minus_near']].isna().all().all()
    assert slock['weights_sha256']==sha(SUPPORT/'weights.tsv.gz') and slock['case_eligibility_sha256']==sha(SUPPORT/'case_eligibility.tsv')
    run.update({'completed_utc':now(),'elapsed_seconds':time.monotonic()-t0,'status':'completed_conditional_RNA_on_frozen_mean_calibration_not_full_composition_adjustment','n_attempted_cases':12,'n_mean_eligible_cases':int(cases.eligible.sum()),'n_strict_distribution_cases':int(cases.stricter_distribution_flag.sum()),'source_weight_files_unchanged':True,'failed_case_metrics_preserved_as_NA':True,'RNA_profile_units':'log1p CPM with total weighted count mass across full original feature universe as denominator','outputs':{p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name not in ['RUN_MANIFEST.json','STARTED.json']}});jsave(out/'RUN_MANIFEST.json',run)
    print(pd.DataFrame(summaries).query("control=='frozen_mean_calibrated_weights'").to_string(index=False),flush=True)

if __name__=='__main__':main()
