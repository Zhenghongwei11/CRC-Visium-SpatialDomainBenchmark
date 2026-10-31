#!/usr/bin/env python3
"""Full-original-feature weighted profiles after a frozen common-RCTD support lock."""
from __future__ import annotations
import os
for _env in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_env]='2'
import argparse,itertools,json,sys,time
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import scipy
from threadpoolctl import threadpool_limits
from regional_profiles import counts_for,geometry_for,profile,measures
from rctd_calibration import ROOT,DOC,OLD,DEFAULT_OUT as DEFAULT_SUPPORT,TIERS,QC,TOL,sha,now,jsave,read_table,source_pairs,feature_arrays,effective_n,weighted_ks

DEFAULT_OUT=ROOT/'results/rctd_profiles'
METRICS=('spearman','scaled_abs_difference','unscaled_abs_difference')
CONTROLS=('frozen_mean_calibrated_weights','unweighted_full_regions','unweighted_model_available_regions','old_spot_matched','old_spot_and_depth_matched')
VALUES=('near_value','far_value','far_minus_near')

def compare_profiles(logs,broad,scale):
    missing={m:dict.fromkeys(VALUES,np.nan) for m in METRICS}
    if len(logs)!=4 or any(not np.isfinite(v[broad]).all() for v in logs):return missing
    n=measures(logs[0][broad],logs[2][broad],scale[broad]);f=measures(logs[1][broad],logs[3][broad],scale[broad])
    return {m:dict(near_value=n[m],far_value=f[m],far_minus_near=f[m]-n[m]) for m in METRICS}

def recompute_support(w,cov,cases,targets,groups):
    key=['resource','patient_id','pair_id','sample_id','band','spot_index','barcode','block'];assert not cov[key].duplicated().any()
    checks={'attempted_cases':len(cases),'eligible_cases':0,'ineligible_cases_kept':0,'regions_checked':0,'feature_means_checked':0,'max_recomputed_scaled_mean_range':0.,'weighted_marginal_KS_checks':0}
    for case in cases.itertuples(index=False):
        local=w[w.case_id.eq(case.case_id)];target=targets[targets.case_id.eq(case.case_id)]
        assert not local[['sample_id','band','spot_index']].duplicated().any()
        if not case.eligible:
            checks['ineligible_cases_kept']+=1
            # Partial finite failed-optimizer output may be retained diagnostically; no failed case is used for RNA weighting.
            continue
        checks['eligible_cases']+=1
        data=local.merge(cov,on=key,validate='many_to_one',suffixes=('','_cov'));assert len(data)==len(local)
        regions=[]
        for sid,band in [(case.section_a,'near'),(case.section_a,'far'),(case.section_b,'near'),(case.section_b,'far')]:
            r=data[data.sample_id.eq(sid)&data.band.eq(band)].sort_values('spot_index');assert len(r)>=20 and r.block.nunique()>=3
            regions.append(dict(sample_id=sid,band=band,index=r.spot_index.to_numpy(int),block=r.block.to_numpy(str),shares=r[groups].to_numpy(float),qc=r[list(QC)].to_numpy(float),weight=r.weight.to_numpy(float)))
        features,names,_=feature_arrays(regions,case.tier,groups);ct=target.set_index('feature').loc[names];achieved=[]
        for r,x in zip(regions,features):
            v=r['weight'];assert np.isfinite(v).all() and (v>=0).all() and v.max()<=.05+TOL and abs(v.sum()-1)<=TOL
            mass=np.array([v[r['block']==b].sum() for b in np.unique(r['block'])]);assert mass.max()<=1/3+TOL and effective_n(v)>=20-TOL and effective_n(mass)>=3-TOL
            m=v@x;assert np.allclose(m,ct.shared_target.to_numpy(float),rtol=0,atol=1e-6);achieved.append(m);checks['regions_checked']+=1;checks['feature_means_checked']+=len(names)
        spread=float(np.ptp(achieved,axis=0).max());assert spread<=1e-6 and np.isclose(spread,case.max_scaled_feature_mean_range,atol=1e-10,rtol=0);checks['max_recomputed_scaled_mean_range']=max(checks['max_recomputed_scaled_mean_range'],spread)
        ks=[]
        for i,j in itertools.combinations(range(4),2):
            for k in range(len(groups)):
                ks.append(weighted_ks(regions[i]['shares'][:,k],regions[i]['weight'],regions[j]['shares'][:,k],regions[j]['weight']));checks['weighted_marginal_KS_checks']+=1
        maximum=float(max(ks));assert np.isclose(maximum,case.max_weighted_fraction_KS,atol=1e-12,rtol=0) and bool(maximum<=.2)==bool(case.stricter_distribution_flag)
    return checks

def summarize_controls(comparison,cases):
    comparison=comparison.copy();comparison['included_in_same_patient_summary']=False;comparison['included_in_strict_same_patient_summary']=False;main=[];strict=[]
    for (resource,tier),cg in cases.groupby(['resource','tier'],sort=True):
        attempted=int(cg.patient_id.nunique());assert not cg.patient_id.duplicated().any()
        for metric in METRICS:
            selected=comparison[comparison.resource.eq(resource)&comparison.tier.eq(tier)&comparison.metric.eq(metric)]
            keep=[];strictkeep=[]
            for c in cg.itertuples(index=False):
                local=selected[selected.case_id.eq(c.case_id)];assert len(local)==len(CONTROLS) and set(local.control)==set(CONTROLS)
                good=bool(c.eligible and np.isfinite(local[list(VALUES)].to_numpy(float)).all())
                if good:keep.append(c.case_id)
                if good and c.stricter_distribution_flag:strictkeep.append(c.case_id)
            mask=comparison.resource.eq(resource)&comparison.tier.eq(tier)&comparison.metric.eq(metric)
            comparison.loc[mask&comparison.case_id.isin(keep),'included_in_same_patient_summary']=True
            comparison.loc[mask&comparison.case_id.isin(strictkeep),'included_in_strict_same_patient_summary']=True
            for scope,ids,destination in [('mean_calibration_common_finite_patients',keep,main),('strict_KS_common_finite_patients',strictkeep,strict)]:
                for control in CONTROLS:
                    g=selected[selected.case_id.isin(ids)&selected.control.eq(control)];n=g.patient_id.nunique();assert len(g)==n
                    row={'resource':resource,'tier':tier,'control':control,'metric':metric,'summary_scope':scope,'attempted_patients':attempted,'n_same_common_patients':int(n),'same_common_patient_ids':','.join(sorted(g.patient_id.unique())),'status':'descriptive_common_patient_comparison' if n else 'unavailable_no_common_finite_patients','near_median':float(g.near_value.median()) if n else np.nan,'far_median':float(g.far_value.median()) if n else np.nan,'median_patient_far_minus_near':float(g.far_minus_near.median()) if n else np.nan,'patient_delta_min':float(g.far_minus_near.min()) if n else np.nan,'patient_delta_max':float(g.far_minus_near.max()) if n else np.nan,'n_delta_positive':int((g.far_minus_near>0).sum()),'n_delta_negative':int((g.far_minus_near<0).sum()),'interpretation':'model_conditional_RNA_mixture_sensitivity_not_true_cell_adjustment_or_causal_attribution'};destination.append(row)
    return comparison,pd.DataFrame(main),pd.DataFrame(strict)

def toy_check(path):
    # Synthetic arithmetic fixture, not reference cells or biological data.
    x=np.array([[10,0,90,0],[0,10,0,90]],float);v=np.array([.25,.75]);weighted=np.asarray((x*v[:,None]).sum(0));assert weighted.sum()==v@x.sum(1)==100
    transformed=profile(weighted);assert np.allclose(transformed,np.log1p(weighted/100*1e6)) and not np.allclose(transformed[:2],np.log1p(weighted[:2]/weighted[:2].sum()*1e6))
    cases=[];rows=[]
    for resource,n in [('TOY_A',2),('TOY_B',1)]:
        for patient in range(n):
            for tier in TIERS:
                cid=f'{resource}|{patient}|{tier}';good=resource=='TOY_A' and patient==0;cases.append(dict(resource=resource,patient_id=str(patient),tier=tier,case_id=cid,eligible=good,stricter_distribution_flag=good and tier==TIERS[0]))
                for metric in METRICS:
                    for control in CONTROLS:
                        value=np.nan if not good and control==CONTROLS[0] else 1.
                        rows.append(dict(resource=resource,patient_id=str(patient),tier=tier,case_id=cid,metric=metric,control=control,near_value=value,far_value=value+1,far_minus_near=1. if np.isfinite(value) else np.nan))
    c,m,s=summarize_controls(pd.DataFrame(rows),pd.DataFrame(cases));assert len(m)==2*len(TIERS)*len(METRICS)*len(CONTROLS)==len(s)
    assert m[m.resource.eq('TOY_B')].n_same_common_patients.eq(0).all() and m[m.resource.eq('TOY_B')].median_patient_far_minus_near.isna().all()
    assert s[~s.tier.eq(TIERS[0])].n_same_common_patients.eq(0).all() and s[~s.tier.eq(TIERS[0])].near_median.isna().all()
    assert c[c.resource.eq('TOY_A')&c.patient_id.eq('1')].included_in_same_patient_summary.eq(False).all()
    from rctd_calibration import calibrate_case
    rng=np.random.default_rng(20261026);groups=['g'+str(i) for i in range(7)];shares=rng.dirichlet(np.ones(7),24);qc=np.column_stack([np.log1p(np.arange(24)+2000),np.log1p(np.arange(24)+500),np.full(24,.02)])
    regions=[dict(sample_id='S'+str(i//2),band=['near','far'][i%2],index=np.arange(24),barcodes=np.array(['b'+str(j) for j in range(24)]),block=np.repeat(['a','b','c','d'],6),shares=shares.copy(),qc=qc.copy(),native_sum=np.ones(24),n_original=24) for i in range(4)]
    fixture_pair=dict(resource='TOY',patient_id='P',section_a='S0',section_b='S1',pair_id='TOY_PAIR');ww=[];tt=[];cc=[];cv=[]
    for r in regions:
        for j,idx in enumerate(r['index']):cv.append({**fixture_pair,'sample_id':r['sample_id'],'band':r['band'],'spot_index':int(idx),'barcode':r['barcodes'][j],'block':r['block'][j],**{g:float(r['shares'][j,k]) for k,g in enumerate(groups)},**{name:float(r['qc'][j,k]) for k,name in enumerate(QC)}})
    for tier in TIERS:
        rr,_=calibrate_case(regions,{**fixture_pair,'case_id':'TOY_PAIR|'+tier,'tier':tier},groups);ww+=rr['weights'];tt+=rr['shared_targets'];cc+=rr['case_eligibility']
    support_reverification=recompute_support(pd.DataFrame(ww),pd.DataFrame(cv),pd.DataFrame(cc),pd.DataFrame(tt),groups)
    assert support_reverification['eligible_cases']==len(TIERS) and support_reverification['regions_checked']==4*len(TIERS)
    path.parent.mkdir(parents=True,exist_ok=True);jsave(path,{'checked_utc':now(),'support_table_schema_and_moment_ESS_KS_reverification':support_reverification,'status':'toy_full_denominator_and_common_patient_failure_NA_checks_passed','RNA_or_model_files_read':False,'RNA_profiles_executed':False,'model_fit_run':False,'synthetic_full_library_denominator_verified':True,'all_resource_tier_metric_control_rows_retained':len(m),'strict_rows_retained':len(s),'failed_metrics_not_replaced_by_zero':True,'script_sha256':sha(__file__)})

def main():
    global OLD
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--support',type=Path,default=DEFAULT_SUPPORT);ap.add_argument('--profiles',type=Path,default=OLD);ap.add_argument('--counts',type=Path,default=ROOT/'original_inputs/counts');ap.add_argument('--output',type=Path,default=DEFAULT_OUT);ap.add_argument('--toy',action='store_true');ap.add_argument('--toy-output',type=Path,default=ROOT/'results/profiles_toy.json');args=ap.parse_args()
    if args.toy:toy_check(args.toy_output);return
    OLD=args.profiles.resolve();os.environ['CRC_COUNTS_ROOT']=str(args.counts.resolve())
    support=args.support.resolve();out=args.output
    for p in [OLD,support]:
        assert out.resolve()!=p.resolve() and p.resolve() not in out.resolve().parents,'Protected existing output root'
    if (out/'RUN_MANIFEST.json').exists():raise ValueError('Completed result exists; preserve and version rerun')
    slock_path=support/'WEIGHTS_LOCK.json';lock=json.loads(slock_path.read_text());srun=json.loads((support/'RUN_MANIFEST.json').read_text());assert not lock['new_weighted_RNA_calculated'] and srun['status']=='completed_support_only_no_weighted_RNA'
    for name,digest in srun['outputs'].items():assert sha(support/name)==digest
    for name,digest in lock['tables'].items():assert sha(support/name)==digest
    for name,digest in lock['input_hashes'].items():assert sha(ROOT/name)==digest
    assert sha(ROOT/'scripts/rctd_calibration.py')==lock['script_sha256']
    w=read_table(support/'weights.tsv.gz');cov=read_table(support/'spot_covariates.tsv.gz');cases=read_table(support/'case_eligibility.tsv');targets=read_table(support/'shared_targets.tsv');pairs=read_table(support/'SOURCE_PAIR_MANIFEST.tsv');groups=srun['groups']
    assert len(cases)==len(pairs)*len(TIERS) and set(cases.pair_id)==set(pairs.pair_id)
    verified=recompute_support(w,cov,cases,targets,groups);verified.update(checked_utc=now(),weights_lock_sha256=sha(slock_path),new_RNA_outcomes_seen=False)
    oldrun=json.loads((OLD/'RUN_MANIFEST.json').read_text());panel_path=OLD/'gene_panels.tsv';oldfull_path=OLD/'within_patient_summaries.tsv';oldmatch_path=OLD/'matched_sampling_patient_summaries.tsv'
    for p in [panel_path,oldfull_path,oldmatch_path]:assert sha(p)==oldrun['outputs'][p.name]['sha256']
    allpanel=read_table(panel_path);panels={r:g.reset_index(drop=True) for r,g in allpanel.groupby('resource',sort=False)}
    oldfull=read_table(oldfull_path);oldfull=oldfull[oldfull.scope.eq('all_eligible')&oldfull.geometry.eq('exacthex')&oldfull.compartment.eq('stroma')&oldfull.panel.eq('broad')]
    oldmatch=read_table(oldmatch_path);oldmatch=oldmatch[oldmatch.compartment.eq('stroma')&oldmatch.panel.eq('broad')]
    manifest_path=ROOT/'config/source_pipeline/source_counts_manifest.json';byid={s['sample_id']:s for s in json.loads(manifest_path.read_text())['sections']}
    out.mkdir(parents=True,exist_ok=True);jsave(out/'PRE_RNA_INPUT_VERIFICATION.json',verified)
    started=now();assert datetime.fromisoformat(started)>datetime.fromisoformat(lock['locked_utc'])
    run={'started_utc':started,'status':'running_profiles_after_fixed_common_RCTD_support','script_sha256':sha(__file__),'weights_lock_sha256':sha(slock_path),'support_manifest_sha256':sha(support/'RUN_MANIFEST.json'),'support_input_hashes':lock['input_hashes'],'gene_panel_sha256':sha(panel_path),'old_full_patient_summary_sha256':sha(oldfull_path),'old_matched_patient_summary_sha256':sha(oldmatch_path),'helper_sha256':sha(ROOT/'scripts/regional_profiles.py'),'new_gene_selection':False,'integer_thinning':False,'profile_denominator':'full original feature weighted raw count mass, not shared RCTD genes or broad-gene subset','analysis_class':'outcome_informed_same_RNA_model_conditional_sensitivity','source_sections':[],'resource_panels':{},'software':{'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__}}
    jsave(out/'STARTED.json',run);start=time.monotonic();patient_rows=[];comparison=[];region_rows=[];arrays={}
    for resource,panel in panels.items():
        assert panel.gene_id.is_unique and np.isfinite(panel.fixed_gene_scale).all() and (panel.fixed_gene_scale>0).all();arrays[resource+'|gene_ids']=panel.gene_id.to_numpy(str);arrays[resource+'|broad_mask']=panel.broad.to_numpy(bool);arrays[resource+'|fixed_gene_scale']=panel.fixed_gene_scale.to_numpy(float);run['resource_panels'][resource]={'full_gene_ids':len(panel),'broad_genes':int(panel.broad.sum())}
    with threadpool_limits(limits=2):
        for pair in pairs.itertuples(index=False):
            resource=pair.resource;patient=pair.patient_id;panel=panels[resource];broad=panel.broad.to_numpy(bool);scale=panel.fixed_gene_scale.to_numpy(float);data={};order=[(pair.section_a,'near'),(pair.section_a,'far'),(pair.section_b,'near'),(pair.section_b,'far')];pcases=cases[cases.pair_id.eq(pair.pair_id)]
            for sid in [pair.section_a,pair.section_b]:
                section=byid[sid];counts,spots,ids,symbols,qc,members,checks=counts_for(section);assert np.array_equal(ids,panel.gene_id.to_numpy(str))
                geo,blocks,gpath=geometry_for(section,spots)
                oldnpz=OLD/'section_profiles'/(sid+'.npz');assert sha(oldnpz)==oldrun['outputs']['section_profiles/'+sid+'.npz']['sha256']
                with np.load(oldnpz) as old:
                    for band,mask in zip(['near','far'],geo['exacthex:stroma']):assert np.array_equal(np.asarray(counts[mask].sum(0)).ravel(),old['exacthex__stroma__'+band])
                data[sid]=(counts,spots,qc,geo,blocks);run['source_sections'].append({'resource':resource,'patient_id':patient,'sample_id':sid,'source_files':members,'loader_checks':checks,'source_spot_inputs_sha256':sha(ROOT/section['source_root']/'spot_inputs.tsv.gz'),'geometry_sha256':sha(gpath)})
            model_logs=[]
            for sid,band in order:
                counts,spots,qc,geo,blocks=data[sid];r=cov[cov.pair_id.eq(pair.pair_id)&cov.sample_id.eq(sid)&cov.band.eq(band)].sort_values('spot_index');ix=r.spot_index.to_numpy(int)
                assert np.array_equal(r.barcode.to_numpy(str),spots.barcode.to_numpy(str)[ix]) and geo['exacthex:stroma'][0 if band=='near' else 1][ix].all()
                expected_qc=np.column_stack([np.log1p(qc['total'][ix]),np.log1p(qc['detected'][ix]),qc['mito'][ix]/qc['total'][ix]])
                assert np.allclose(r[list(QC)].to_numpy(float),expected_qc,rtol=0,atol=1e-12),'Frozen QC differs from protected full-feature RNA source'
                vector=np.asarray(counts[ix].sum(0)).ravel();model_logs.append(profile(vector))
            modelmetrics=compare_profiles(model_logs,broad,scale)
            for case in pcases.itertuples(index=False):
                base={name:getattr(case,name) for name in ['resource','patient_id','section_a','section_b','pair_id','case_id','tier']};base.update(eligible_mean_calibration=bool(case.eligible),stricter_distribution_flag=bool(case.stricter_distribution_flag),max_weighted_fraction_KS=float(case.max_weighted_fraction_KS),calibration_failure_reason=case.failure_reason,profile_status='mean_calibrated_strict_KS_passed' if case.eligible and case.stricter_distribution_flag else 'mean_calibrated_strict_KS_failed' if case.eligible else 'unavailable_calibration_failed');logs=[]
                for sid,band in order:
                    counts,spots,qc,geo,blocks=data[sid];r=w[w.case_id.eq(case.case_id)&w.sample_id.eq(sid)&w.band.eq(band)].sort_values('spot_index');ix=r.spot_index.to_numpy(int)
                    assert np.array_equal(r.barcode.to_numpy(str),spots.barcode.to_numpy(str)[ix]) and geo['exacthex:stroma'][0 if band=='near' else 1][ix].all()
                    mass=np.nan;independent=np.nan;key=''
                    if case.eligible:
                        v=r.weight.to_numpy(float);assert np.isfinite(v).all() and (v>=-TOL).all()
                        # No clipping, imputation or thinning. Materially negative values are invalid, not corrected.
                        assert (v>=0).all();vector=np.asarray(counts[ix].multiply(v[:,None]).sum(0)).ravel();assert np.isfinite(vector).all() and (vector>=0).all()
                        mass=float(vector.sum());independent=float(v@qc['total'][ix]);assert np.isclose(mass,independent,rtol=1e-12,atol=1e-8)
                        transformed=profile(vector);logs.append(transformed);key=case.case_id+'|'+sid+'|'+band;arrays[key+'|weighted_counts']=vector;arrays[key+'|broad_log1pCPM']=transformed[broad]
                    region_rows.append({**base,'sample_id':sid,'band':band,'n_model_available_spots':len(ix),'n_positive_weight_spots':int((r.weight>1e-12).sum()) if case.eligible else 0,'fractional_weighted_full_library':mass,'independent_weighted_full_spot_library':independent,'npz_prefix':key})
                metrics=compare_profiles(logs,broad,scale) if case.eligible else {m:dict.fromkeys(VALUES,np.nan) for m in METRICS}
                for metric in METRICS:
                    patient_rows.append({**base,'panel':'broad','n_genes':int(broad.sum()),'metric':metric,**metrics[metric]})
                    original=oldfull[oldfull.resource.eq(resource)&oldfull.patient_id.eq(patient)&oldfull.metric.eq(metric)];assert len(original)==1
                    controls={'frozen_mean_calibrated_weights':metrics[metric],'unweighted_model_available_regions':modelmetrics[metric],'unweighted_full_regions':{k:float(original.iloc[0][k]) for k in VALUES}}
                    for label,oldname in [('old_spot_matched','spot_matched'),('old_spot_and_depth_matched','spot_and_depth_matched')]:
                        oldrow=oldmatch[oldmatch.resource.eq(resource)&oldmatch.patient_id.eq(patient)&oldmatch.metric.eq(metric)&oldmatch.control.eq(oldname)];assert len(oldrow)==1;controls[label]={k:float(oldrow.iloc[0][k]) for k in VALUES}
                    for control in CONTROLS:comparison.append({**base,'panel':'broad','metric':metric,'control':control,**controls[control]})
            del data,counts
            print('RCTD_CONDITIONAL_PROFILES',resource,patient,'eligible_tiers',int(pcases.eligible.sum()),flush=True)
    result=pd.DataFrame(patient_rows);comparison,main_summary,strict_summary=summarize_controls(pd.DataFrame(comparison),cases)
    assert len(result)==len(cases)*len(METRICS) and len(comparison)==len(result)*len(CONTROLS)
    assert result[~result.eligible_mean_calibration][list(VALUES)].isna().all().all()
    for frame,name in [(result,'weighted_profile_patient_results.tsv'),(comparison,'same_patient_control_comparisons.tsv'),(pd.DataFrame(region_rows),'region_weighted_libraries.tsv'),(main_summary,'same_common_patient_resource_summary.tsv'),(strict_summary,'stricter_distribution_support_summary.tsv')]:frame.to_csv(out/name,sep='\t',index=False,na_rep='NA')
    np.savez_compressed(out/'weighted_region_profiles.npz',**arrays)
    for name,digest in lock['tables'].items():assert sha(support/name)==digest
    run.update(completed_utc=now(),elapsed_seconds=time.monotonic()-start,status='completed_same_RNA_model_conditional_profiles_not_cell_composition_validation',n_attempted_pairs=len(pairs),n_attempted_cases=len(cases),n_mean_eligible_cases=int(cases.eligible.sum()),n_strict_distribution_cases=int(cases.stricter_distribution_flag.sum()),weights_unchanged=True,failed_case_weighted_metrics_preserved_as_NA=True,source_counts='protected original full matrices via counts_for; staged model counts never used',outputs={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name not in ['STARTED.json','RUN_MANIFEST.json']});jsave(out/'RUN_MANIFEST.json',run)

if __name__=='__main__':main()
