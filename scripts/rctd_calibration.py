#!/usr/bin/env python3
"""Frozen common-RCTD-share calibration from full-section metadata; no RNA outcomes."""
from __future__ import annotations
import os
for _env in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_env]='2'
import argparse,itertools,json,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
from threadpoolctl import threadpool_limits
from calibration_math import program,effective_n,weighted_ks,sha,now,jsave
from regional_profiles import geometry_for

ROOT=Path(__file__).resolve().parents[1]
DOC=ROOT/'config'
MODEL=ROOT/'results/rctd'
OLD=ROOT/'results/regional_profiles'
DEFAULT_OUT=ROOT/'results/rctd_calibration'
TIERS=('composition_means','composition_and_qc_means','composition_second_moments_and_qc')
QC=('log1p_total_umis','log1p_detected_genes','mitochondrial_umi_fraction')
TOL=1e-8
TABLES=('weights','spot_covariates','feature_balance','shared_targets','group_aggregate_shares','region_weight_diagnostics','weighted_marginal_KS','case_eligibility')

def read_table(p):return pd.read_csv(p,sep='\t',float_precision='round_trip')
def mean_or_nan(x):return x.mean(0) if len(x) else np.full(x.shape[1],np.nan)
def path_key(path):
    path=Path(path).resolve()
    try:return str(path.relative_to(ROOT))
    except ValueError:return str(path)

def protect_output(out):
    assert out.resolve()!=OLD.resolve() and out.resolve()!=MODEL.resolve(), 'Output must differ from inputs'
    if (out/'RUN_MANIFEST.json').exists() or (out/'WEIGHTS_LOCK.json').exists():raise ValueError('Completed output exists; choose a new output directory')

def load_contracts():
    amendment_path=DOC/'rctd_calibration.json';amend=json.loads(amendment_path.read_text())
    model_path=DOC/'rctd_model.json';model=json.loads(model_path.read_text())
    taxonomy_path=DOC/'reference_groups.json';tax=json.loads(taxonomy_path.read_text())
    groups=list(tax['common_reference_groups']);assert len(groups)==7 and list(amend['tiers'])==list(TIERS)
    gate=amend['numeric_gates'];assert gate==dict(region_weight_sum=1,weight_cap=.05,block_mass_cap=1/3,min_spots=20,min_blocks=3,min_spot_ESS=20,min_block_ESS=3,max_feature_mean_difference=1e-6,solver_seconds=120,max_marginal_KS_strict_flag=.2)
    paths=[amendment_path,model_path,taxonomy_path,ROOT/'scripts/calibration_math.py',ROOT/'scripts/regional_profiles.py']
    return amend,model,groups,{path_key(p):sha(p) for p in paths}

def source_pairs():
    manifest_path=ROOT/'config/source_pipeline/source_counts_manifest.json';sections=json.loads(manifest_path.read_text())['sections'];byid={s['sample_id']:s for s in sections}
    ep=OLD/'regional_eligibility.tsv'
    e=read_table(ep);e=e[e.geometry.eq('exacthex')&e.compartment.eq('stroma')&e.evaluable]
    pairs=[]
    for (resource,patient),g in e.groupby(['resource','patient_id'],sort=True):
        for a,b in itertools.combinations(sorted(g.sample_id.unique()),2):
            pid='|'.join([resource,patient,a,b]);pairs.append(dict(resource=resource,patient_id=patient,section_a=a,section_b=b,pair_id=pid))
    assert pairs and not pd.DataFrame(pairs)[['resource','patient_id']].duplicated().any(),'Current estimand expects one eligible section pair per source patient'
    return sections,byid,pairs,{path_key(manifest_path):sha(manifest_path),path_key(ep):sha(ep)}

def verify_full_batch(sections,model_protocol):
    records=[]
    for section in sections:
        sid=section['sample_id'];p=MODEL/'all'/sid/'RUN_STATUS.json';r=json.loads(p.read_text())
        assert r['mode']=='all' and r['status']=='fit_completed' and r['protocol_sha256']==model_protocol and r['section']==sid
        for f in r['outputs']:
            path=p.parent/f['file'];assert path.exists() and sha(path)==f['sha256']
        records.append({'sample_id':sid,'resource':section['resource'],'patient_id':section['patient_id'],'RUN_STATUS_path':path_key(p),'RUN_STATUS_sha256':sha(p),'runner_sha256':r['runner_sha256'],'requested_spots':r['requested_spots'],'retained_spots':r['retained_spots']})
    return records,{}

def load_section_covariates(section,groups,model_sha):
    sid=section['sample_id'];directory=MODEL/'inputs/all'/sid;mp=directory/'INPUT_MANIFEST.json';m=json.loads(mp.read_text())
    assert m['mode']=='all' and m['sample_id']==sid and m['protocol_sha256']==model_sha and m['selected_spots']==m['registered_spots']
    meta_path=directory/'spots.tsv.gz';assert sha(meta_path)==m['files']['spots.tsv.gz']['sha256']
    meta=read_table(meta_path).sort_values('original_row_index').reset_index(drop=True)
    needed=['barcode','original_row_index','array_row','array_col','coarse_label','raw_label','blocks','nUMI_full_feature','detected_genes_full_feature','mitochondrial_UMIs_full_feature']
    assert set(needed)<=set(meta) and len(meta)==m['registered_spots'] and np.array_equal(meta.original_row_index,np.arange(len(meta))) and not meta.barcode.duplicated().any()
    source_path=ROOT/section['source_root']/'spot_inputs.tsv.gz';source=read_table(source_path)
    assert np.array_equal(meta.barcode.to_numpy(str),source.barcode.to_numpy(str))
    for k in ['array_row','array_col','coarse_label']:
        assert np.array_equal(meta[k].fillna('<NA>').to_numpy(),source[k].fillna('<NA>').to_numpy())
    geo,blocks,gpath=geometry_for(section,source);assert np.array_equal(blocks,meta.blocks.to_numpy(str))
    for band,key in [('near','near_stroma'),('far','far_stroma')]:assert np.array_equal(meta[key].to_numpy(bool),geo['exacthex:stroma'][0 if band=='near' else 1])
    p=MODEL/'all'/sid;status=json.loads((p/'RUN_STATUS.json').read_text());assert status['input_manifest_sha256']==sha(mp)
    wpath=p/'INFERRED_NORMALIZED_MIXTURE_WEIGHTS.tsv.gz';npath=p/'NATIVE_UNCONSTRAINED_WEIGHTS.tsv.gz';retpath=p/'REQUESTED_RETAINED_BARCODES.tsv.gz'
    w=read_table(wpath).set_index('barcode');native=read_table(npath).set_index('barcode');ret=read_table(retpath).set_index('barcode')
    assert list(w.columns)==groups and set(w.index)==set(ret.index[ret.retained_native_bulk_filter]) and set(w.index)<=set(meta.barcode)
    assert np.isfinite(w.to_numpy()).all() and (w.to_numpy()>=0).all() and np.allclose(w.sum(1),1,rtol=0,atol=1e-8)
    assert np.allclose(native.div(native.sum(1),axis=0).loc[w.index,groups].to_numpy(),w.to_numpy(),rtol=0,atol=1e-12)
    aligned=w.reindex(meta.barcode).to_numpy(float);matched=np.isfinite(aligned).all(1)
    total=meta.nUMI_full_feature.to_numpy(float);detected=meta.detected_genes_full_feature.to_numpy(float);mito=meta.mitochondrial_UMIs_full_feature.to_numpy(float)
    assert np.isfinite(np.column_stack([total,detected,mito])).all() and (total>0).all() and (detected>=0).all() and (mito>=0).all() and (mito<=total).all()
    qc=np.column_stack([np.log1p(total),np.log1p(detected),mito/total]);nativesum=native.sum(1).reindex(meta.barcode).to_numpy(float)
    paths=[mp,meta_path,source_path,gpath,wpath,npath,retpath,p/'RUN_STATUS.json']
    identity={path_key(x):sha(x) for x in paths}
    coverage={'resource':section['resource'],'patient_id':section['patient_id'],'sample_id':sid,'registered_spots':len(meta),'model_available':int(matched.sum()),'model_missing_native_filter':int((~matched).sum()),'QC_source':'full original feature totals/detected unique geneIDs/mitochondrial UMI counts from immutable full-input metadata; no shared-gene recomputation','staged_count_matrix_read':False}
    return dict(meta=meta,shares=aligned,matched=matched,qc=qc,native_sum=nativesum,geo=geo,blocks=blocks),identity,coverage

def feature_arrays(regions,tier,groups):
    shares=[r['shares'] for r in regions]
    rms=np.maximum(np.sqrt(np.mean([mean_or_nan(a*a) for a in shares],0)),1e-6)
    features=[a/rms for a in shares];names=['group_mean:'+g for g in groups]
    if tier!='composition_means':
        names+=['qc:log1p_total_umis','qc:log1p_detected_genes','qc:mitochondrial_fraction'];features=[np.column_stack([x,r['qc']]) for x,r in zip(features,regions)]
    if tier=='composition_second_moments_and_qc':
        names+=['group_second_moment:'+g for g in groups];features=[np.column_stack([x,(a/rms)**2]) for x,a in zip(features,shares)]
    return features,names,rms

def no_solution(features,message):
    p=features[0].shape[1]
    return dict(weights=[np.full(len(x),np.nan) for x in features],target=np.full(p,np.nan),prior=np.mean([mean_or_nan(x) for x in features],0),status=-2,success=False,message=message,objective=np.nan,objective_check=np.nan,solver_seconds=0,n_variables=0,n_equalities=0,n_inequalities=0,iterations=0,warnings=[],eq_violation=np.nan,ineq_violation=np.nan)

def calibrate_case(regions,base,groups):
    features,names,rms=feature_arrays(regions,base['tier'],groups);pre=[]
    for r in regions:
        tag=r['sample_id']+':'+r['band']
        if len(r['index'])<20:pre.append(tag+':model_spots_below20')
        if len(np.unique(r['block']))<3:pre.append(tag+':blocks_below3')
    if any(not np.isfinite(x).all() for x in features):pre.append('nonfinite_feature_or_missing_four_region_RMS')
    solution=no_solution(features,';'.join(pre)) if pre else program(features,[r['block'] for r in regions])
    finite=all(len(v)>0 and np.isfinite(v).all() for v in solution['weights'])
    achieved=np.vstack([v@x for v,x in zip(solution['weights'],features)]) if finite else np.full((4,len(names)),np.nan)
    spread=float(np.ptp(achieved,axis=0).max()) if finite else np.nan;reasons=list(pre)
    if not solution['success']:reasons.append('LP_not_optimal:'+str(solution['status']))
    if not finite:reasons.append('no_finite_weight_solution')
    if finite and spread>1e-6:reasons.append('scaled_feature_mean_range_exceeds1e-6')
    result={k:[] for k in TABLES};region_records=[]
    for ri,(r,v,x) in enumerate(zip(regions,solution['weights'],features)):
        blocks=np.unique(r['block']);mass=np.array([v[r['block']==b].sum() for b in blocks]) if finite else np.full(len(blocks),np.nan)
        ess=effective_n(v);bess=effective_n(mass);vsum=float(v.sum()) if finite else np.nan;minv=float(v.min()) if finite else np.nan;maxv=float(v.max()) if finite else np.nan;maxmass=float(mass.max()) if finite else np.nan
        if finite:
            tag=r['sample_id']+':'+r['band']
            for bad,label in [(abs(vsum-1)>TOL,'weight_sum'),(minv<0 or maxv>.05+TOL,'weight_bounds'),(maxmass>1/3+TOL,'block_mass_cap'),(ess<20-TOL,'spot_ESS_below20'),(bess<3-TOL,'block_ESS_below3')]:
                if bad:reasons.append(tag+':'+label)
        region_records.append({**base,'sample_id':r['sample_id'],'band':r['band'],'n_original_region_spots':r['n_original'],'n_model_available_spots':len(v),'n_missing_model_spots':r['n_original']-len(v),'n_available_blocks':len(blocks),'weight_sum':vsum,'min_weight':minv,'max_weight':maxv,'spot_ESS':ess,'block_ESS':bess,'max_block_mass':maxmass,'n_supported_spots':int((v>1e-12).sum()) if finite else 0,'n_supported_blocks':int((mass>1e-12).sum()) if finite else 0,'unweighted_native_weight_sum_mean':float(np.mean(r['native_sum'])) if len(r['native_sum']) else np.nan,'weighted_native_weight_sum_mean':float(v@r['native_sum']) if finite else np.nan,'native_sum_interpretation':'unconstrained RNA-model diagnostic, not calibrated physical cell burden'})
        for i,idx in enumerate(r['index']):result['weights'].append({**base,'sample_id':r['sample_id'],'band':r['band'],'spot_index':int(idx),'barcode':r['barcodes'][i],'block':r['block'][i],'weight':v[i]})
        for j,name in enumerate(names):result['feature_balance'].append({**base,'sample_id':r['sample_id'],'band':r['band'],'feature':name,'unweighted_mean':float(mean_or_nan(x)[j]),'weighted_mean':float(achieved[ri,j]),'shared_target':float(solution['target'][j])})
        weighted=v@r['shares'] if finite else np.full(len(groups),np.nan);den=float(weighted.sum())
        for j,g in enumerate(groups):result['group_aggregate_shares'].append({**base,'sample_id':r['sample_id'],'band':r['band'],'group':g,'equal_region_RMS':float(rms[j]),'unweighted_mean_share':float(mean_or_nan(r['shares'])[j]),'weighted_mean_share':float(weighted[j]),'weighted_aggregate_share':float(weighted[j]/den) if den>0 else np.nan})
    for j,name in enumerate(names):result['shared_targets'].append({**base,'feature':name,'equal_region_prior_mean':float(solution['prior'][j]),'shared_target':float(solution['target'][j]),'max_region_mean_range':float(np.ptp(achieved[:,j])) if finite else np.nan})
    ks=[]
    for i,j in itertools.combinations(range(4),2):
        for k,g in enumerate(groups):
            statistic=weighted_ks(regions[i]['shares'][:,k],solution['weights'][i],regions[j]['shares'][:,k],solution['weights'][j]);unweighted=weighted_ks(regions[i]['shares'][:,k],np.ones(len(regions[i]['index'])),regions[j]['shares'][:,k],np.ones(len(regions[j]['index'])));ks.append(statistic)
            result['weighted_marginal_KS'].append({**base,'sample_id_a':regions[i]['sample_id'],'band_a':regions[i]['band'],'sample_id_b':regions[j]['sample_id'],'band_b':regions[j]['band'],'group':g,'weighted_KS':statistic,'unweighted_KS':unweighted})
    eligible=not reasons;maxks=float(max(ks)) if all(np.isfinite(ks)) else np.nan;strict=bool(eligible and np.isfinite(maxks) and maxks<=.2)
    for r in region_records:r.update(eligible=eligible,distribution_flag=strict)
    result['region_weight_diagnostics']=region_records
    result['case_eligibility']=[{**base,'n_features':len(names),'n_weight_variables':sum(len(x) for x in features),'solver_status':solution['status'],'solver_success':solution['success'],'solver_message':solution['message'],'solver_seconds':solution['solver_seconds'],'solver_iterations':solution['iterations'],'objective':solution['objective'],'objective_recomputed':solution['objective_check'],'maximum_equality_residual':solution['eq_violation'],'maximum_inequality_violation':solution['ineq_violation'],'max_scaled_feature_mean_range':spread,'eligible':eligible,'failure_reason':';'.join(dict.fromkeys(reasons)),'max_weighted_fraction_KS':maxks,'stricter_distribution_flag':strict}]
    record={**base,'status':solution['status'],'success':solution['success'],'message':solution['message'],'seconds':solution['solver_seconds'],'warnings':solution['warnings'],'n_variables':solution['n_variables'],'n_equalities':solution['n_equalities'],'n_inequalities':solution['n_inequalities']}
    return result,record

def toy_check(path):
    rng=np.random.default_rng(20261026);groups=['g'+str(i) for i in range(7)];shares=rng.dirichlet(np.ones(7),24);qc=np.column_stack([np.log1p(np.arange(24)+2000),np.log1p(np.arange(24)+500),np.full(24,.02)])
    regions=[dict(sample_id='toy'+str(i//2),band=['near','far'][i%2],index=np.arange(24),barcodes=np.array(['b'+str(j) for j in range(24)]),block=np.repeat(['a','b','c','d'],6),shares=shares.copy(),qc=qc.copy(),native_sum=np.ones(24),n_original=24) for i in range(4)]
    checks=[]
    for tier in TIERS:
        rows,rec=calibrate_case(regions,dict(resource='TOY',patient_id='TOY',section_a='toy0',section_b='toy1',pair_id='TOY',case_id='TOY|'+tier,tier=tier),groups);c=rows['case_eligibility'][0]
        assert c['eligible'] and c['stricter_distribution_flag'] and abs(c['objective'])<1e-7 and c['max_scaled_feature_mean_range']<1e-8
        for d in rows['region_weight_diagnostics']:assert abs(d['weight_sum']-1)<1e-8 and d['spot_ESS']>=20-1e-8 and d['block_ESS']>=3-1e-8
        checks.append(dict(tier=tier,eligible=True,strict=True,objective=c['objective']))
    bad=[dict(r) for r in regions];bad[0]['shares']=np.tile(np.eye(7)[0],(24,1));bad[1]['shares']=np.tile(np.eye(7)[1],(24,1))
    r,_=calibrate_case(bad,dict(resource='TOY',patient_id='TOY',section_a='toy0',section_b='toy1',pair_id='bad',case_id='bad',tier=TIERS[0]),groups)
    assert not r['case_eligibility'][0]['eligible'] and all(np.isnan(x['weight']) for x in r['weights'])
    path.parent.mkdir(parents=True,exist_ok=True);jsave(path,{'checked_utc':now(),'status':'toy_constraints_and_infeasible_NA_passed','RNA_or_native_model_files_read':False,'model_fit_run':False,'actual_support_calibration_run':False,'checks':checks,'infeasible_weights_missing':True,'script_sha256':sha(__file__)})

def main():
    global MODEL,OLD
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,default=DEFAULT_OUT);ap.add_argument('--model',type=Path,required=False,default=MODEL);ap.add_argument('--profiles',type=Path,required=False,default=OLD);ap.add_argument('--toy',action='store_true');ap.add_argument('--toy-output',type=Path,default=ROOT/'results/calibration_toy.json');args=ap.parse_args()
    MODEL=args.model.resolve();OLD=args.profiles.resolve()
    if args.toy:toy_check(args.toy_output);return
    protect_output(args.output);amend,model,groups,identities=load_contracts();sections,byid,pairs,sourcehash=source_pairs();identities.update(sourcehash)
    modelrows,batchhash=verify_full_batch(sections,sha(DOC/'rctd_model.json'));identities.update(batchhash)
    out=args.output;out.mkdir(parents=True,exist_ok=True);run={'started_utc':now(),'status':'running_support_only_no_weighted_RNA','analysis_class':'outcome_informed_same_RNA_model_conditional_sensitivity','script_sha256':sha(__file__),'groups':groups,'tiers':TIERS,'input_hashes':identities,'full_native_model_records':modelrows,'attempted_pairs':pairs,'solver_records':[],'source_sections':[],'software':{'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__}}
    jsave(out/'STARTED.json',run);tables={k:[] for k in TABLES};start=time.monotonic()
    with threadpool_limits(limits=2):
        for pair in pairs:
            regions=[]
            for sid in [pair['section_a'],pair['section_b']]:
                source=byid[sid];data,ids,coverage=load_section_covariates(source,groups,sha(DOC/'rctd_model.json'));identities.update(ids);run['source_sections'].append(coverage)
                for band,mask in zip(['near','far'],data['geo']['exacthex:stroma']):
                    ix=np.flatnonzero(mask & data['matched']);m=data['meta'];r=dict(sample_id=sid,band=band,index=ix,barcodes=m.barcode.to_numpy(str)[ix],block=data['blocks'][ix],shares=data['shares'][ix],qc=data['qc'][ix],native_sum=data['native_sum'][ix],n_original=int(mask.sum()));regions.append(r)
                    for j,k in enumerate(ix):tables['spot_covariates'].append({**pair,'sample_id':sid,'band':band,'spot_index':int(k),'barcode':r['barcodes'][j],'block':r['block'][j],**{g:float(r['shares'][j,n]) for n,g in enumerate(groups)},**{name:float(r['qc'][j,n]) for n,name in enumerate(QC)},'native_unconstrained_weight_sum':float(r['native_sum'][j])})
            for tier in TIERS:
                base={**pair,'tier':tier,'case_id':pair['pair_id']+'|'+tier};local,record=calibrate_case(regions,base,groups)
                for name in TABLES:
                    if name!='spot_covariates':tables[name].extend(local[name])
                run['solver_records'].append(record);jsave(out/'PROGRESS.json',{'completed_case_attempts':len(tables['case_eligibility']),'expected_case_attempts':len(pairs)*len(TIERS),'last_case':base['case_id']});print('RCTD_SUPPORT',base['case_id'],local['case_eligibility'][0]['eligible'],flush=True)
    filenames={k:k+('.tsv.gz' if k in ['weights','spot_covariates'] else '.tsv') for k in TABLES}
    empty_headers={'weights':['resource','patient_id','section_a','section_b','pair_id','case_id','tier','sample_id','band','spot_index','barcode','block','weight'],
                   'spot_covariates':['resource','patient_id','section_a','section_b','pair_id','sample_id','band','spot_index','barcode','block',*groups,*QC,'native_unconstrained_weight_sum']}
    for key,filename in filenames.items():
        frame=pd.DataFrame(tables[key]) if tables[key] else pd.DataFrame(columns=empty_headers.get(key,[]))
        frame.to_csv(out/filename,sep='\t',index=False,na_rep='NA')
    pd.DataFrame(pairs).to_csv(out/'SOURCE_PAIR_MANIFEST.tsv',sep='\t',index=False);pd.DataFrame(run['source_sections']).to_csv(out/'MODEL_AVAILABILITY_BY_SECTION.tsv',sep='\t',index=False)
    cases=pd.DataFrame(tables['case_eligibility']);assert len(cases)==len(pairs)*len(TIERS)
    tablehashes={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name not in ['STARTED.json','PROGRESS.json','RUN_MANIFEST.json']}
    lock={'locked_utc':now(),'status':'all_common_reference_support_weights_and_failures_fixed_before_weighted_RNA','script_sha256':sha(__file__),'input_hashes':identities,'tables':tablehashes,'n_attempted_pairs':len(pairs),'n_attempted_cases':len(cases),'n_eligible_cases':int(cases.eligible.sum()),'n_strict_distribution_cases':int(cases.stricter_distribution_flag.sum()),'new_weighted_RNA_calculated':False};jsave(out/'WEIGHTS_LOCK.json',lock)
    run.update(input_hashes=identities,completed_utc=now(),elapsed_seconds=time.monotonic()-start,status='completed_support_only_no_weighted_RNA',n_attempted_pairs=len(pairs),n_attempted_cases=len(cases),n_eligible_cases=int(cases.eligible.sum()),n_strict_distribution_cases=int(cases.stricter_distribution_flag.sum()),no_RNA_count_matrix_read=True,no_new_weighted_RNA_outcomes=True,interpretation=amend['interpretation'],outputs={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name not in ['STARTED.json','PROGRESS.json','RUN_MANIFEST.json']});jsave(out/'RUN_MANIFEST.json',run)

if __name__=='__main__':main()
