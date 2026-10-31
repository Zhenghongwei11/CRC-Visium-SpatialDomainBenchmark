#!/usr/bin/env python3
"""Frozen continuous common-support calibration; no weighted RNA outcomes."""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_key]='2'
import argparse
from datetime import datetime,timezone
import hashlib
import itertools
import json
from pathlib import Path
import re
import sys
import time
import warnings

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, hstack, vstack, eye, block_diag
from threadpoolctl import threadpool_limits
from regional_profiles import counts_for,geometry_for

ROOT=Path(__file__).resolve().parents[1]
DOC=ROOT/'config'
DEFAULT_OUT=ROOT/'results/author_calibration'
MODEL=ROOT/'data/reference_inputs/cell2location/W_cell_density.csv'
KEYS=ROOT/'data/reference_inputs/cell2location/barcode_map.tsv'
TIERS=('composition_means','composition_and_qc_means','composition_second_moments_and_qc')
TOL=1e-8


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def now():return datetime.now(timezone.utc).isoformat()


def jsave(path,record):Path(path).write_text(json.dumps(record,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def effective_n(weight):
    return float(weight.sum()**2/np.dot(weight,weight)) if np.isfinite(weight).all() and np.dot(weight,weight)>0 else np.nan


def weighted_ks(a,wa,b,wb):
    if not(np.isfinite(a).all() and np.isfinite(b).all() and np.isfinite(wa).all() and np.isfinite(wb).all()) or min(wa.sum(),wb.sum())<=0:return np.nan
    ia=np.argsort(a,kind='stable');ib=np.argsort(b,kind='stable');a=a[ia];b=b[ib];wa=wa[ia]/wa.sum();wb=wb[ib]/wb.sum();grid=np.unique(np.r_[a,b]);ca=np.r_[0,np.cumsum(wa)];cb=np.r_[0,np.cumsum(wb)];return float(np.max(abs(ca[np.searchsorted(a,grid,side='right')]-cb[np.searchsorted(b,grid,side='right')])))


def program(features,blocks):
    counts=[len(x) for x in features];N=sum(counts);P=features[0].shape[1];offset=np.r_[0,np.cumsum(counts)]
    uniform=np.concatenate([np.full(n,1/n) for n in counts]);prior=np.mean([x.mean(axis=0) for x in features],axis=0)
    objective=np.r_[np.zeros(N+P),np.full(N,.01),np.ones(P)]
    sum_rows=csr_matrix((np.ones(N),(np.repeat(np.arange(4),counts),np.arange(N))),shape=(4,N))
    feature_eq=block_diag([csr_matrix(x.T) for x in features],format='csr');target_eq=vstack([-eye(P,format='csr')]*4,format='csr')
    Aeq=vstack([hstack([sum_rows,csr_matrix((4,P+N+P))]),hstack([feature_eq,target_eq,csr_matrix((4*P,N+P))])],format='csr');beq=np.r_[np.ones(4),np.zeros(4*P)]
    I=eye(N,format='csr');J=eye(P,format='csr');zNP=csr_matrix((N,P));zPN=csr_matrix((P,N))
    Aub=[hstack([I,zNP,-I,zNP]),hstack([-I,zNP,-I,zNP]),hstack([zPN,J,zPN,-J]),hstack([zPN,-J,zPN,-J])];bub=[uniform,-uniform,prior,-prior]
    rr=[];cc=[];nr=0
    for r,block in enumerate(blocks):
        for label in sorted(np.unique(block)):
            local=np.flatnonzero(block==label)+offset[r];rr.extend([nr]*len(local));cc.extend(local.tolist());nr+=1
    mass=csr_matrix((np.ones(len(rr)),(rr,cc)),shape=(nr,N));Aub.append(hstack([mass,csr_matrix((nr,P+N+P))]));bub.append(np.full(nr,1/3))
    Aub=vstack(Aub,format='csr');bub=np.concatenate(bub)
    assert Aeq.shape[1]==Aub.shape[1]==len(objective)==2*N+2*P
    bounds=[(0,.05)]*N+[(None,None)]*P+[(0,None)]*(N+P)
    t0=time.monotonic();error=None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:result=linprog(objective,A_ub=Aub,b_ub=bub,A_eq=Aeq,b_eq=beq,bounds=bounds,method='highs',options={'threads':2,'time_limit':120,'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
        except Exception as exc:result=None;error=type(exc).__name__+': '+str(exc)
    warnings_list=[str(x.message) for x in caught]
    if result is None or result.x is None:
        return dict(weights=[np.full(n,np.nan) for n in counts],target=np.full(P,np.nan),prior=prior,status=int(result.status) if result is not None else -1,success=False,message=str(result.message) if result is not None else error,objective=np.nan,objective_check=np.nan,solver_seconds=time.monotonic()-t0,n_variables=len(objective),n_equalities=len(beq),n_inequalities=len(bub),iterations=int(result.nit) if result is not None and result.nit is not None else 0,warnings=warnings_list,eq_violation=np.nan,ineq_violation=np.nan)
    vector=result.x;w=vector[:N];target=vector[N:N+P];recomputed=float(np.sum(abs(target-prior))+.01*np.sum(abs(w-uniform)))
    assert np.isclose(result.fun,recomputed,atol=1e-6,rtol=1e-7),'L1 objective mismatch'
    return dict(weights=[w[offset[i]:offset[i+1]] for i in range(4)],target=target,prior=prior,status=int(result.status),success=bool(result.success),message=str(result.message),objective=float(result.fun),objective_check=recomputed,solver_seconds=time.monotonic()-t0,n_variables=len(objective),n_equalities=len(beq),n_inequalities=len(bub),iterations=int(result.nit),warnings=warnings_list,eq_violation=float(np.max(abs(Aeq@vector-beq))),ineq_violation=float(np.max(np.maximum(Aub@vector-bub,0))))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=DEFAULT_OUT);parser.add_argument('--counts',type=Path,required=True);parser.add_argument('--profiles',type=Path,required=True);args=parser.parse_args();out=args.output;os.environ['CRC_COUNTS_ROOT']=str(args.counts.resolve())
    if(out/'RUN_MANIFEST.json').exists():raise ValueError('Completed calibration exists; preserve it and use a versioned output for rerun')
    out.mkdir(parents=True,exist_ok=True)
    stage_path=DOC/'regional_profiles.json';stage=json.loads(stage_path.read_text());stage_lock=json.loads((DOC/'PROTOCOL_LOCK_v1.json').read_text());assert sha(stage_path)==stage_lock['sha256']
    taxonomy_path=DOC/'reference_groups.json';taxonomy=json.loads(taxonomy_path.read_text())
    groups=list(taxonomy['author_model_summary_groups']);labels=[x for xs in taxonomy['author_model_summary_groups'].values() for x in xs];assert len(groups)==8 and len(labels)==len(set(labels))==36
    source=pd.read_csv(MODEL).set_index('spot_id');source.columns=[re.sub(r'^mean_spot_factors','',c) for c in source.columns];assert set(source.columns)==set(labels) and not source.index.duplicated().any();assert np.isfinite(source.to_numpy()).all() and (source.to_numpy()>=0).all()
    grouped=pd.DataFrame({name:source[types].sum(axis=1) for name,types in taxonomy['author_model_summary_groups'].items()},index=source.index);assert np.allclose(grouped.sum(axis=1),source.sum(axis=1),atol=1e-12,rtol=1e-12)
    keymap=pd.read_csv(KEYS,sep='\t');assert not keymap[['sample_id','barcode']].duplicated().any()
    manifest_path=ROOT/stage['source_manifest'];sections=json.loads(manifest_path.read_text())['sections'];by_id={s['sample_id']:s for s in sections}
    eligible_path=args.profiles/'regional_eligibility.tsv';eligibility=pd.read_csv(eligible_path,sep='\t');eligible=eligibility[(eligibility.resource=='Valdeolivas')&(eligibility.geometry=='exacthex')&(eligibility.compartment=='stroma')&eligibility.evaluable]
    run={'started_utc':now(),'status':'running_weights_only_no_new_RNA_outcomes','analysis_class':'outcome_informed_author_model_conditional_sensitivity','stage2_sha256':sha(stage_path),'taxonomy_sha256':sha(taxonomy_path),'source_mean_sha256':sha(MODEL),'source_keymap_sha256':sha(KEYS),'source_manifest_sha256':sha(manifest_path),'source_eligibility_sha256':sha(eligible_path),'script_sha256':sha(__file__),'helper_sha256':sha(ROOT/'scripts/regional_profiles.py'),'numeric_gate_tolerance':TOL,'QC_scale':'raw log1p(totalUMIs), raw log1p(detected unique gene IDs), raw mitochondrial fraction; group abundance alone uses equal-region RMS','groups':groups,'tiers':TIERS,'software':{'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__},'source_sections':[],'solver_records':[]}
    jsave(out/'STARTED.json',run)
    weight_rows=[];feature_rows=[];target_rows=[];share_rows=[];region_rows=[];ks_rows=[];case_rows=[];covariate_rows=[];t0=time.monotonic()
    with threadpool_limits(limits=2):
        for patient,pg in eligible.groupby('patient_id',sort=True):
            for a,b in itertools.combinations(sorted(pg.sample_id),2):
                regions=[]
                for sid in(a,b):
                    s=by_id[sid];counts,spots,ids,symbols,qc,members,checks=counts_for(s);del counts
                    geo,blocks,geometry_path=geometry_for(s,spots);local=keymap[keymap.sample_id==sid].set_index('barcode').reindex(spots.barcode);model=grouped.reindex(local.spot_id.to_numpy()).to_numpy(float);matched=np.isfinite(model).all(1)
                    full_qc=np.column_stack([np.log1p(qc['total']),np.log1p(qc['detected']),qc['mito']/qc['total']])
                    run['source_sections'].append({'sample_id':sid,'patient_id':patient,'source_files':members,'loader_checks':checks,'spot_inputs_sha256':sha(ROOT/s['source_root']/'spot_inputs.tsv.gz'),'geometry_sha256':sha(geometry_path),'registered_spots':len(spots),'model_available':int(matched.sum()),'model_missing':int((~matched).sum())})
                    for band,mask in zip(['near','far'],geo['exacthex:stroma']):
                        index=np.flatnonzero(mask&matched)
                        regions.append(dict(sample_id=sid,band=band,index=index,barcodes=spots.barcode.to_numpy(str)[index],block=blocks[index],abundance=model[index],qc=full_qc[index],n_original=int(mask.sum()),source_spot_root=s['source_root']))
                rms=np.sqrt(np.mean([np.mean(r['abundance']**2,axis=0) for r in regions],axis=0));rms=np.maximum(rms,1e-6)
                for r in regions:
                    r['scaled']=r['abundance']/rms
                    for i,(spotindex,bc,block) in enumerate(zip(r['index'],r['barcodes'],r['block'])):
                        covariate_rows.append({'resource':'Valdeolivas','patient_id':patient,'sample_id':r['sample_id'],'band':r['band'],'spot_index':int(spotindex),'barcode':bc,'block':block,**{g:float(r['abundance'][i,j]) for j,g in enumerate(groups)},'log1p_total_umis':r['qc'][i,0],'log1p_detected_genes':r['qc'][i,1],'mitochondrial_umi_fraction':r['qc'][i,2]})
                for tier in TIERS:
                    case_id=patient+'|'+tier;base={'resource':'Valdeolivas','patient_id':patient,'section_a':a,'section_b':b,'case_id':case_id,'tier':tier};names=['group_mean:'+g for g in groups];features=[r['scaled'] for r in regions]
                    if tier!='composition_means':
                        names+=['qc:log1p_total_umis','qc:log1p_detected_genes','qc:mitochondrial_fraction'];features=[np.column_stack([x,r['qc']]) for x,r in zip(features,regions)]
                    if tier=='composition_second_moments_and_qc':
                        names+=['group_second_moment:'+g for g in groups];features=[np.column_stack([x,r['scaled']**2]) for x,r in zip(features,regions)]
                    pre_fail=[]
                    for r in regions:
                        if len(r['index'])<20:pre_fail.append(r['sample_id']+':'+r['band']+':model_spots_below20')
                        if len(np.unique(r['block']))<3:pre_fail.append(r['sample_id']+':'+r['band']+':blocks_below3')
                    if pre_fail:
                        solution=dict(weights=[np.full(len(r['index']),np.nan) for r in regions],target=np.full(len(names),np.nan),prior=np.mean([x.mean(0) for x in features],0),status=-2,success=False,message=';'.join(pre_fail),objective=np.nan,objective_check=np.nan,solver_seconds=0,n_variables=0,n_equalities=0,n_inequalities=0,iterations=0,warnings=[],eq_violation=np.nan,ineq_violation=np.nan)
                    else:
                        print('LP_START',patient,tier,'N',sum(len(x) for x in features),'P',len(names),flush=True);solution=program(features,[r['block'] for r in regions])
                    valid_weights=all(np.isfinite(w).all() for w in solution['weights']);achieved=np.vstack([w@x for w,x in zip(solution['weights'],features)]) if valid_weights else np.full((4,len(names)),np.nan)
                    mean_range=float(np.max(np.ptp(achieved,axis=0))) if valid_weights else np.nan
                    gate_reasons=list(pre_fail)
                    if not solution['success']:gate_reasons.append('LP_not_optimal:'+str(solution['status']))
                    if not valid_weights:gate_reasons.append('no_finite_weight_solution')
                    if valid_weights and mean_range>1e-6:gate_reasons.append('scaled_feature_mean_range_exceeds1e-6')
                    region_diagnostics=[]
                    for r,w,x in zip(regions,solution['weights'],features):
                        finite=np.isfinite(w).all();unique_blocks=np.unique(r['block']);mass=np.array([w[r['block']==bb].sum() for bb in unique_blocks]) if finite else np.full(len(unique_blocks),np.nan)
                        ess=effective_n(w);bess=effective_n(mass);maxw=float(w.max()) if len(w) and finite else np.nan;minw=float(w.min()) if len(w) and finite else np.nan;total=float(w.sum()) if finite else np.nan;maxmass=float(mass.max()) if len(mass) and finite else np.nan
                        if finite:
                            if abs(total-1)>TOL:gate_reasons.append(r['sample_id']+':'+r['band']+':weight_sum')
                            if minw<-TOL or maxw>.05+TOL:gate_reasons.append(r['sample_id']+':'+r['band']+':weight_bounds')
                            if maxmass>1/3+TOL:gate_reasons.append(r['sample_id']+':'+r['band']+':block_mass_cap')
                            if ess<20-TOL:gate_reasons.append(r['sample_id']+':'+r['band']+':spot_ESS_below20')
                            if bess<3-TOL:gate_reasons.append(r['sample_id']+':'+r['band']+':block_ESS_below3')
                        dr={**base,'sample_id':r['sample_id'],'band':r['band'],'n_original_region_spots':r['n_original'],'n_model_available_spots':len(w),'n_missing_model_spots':r['n_original']-len(w),'n_available_blocks':len(unique_blocks),'weight_sum':total,'min_weight':minw,'max_weight':maxw,'spot_ESS':ess,'block_ESS':bess,'max_block_mass':maxmass,'n_supported_spots':int((w>1e-12).sum()) if finite else 0,'n_supported_blocks':int((mass>1e-12).sum()) if finite else 0,'n_zero_total_model_abundance':int((r['abundance'].sum(1)<=0).sum())};region_diagnostics.append(dr)
                        for i,(spotindex,bc,bb) in enumerate(zip(r['index'],r['barcodes'],r['block'])):weight_rows.append({**base,'sample_id':r['sample_id'],'band':r['band'],'spot_index':int(spotindex),'barcode':bc,'block':bb,'source_spot_root':r['source_spot_root'],'weight':w[i]})
                        for j,name in enumerate(names):feature_rows.append({**base,'sample_id':r['sample_id'],'band':r['band'],'feature':name,'unweighted_mean':float(x[:,j].mean()),'weighted_mean':float(achieved[len(region_diagnostics)-1,j]),'shared_target':float(solution['target'][j])})
                        sums=w@r['abundance'] if finite else np.full(8,np.nan);den=float(sums.sum())
                        for j,g in enumerate(groups):share_rows.append({**base,'sample_id':r['sample_id'],'band':r['band'],'group':g,'equal_region_RMS':float(rms[j]),'unweighted_mean_abundance':float(r['abundance'][:,j].mean()),'weighted_mean_abundance':float(sums[j]),'weighted_aggregate_share':float(sums[j]/den) if den>0 else np.nan})
                    for j,name in enumerate(names):target_rows.append({**base,'feature':name,'equal_region_prior_mean':float(solution['prior'][j]),'shared_target':float(solution['target'][j]),'max_region_mean_range':float(np.ptp(achieved[:,j])) if valid_weights else np.nan})
                    all_ks=[]
                    for i,j in itertools.combinations(range(4),2):
                        ai,aj=regions[i]['abundance'],regions[j]['abundance'];si,sj=ai.sum(1),aj.sum(1);fractions_i=ai/si[:,None] if(si>0).all() else np.full_like(ai,np.nan);fractions_j=aj/sj[:,None] if(sj>0).all() else np.full_like(aj,np.nan)
                        for k,g in enumerate(groups):
                            statistic=weighted_ks(fractions_i[:,k],solution['weights'][i],fractions_j[:,k],solution['weights'][j]);unweighted=weighted_ks(fractions_i[:,k],np.ones(len(ai)),fractions_j[:,k],np.ones(len(aj)));all_ks.append(statistic)
                            ks_rows.append({**base,'sample_id_a':regions[i]['sample_id'],'band_a':regions[i]['band'],'sample_id_b':regions[j]['sample_id'],'band_b':regions[j]['band'],'group':g,'weighted_KS':statistic,'unweighted_KS':unweighted})
                    eligible_case=not gate_reasons;maxks=float(max(all_ks)) if all(np.isfinite(all_ks)) else np.nan;distribution_flag=bool(eligible_case and np.isfinite(maxks) and maxks<=.20)
                    for dr in region_diagnostics:dr.update(eligible=eligible_case,distribution_flag=distribution_flag);region_rows.append(dr)
                    case_rows.append({**base,'n_features':len(names),'n_weight_variables':sum(len(x) for x in features),'solver_status':solution['status'],'solver_success':solution['success'],'solver_message':solution['message'],'solver_seconds':solution['solver_seconds'],'solver_iterations':solution['iterations'],'objective':solution['objective'],'objective_recomputed':solution['objective_check'],'maximum_equality_residual':solution['eq_violation'],'maximum_inequality_violation':solution['ineq_violation'],'max_scaled_feature_mean_range':mean_range,'eligible':eligible_case,'failure_reason':';'.join(dict.fromkeys(gate_reasons)),'max_weighted_fraction_KS':maxks,'stricter_distribution_flag':distribution_flag})
                    run['solver_records'].append({**base,'status':solution['status'],'message':solution['message'],'success':solution['success'],'seconds':solution['solver_seconds'],'warnings':solution['warnings'],'n_variables':solution['n_variables'],'n_equalities':solution['n_equalities'],'n_inequalities':solution['n_inequalities']})
                    print('LP_DONE',patient,tier,'status',solution['status'],'eligible',eligible_case,'maxKS',maxks,flush=True)
                    jsave(out/'PROGRESS.json',{'completed_case_attempts':len(case_rows),'expected_case_attempts':12,'last_case':case_id})
    def save(items,name):pd.DataFrame(items).to_csv(out/name,sep='\t',index=False,na_rep='NA')
    save(weight_rows,'weights.tsv.gz');save(covariate_rows,'spot_covariates.tsv.gz');save(feature_rows,'feature_balance.tsv');save(target_rows,'shared_targets.tsv');save(share_rows,'group_aggregate_shares.tsv');save(region_rows,'region_weight_diagnostics.tsv');save(ks_rows,'weighted_marginal_KS.tsv');save(case_rows,'case_eligibility.tsv')
    cases=pd.DataFrame(case_rows);assert len(cases)==12
    lock={'locked_utc':now(),'status':'all_weights_and_eligibility_fixed_before_any_new_weighted_RNA_metric','script_sha256':sha(__file__),'stage2_sha256':sha(stage_path),'taxonomy_sha256':sha(taxonomy_path),'weights_sha256':sha(out/'weights.tsv.gz'),'case_eligibility_sha256':sha(out/'case_eligibility.tsv'),'n_attempted_cases':len(cases),'n_eligible_cases':int(cases.eligible.sum()),'n_stricter_distribution_cases':int(cases.stricter_distribution_flag.sum()),'new_weighted_RNA_calculated':False};jsave(out/'WEIGHTS_LOCK.json',lock)
    run.update({'completed_utc':now(),'elapsed_seconds':time.monotonic()-t0,'status':'completed_support_calibration_only_no_weighted_RNA','n_attempted_patients':cases.patient_id.nunique(),'n_attempted_cases':len(cases),'n_eligible_cases':int(cases.eligible.sum()),'n_distribution_flag_cases':int(cases.stricter_distribution_flag.sum()),'source36_sum_equals8_groups':True,'no_new_weighted_RNA_outcomes':True,'inference_limit':'same-RNA modeled abundance calibration is not independent composition/identity validation; mean balance is not full distribution balance','outputs':{p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name not in('RUN_MANIFEST.json','STARTED.json','PROGRESS.json')}});jsave(out/'RUN_MANIFEST.json',run)
    print(cases[['patient_id','tier','solver_status','eligible','max_scaled_feature_mean_range','max_weighted_fraction_KS','stricter_distribution_flag']].to_string(index=False),flush=True)

if __name__=='__main__':main()
