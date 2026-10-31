#!/usr/bin/env python3
"""Finite post-outcome dominant-model-stratum sensitivity for Val paired stroma."""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='2'
import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import re
import sys
import time

import numpy as np
import pandas as pd
import scipy
from scipy.stats import spearmanr
from regional_profiles import counts_for, geometry_for, profile, measures

ROOT=Path(__file__).resolve().parents[1]
DOC=ROOT/'config'
DEFAULT_OUT=ROOT/'results/dominant_balance'
PROFILE_OUT=ROOT/'results/regional_profiles'
BASE_SOURCE=ROOT/'data/reference_inputs/cell2location/W_cell_density.csv'
MASTER=20261026
METRICS=('scaled_abs_difference','spearman')


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def fixed_seed(*parts):
    return int.from_bytes(hashlib.sha256('|'.join(map(str,(MASTER,*parts))).encode()).digest()[:8],'little')


def write_json(path,record):
    Path(path).write_text(json.dumps(record,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def shares(values,selected):
    total=values[selected].sum(axis=0)
    denominator=float(total.sum())
    return total/denominator if denominator>0 else np.full(values.shape[1],np.nan)


def delta_rows(vectors,mask,scale):
    logs=[profile(v)[mask] for v in vectors]
    near=measures(logs[0],logs[2],scale[mask]);far=measures(logs[1],logs[3],scale[mask])
    # Independent SciPy correlation checks the inherited rank-correlation implementation.
    for a,b,record in [(logs[0],logs[2],near),(logs[1],logs[3],far)]:
        assert np.isclose(spearmanr(a,b).statistic,record['spearman'],atol=1e-12,rtol=0)
    return {m:{'near_value':near[m],'far_value':far[m],'far_minus_near':far[m]-near[m]} for m in METRICS}


def main():
    global PROFILE_OUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUT)
    parser.add_argument('--counts',type=Path,default=ROOT/'original_inputs/counts')
    parser.add_argument('--profiles',type=Path,default=PROFILE_OUT)
    args=parser.parse_args();out=args.output
    PROFILE_OUT=args.profiles.resolve();os.environ['CRC_COUNTS_ROOT']=str(args.counts.resolve())
    if (out/'RUN_MANIFEST.json').exists():raise ValueError('Completed output exists; preserve it and choose a versioned output for rerun')
    out.mkdir(parents=True,exist_ok=True)
    addendum_path=DOC/'dominant_balance.json'
    addendum=json.loads(addendum_path.read_text());plan=json.loads((DOC/'COMPOSITION_PLAN_v1.json').read_text());lock=json.loads((DOC/'COMPOSITION_LOCK_v1.json').read_text())
    assert sha(DOC/'COMPOSITION_PLAN_v1.json')==lock['sha256']
    groups=list(plan['groups']);labels=[x for values in plan['groups'].values() for x in values]
    assert len(labels)==len(set(labels))==36 and len(groups)==7
    source=pd.read_csv(BASE_SOURCE).set_index('spot_id');source.columns=[re.sub(r'^mean_spot_factors','',str(x)) for x in source.columns]
    assert set(source.columns)==set(labels) and not source.index.duplicated().any()
    assert np.isfinite(source.to_numpy()).all() and (source.to_numpy()>=0).all()
    source_groups=pd.DataFrame({g:source[types].sum(axis=1) for g,types in plan['groups'].items()},index=source.index)
    assert np.allclose(source_groups.sum(axis=1),source.sum(axis=1),rtol=1e-13,atol=1e-13)
    source_manifest=ROOT/'config/source_pipeline/source_counts_manifest.json'
    sections=json.loads(source_manifest.read_text())['sections'];by_id={s['sample_id']:s for s in sections}
    keys_path=ROOT/'data/reference_inputs/cell2location/barcode_map.tsv';keys=pd.read_csv(keys_path,sep='\t')
    assert not keys[['sample_id','barcode']].duplicated().any()
    e=pd.read_csv(PROFILE_OUT/'regional_eligibility.tsv',sep='\t')
    e=e[(e.resource=='Valdeolivas')&(e.geometry=='exacthex')&(e.compartment=='stroma')&e.evaluable]
    panel_path=PROFILE_OUT/'gene_panels.tsv';panel=pd.read_csv(panel_path,sep='\t');panel=panel[panel.resource=='Valdeolivas']
    broad=panel.broad.to_numpy(bool);scale=panel.fixed_gene_scale.to_numpy(float)
    started=datetime.now(timezone.utc).isoformat();t0=time.monotonic()
    run={'started_utc':started,'status':'running_post_outcome_dominant_model_stratum_balance','addendum_sha256':sha(addendum_path),'composition_plan_sha256':sha(DOC/'COMPOSITION_PLAN_v1.json'),'script_sha256':sha(__file__),'profile_helper_sha256':sha(ROOT/'scripts/regional_profiles.py'),'source_mean_csv_sha256':sha(BASE_SOURCE),'keymap_sha256':sha(keys_path),'source_manifest_sha256':sha(source_manifest),'gene_panels_sha256':sha(panel_path),'earlier_results_known':True,'seed_master':MASTER,'groups_in_tie_order':groups,'attempted_pairs':[],'software':{'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__},'fixed_design':addendum}
    write_json(out/'STARTED.json',run)
    availability=[];strata_rows=[];continuous_before=[];continuous_after=[];l1_rows=[];draw_rows=[];metric_rows=[];baseline_rows=[];provenance=[]
    for patient,pg in e.groupby('patient_id',sort=True):
        if len(pg)<2:continue
        for a,b in itertools.combinations(sorted(pg.sample_id),2):
            base={'resource':'Valdeolivas','patient_id':patient,'section_a':a,'section_b':b,'pair_id':a+'||'+b}
            print('BALANCE_PAIR_START',patient,a,b,flush=True)
            matrices=[];model_groups=[];strata=[];blocks=[];region_indices=[];regions=[];source_records=[]
            for sid in (a,b):
                s=by_id[sid];count,spots,ids,symbols,qc,members,checks=counts_for(s)
                assert np.array_equal(ids,panel.gene_id.to_numpy(str))
                variants,block_ids,_=geometry_for(s,spots)
                local=keys[keys.sample_id==sid].set_index('barcode').reindex(spots.barcode)
                assert not local.index.duplicated().any()
                values=source_groups.reindex(local.spot_id.to_numpy()).to_numpy(float)
                matched=np.isfinite(values).all(axis=1)
                dominant=np.full(len(spots),-1,dtype=int);dominant[matched]=np.argmax(values[matched],axis=1)
                # Ordered argmax is the pre-specified tie handling, including any all-zero source rows.
                source_records.append({'sample_id':sid,'source_files':members,'checks':checks,'registered_spots':len(spots),'model_available_spots':int(matched.sum()),'model_missing_spots':int((~matched).sum()),'zero_total_model_available_spots':int((values[matched].sum(axis=1)==0).sum())})
                near,far=variants['exacthex:stroma']
                for band,mask in [('near',near),('far',far)]:
                    ix=np.flatnonzero(mask&matched)
                    matrices.append(count);model_groups.append(values);strata.append(dominant);blocks.append(block_ids);region_indices.append(ix);regions.append((sid,band))
                    availability.append(dict(base,sample_id=sid,band=band,n_original_region_spots=int(mask.sum()),n_model_available_spots=len(ix),n_missing_model_spots=int(mask.sum())-len(ix),n_model_available_blocks=len(np.unique(block_ids[ix]))))
            available=np.array([[np.sum(dom[ix]==j) for j in range(7)] for dom,ix in zip(strata,region_indices)],int)
            targets=available.min(axis=0);target_n=int(targets.sum())
            before_shares=[shares(v,ix) for v,ix in zip(model_groups,region_indices)]
            baseline_vectors=[np.asarray(m[ix].sum(axis=0)).ravel() for m,ix in zip(matrices,region_indices)]
            baseline_metrics=delta_rows(baseline_vectors,broad,scale)
            for metric,values in baseline_metrics.items():baseline_rows.append(dict(base,control='all_model_available_spots',metric=metric,**values))
            for r,(sid,band) in enumerate(regions):
                for j,group in enumerate(groups):
                    strata_rows.append(dict(base,sample_id=sid,band=band,group=group,n_available=int(available[r,j]),n_target=int(targets[j]),n_target_all_groups=target_n))
                    continuous_before.append(dict(base,sample_id=sid,band=band,group=group,n_model_available_spots=len(region_indices[r]),relative_model_share=before_shares[r][j]))
            before_l1={label:float(abs(before_shares[i]-before_shares[j]).sum()) for label,i,j in [('between_sections_near',0,2),('between_sections_far',1,3),('within_section_a_near_far',0,1),('within_section_b_near_far',2,3)]}
            accepted=0
            for rep in range(50):
                rng=np.random.default_rng(fixed_seed('dominant_strata',patient,a,b,rep));selected=[]
                for ix,dom in zip(region_indices,strata):
                    chosen=[rng.choice(ix[dom[ix]==j],int(targets[j]),replace=False) for j in range(7) if targets[j]>0]
                    selected.append(np.concatenate(chosen) if chosen else np.empty(0,int))
                for choose,dom in zip(selected,strata):
                    assert len(choose)==target_n and len(set(choose))==target_n
                    assert np.array_equal(np.bincount(dom[choose],minlength=7),targets)
                sampled_blocks=[len(np.unique(block[ix])) for block,ix in zip(blocks,selected)]
                reasons=[]
                if target_n<20:reasons.append('common_target_spots_below20')
                for (sid,band),nb in zip(regions,sampled_blocks):
                    if nb<3:reasons.append(sid+':'+band+':blocks_below3')
                ok=not reasons
                after_shares=[shares(v,ix) for v,ix in zip(model_groups,selected)]
                for r,(sid,band) in enumerate(regions):
                    for j,group in enumerate(groups):
                        continuous_after.append(dict(base,repeat=rep,sample_id=sid,band=band,group=group,n_sampled_spots=target_n,n_blocks=sampled_blocks[r],accepted=ok,relative_model_share=after_shares[r][j]))
                after_l1={label:float(abs(after_shares[i]-after_shares[j]).sum()) for label,i,j in [('between_sections_near',0,2),('between_sections_far',1,3),('within_section_a_near_far',0,1),('within_section_b_near_far',2,3)]}
                for comparison in before_l1:
                    l1_rows.append(dict(base,repeat=rep,accepted=ok,comparison=comparison,before_L1=before_l1[comparison],after_L1=after_l1[comparison]))
                depth=np.nan
                if ok:
                    accepted+=1
                    vectors=[np.asarray(m[ix].sum(axis=0)).ravel() for m,ix in zip(matrices,selected)]
                    assert all(np.equal(v,np.round(v)).all() for v in vectors)
                    depth=min(int(v.sum()) for v in vectors)
                    thin_rng=np.random.default_rng(fixed_seed('dominant_strata_depth',patient,a,b,rep))
                    thinned=[thin_rng.multivariate_hypergeometric(v.astype(np.int64),depth,method='marginals') for v in vectors]
                    assert all(int(v.sum())==depth for v in thinned)
                    assert all(np.all(t<=v) for t,v in zip(thinned,vectors))
                    for control,data in [('dominant_strata_balanced',vectors),('dominant_strata_and_depth_balanced',thinned)]:
                        values=delta_rows(data,broad,scale)
                        for metric,record in values.items():metric_rows.append(dict(base,repeat=rep,control=control,metric=metric,**record))
                draw_rows.append(dict(base,repeat=rep,n_spots_each_region=target_n,n_region_a_near_blocks=sampled_blocks[0],n_region_a_far_blocks=sampled_blocks[1],n_region_b_near_blocks=sampled_blocks[2],n_region_b_far_blocks=sampled_blocks[3],accepted=ok,unavailable_reason=';'.join(reasons),depth_thinning_performed=ok,common_total_umis=depth))
            provenance.extend(source_records);run['attempted_pairs'].append(dict(base,available_counts_by_region=available.tolist(),target_counts_by_group=targets.tolist(),n_target=target_n,draws_attempted=50,draws_accepted=accepted,draws_failed=50-accepted))
            print('BALANCE_PAIR_DONE',patient,'target',target_n,'accepted',accepted,'of50',flush=True)
            del matrices,count
    def save(rows,name,columns=None):
        frame=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows,columns=columns)
        frame.to_csv(out/name,sep='\t',index=False,na_rep='NA');return frame
    save(availability,'region_model_availability.tsv');save(strata_rows,'dominant_stratum_targets.tsv');save(continuous_before,'continuous_group_shares_before.tsv');save(continuous_after,'continuous_group_shares_after_draws.tsv');save(l1_rows,'continuous_mixture_L1_draws.tsv')
    draws=save(draw_rows,'draw_eligibility.tsv');baseline=save(baseline_rows,'baseline_model_available_profile_metrics.tsv')
    metrics=save(metric_rows,'balanced_profile_draws.tsv',columns=['resource','patient_id','section_a','section_b','pair_id','repeat','control','metric','near_value','far_value','far_minus_near'])
    pair_availability=save(run['attempted_pairs'],'patient_draw_availability.tsv')
    if not metrics.empty:
        pair=metrics.groupby(['resource','patient_id','section_a','section_b','pair_id','control','metric'],as_index=False)[['near_value','far_value','far_minus_near']].median()
        pair=pair.merge(pair_availability[['pair_id','n_target','draws_attempted','draws_accepted','draws_failed']],on='pair_id',validate='many_to_one')
        patient=pair.groupby(['resource','patient_id','control','metric'],as_index=False)[['near_value','far_value','far_minus_near']].median()
        resource=[]
        for (res,control,metric),g in patient.groupby(['resource','control','metric'],sort=True):
            resource.append({'resource':res,'control':control,'metric':metric,'attempted_patients':len(set(draws.patient_id)),'accepted_patients':len(g),'near_median':g.near_value.median(),'far_median':g.far_value.median(),'median_patient_far_minus_near':g.far_minus_near.median(),'n_patient_delta_positive':int((g.far_minus_near>0).sum()),'n_patient_delta_negative':int((g.far_minus_near<0).sum()),'interpretation':'finite_exploratory_same_RNA_model_dominant_strata_sensitivity_not_full_composition_adjustment'})
        save(pair,'balanced_profile_pair_medians.tsv');save(patient,'balanced_profile_patient_medians.tsv');save(resource,'balanced_profile_resource_summary.tsv')
        l1=pd.DataFrame(l1_rows);l1=l1[l1.accepted]
        l1med=l1.groupby(['resource','patient_id','pair_id','comparison'],as_index=False)[['before_L1','after_L1']].median();save(l1med,'continuous_mixture_L1_patient_medians.tsv')
    else:
        save([], 'balanced_profile_pair_medians.tsv',columns=['resource','patient_id','pair_id','control','metric','far_minus_near'])
        save([], 'balanced_profile_patient_medians.tsv',columns=['resource','patient_id','control','metric','far_minus_near'])
        save([], 'balanced_profile_resource_summary.tsv',columns=['resource','control','metric','attempted_patients','accepted_patients','median_patient_far_minus_near'])
        save([], 'continuous_mixture_L1_patient_medians.tsv',columns=['resource','patient_id','pair_id','comparison','before_L1','after_L1'])
    failed=draws[~draws.accepted]
    source_unique={r['sample_id']:r for r in provenance}
    run.update({'completed_utc':datetime.now(timezone.utc).isoformat(),'elapsed_seconds':time.monotonic()-t0,'status':'completed_exploratory_balance' if len(metrics) else 'all_draws_unavailable_branch_stopped_without_rule_changes','source_sections':list(source_unique.values()),'attempted_patients':len(set(draws.patient_id)),'accepted_patients':len(set(draws.loc[draws.accepted,'patient_id'])),'attempted_draws':len(draws),'accepted_draws':int(draws.accepted.sum()),'failed_draws':len(failed),'failure_reasons':failed.unavailable_reason.value_counts().to_dict(),'not_evaluated':'GSE294385 has no matched author composition input and is not composition-adjusted here.','inference_boundary':addendum['interpretation'],'p_values_or_causal_explained_fraction':False})
    run['outputs']={p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(out.iterdir()) if p.is_file() and p.name not in ['RUN_MANIFEST.json','STARTED.json']}
    write_json(out/'RUN_MANIFEST.json',run)
    print(json.dumps({k:run[k] for k in ['status','attempted_patients','accepted_patients','attempted_draws','accepted_draws','failed_draws','elapsed_seconds']}),flush=True)

if __name__=='__main__':main()
