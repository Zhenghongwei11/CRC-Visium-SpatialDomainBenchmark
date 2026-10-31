#!/usr/bin/env python3
"""Preserve all final high-K candidates and compare two outcome-independent rules."""
from __future__ import annotations
import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_name]='2'
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits
import pure_domains as prior

ROOT=Path(__file__).resolve().parents[1]
DOC=ROOT/'config'
OLD=ROOT/'results/pure_domains'
DEFAULT=ROOT/'results/coverage_qualified_domains'
RULES=('largest_pure','largest_coverage_qualified_pure')


def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def stamp():return datetime.now(timezone.utc).isoformat()


def jsave(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def independent_lattice_graph(spots):
    coordinates=spots[['array_row','array_col']].to_numpy(int);lookup={tuple(c):i for i,c in enumerate(coordinates)}
    row=[];col=[]
    for i,(r,c) in enumerate(coordinates):
        for dr,dc in [(0,2),(1,-1),(1,1)]:
            j=lookup.get((r+dr,c+dc))
            if j is not None:row.extend([i,j]);col.extend([j,i])
    return csr_matrix((np.ones(len(row)),(row,col)),shape=(len(spots),len(spots)))


def coverage(mask,stroma,near,far,blocks,graph,lowest):
    members=np.flatnonzero(mask);n=len(members);ns=int(stroma[mask].sum());nnear=int((mask&near).sum());nfar=int((mask&far).sum())
    bn=len(np.unique(blocks[mask&near]));bf=len(np.unique(blocks[mask&far]));nc=connected_components(graph[members][:,members],directed=False)[0] if n else 0
    pure=bool(n and ns/n>=.90);good=bool(pure and nnear>=20 and nfar>=20 and bn>=3 and bf>=3 and nc==1)
    reasons=[]
    if not n:reasons.append('no_selected_region')
    if n and not pure:reasons.append('whole_domain_purity_below90percent')
    if nnear<20:reasons.append('near_stroma_spots_below20')
    if nfar<20:reasons.append('far_stroma_spots_below20')
    if bn<3:reasons.append('near_blocks_below3')
    if bf<3:reasons.append('far_blocks_below3')
    if n and nc!=1:reasons.append('not_connected')
    return dict(region_size=n,annotated_stroma_count=ns,whole_domain_stroma_fraction=ns/n if n else np.nan,n_near=nnear,n_far=nfar,n_near_blocks=bn,n_far_blocks=bf,n_connected_components=nc,pure_ge90=pure,coverage_qualified=good,lowest_initial_region_id=int(lowest),unavailable_reason=';'.join(reasons))


def process_section(s,protocol,stage2,outroot,identity):
    resource,sid=s['resource'],s['sample_id'];out=outroot/'sections'/resource/sid;done=out/'DONE.json'
    if done.exists():
        d=json.loads(done.read_text())
        assert d['identity']==identity,'Checkpoint code/protocol identity changed'
        for name,digest in d['files'].items():assert sha(out/name)==digest
        return d
    out.mkdir(parents=True,exist_ok=True)
    counts,barcodes,symbols,_,members=prior.load_counts(Path(os.environ['CRC_COUNTS_ROOT'])/resource/sid/'members',s)
    spot_path=ROOT/s['source_root']/'spot_inputs.tsv.gz';spots=pd.read_csv(spot_path,sep='\t')
    assert not spots.barcode.duplicated().any()
    order=pd.Index(barcodes).get_indexer(spots.barcode);assert len(order)==len(barcodes) and min(order)>=0 and len(set(order))==len(order)
    counts=counts[order].tocsr();graph,edges,stroma,near,far,blocks=prior.graph_and_geometry(spots)
    independently_rebuilt=independent_lattice_graph(spots);assert (graph!=independently_rebuilt).nnz==0
    geo=ROOT/'data/reference_inputs/geometry'/resource/sid/'geometry_arrays.npz'
    with np.load(geo) as saved:
        assert np.array_equal(near,saved['nominal_honeycomb_exact|baseline|near'])
        assert np.array_equal(far,saved['nominal_honeycomb_exact|baseline|far'])
        refblocks=saved['historical_raw_6nn|baseline|blocks'];internal=np.array([f'x{b//6}_y{b%6}' for b in blocks]);assert np.array_equal(refblocks,internal)
    excluded=np.isin(symbols,prior.FOUR);denom=np.asarray(counts[:,~excluded].sum(axis=1)).ravel();assert(denom>0).all()
    x=counts.multiply((1e4/denom)[:,None]).tocsr();x.data=np.log1p(x.data)
    means=np.asarray(x.mean(axis=0)).ravel();variance=np.maximum(np.asarray(x.power(2).mean(axis=0)).ravel()-means**2,0)
    candidates=np.flatnonzero(~excluded);indices=candidates[np.argsort(-variance[candidates],kind='stable')[:2000]]
    old_dir=OLD/'sections'/resource/sid;old_features=pd.read_csv(old_dir/'fit_features.tsv',sep='\t')
    assert np.array_equal(indices,old_features.feature_index.to_numpy())
    assert np.array_equal(np.asarray(symbols)[indices],old_features.gene_symbol.to_numpy())
    assert np.allclose(variance[indices],old_features.variance,atol=1e-10,rtol=1e-12)
    dense=x[:,indices].toarray().astype(np.float32)
    pcs=PCA(n_components=min(20,len(spots)-1,len(indices)-1),svd_solver='randomized',random_state=11).fit_transform(dense).astype(float)
    del dense,x
    # The stored v1 per-spot readout is checked, but not used for candidate selection.
    names=prior.unique_names(symbols);values=np.log1p(counts[:,[names.index(g) for g in prior.FOUR]].toarray()/np.asarray(counts.sum(axis=1)).ravel()[:,None]*1e4)
    with np.load(old_dir/'measurement.npz') as old_measure:
        assert np.array_equal(old_measure['barcodes'],spots.barcode.to_numpy(str))
        assert np.array_equal(old_measure['near'],near) and np.array_equal(old_measure['far'],far)
        assert np.allclose(old_measure['values'],values,atol=1e-14,rtol=1e-12)
    old_cov=pd.read_csv(old_dir/'coverage.tsv',sep='\t');old_masks=np.load(old_dir/'selected_regions.npz',allow_pickle=False)
    final_labels={};selected_masks={};candidate_rows=[];setting_rows=[];checks=[]
    for method in protocol['domain_pilot']['methods']:
        for k in protocol['domain_pilot']['K']:
            labels=KMeans(n_clusters=k,n_init=10,random_state=11).fit_predict(pcs) if method=='expression_kmeans' else prior.fit_ward(pcs,graph,k)
            initial=prior.split_connected(labels,graph)
            for q in (.25,.50):
                key=f'{method}|K{k}|q{int(q*100)}';regions,threshold,merges=prior.merge_regions(initial,edges,pcs,stroma,q)
                array=np.full(len(spots),-1,np.int32);pure=[];qualified=[]
                base=dict(resource=resource,patient_id=s['patient_id'],sample_id=sid,method=method,requested_K=k,merge_quantile=q,setting_id=key,actual_fitted_K=len(np.unique(labels)),n_initial_connected_regions=len(np.unique(initial)),n_final_regions=len(regions),merge_threshold=threshold,n_merges=merges)
                for rid,r in sorted(regions.items()):
                    member=r['members'];assert(array[member]==-1).all();array[member]=rid
                    mask=np.zeros(len(spots),bool);mask[member]=True
                    c=coverage(mask,stroma,near,far,blocks,independently_rebuilt,r['lowest'])
                    assert c['n_connected_components']==1
                    assert c['annotated_stroma_count']==r['ns'] and c['region_size']==r['total']
                    candidate_rows.append(dict(base,region_id=int(rid),**c))
                    if c['pure_ge90']:pure.append((rid,r,c))
                    if c['coverage_qualified']:qualified.append((rid,r,c))
                assert(array>=0).all();final_labels[key]=array
                rank=lambda item:(-item[1]['ns'],-item[1]['total'],item[1]['lowest'])
                pure.sort(key=rank);qualified.sort(key=rank)
                original_mask=np.zeros(len(spots),bool);new_mask=np.zeros(len(spots),bool)
                if pure:original_mask[pure[0][1]['members']]=True
                if qualified:new_mask[qualified[0][1]['members']]=True
                oldmask=old_masks[key];assert np.array_equal(original_mask,oldmask),'Refit changed original largest selection: '+sid+' '+key
                old_row=old_cov[old_cov.mask_key==key].iloc[0]
                assert len(regions)==old_row.n_final_regions and len(pure)==old_row.n_pure_regions
                assert np.isclose(threshold,old_row.merge_threshold,atol=1e-10,rtol=1e-12)
                assert bool(len(qualified))==bool(old_row.any_pure_region_evaluable)
                assert prior.eligible(original_mask&near,original_mask&far,blocks)==bool(old_row.primary_evaluable)
                if old_row.primary_evaluable:assert np.array_equal(original_mask,new_mask)
                for rule,chosen,pick in [('largest_pure',original_mask,pure[0] if pure else None),('largest_coverage_qualified_pure',new_mask,qualified[0] if qualified else None)]:
                    selected_masks[key+'|'+rule]=chosen
                    result=coverage(chosen,stroma,near,far,blocks,independently_rebuilt,pick[1]['lowest'] if pick else -1)
                    assert rule!='largest_coverage_qualified_pure' or bool(result['coverage_qualified'])==bool(qualified)
                    setting_rows.append(dict(base,selection_rule=rule,selected_region_id=int(pick[0]) if pick else -1,n_pure_regions=len(pure),n_coverage_qualified_pure_regions=len(qualified),anatomical_evaluable=prior.eligible(near,far,blocks),n_full_near=int(near.sum()),n_full_far=int(far.sum()),selected_mask_key=key+'|'+rule,**result))
                checks.append({'setting_id':key,'original_mask_matches_v1':True,'all_regions_connected_and_partition_spots':True,'new_coverage_matches_prior_any_eligible':True,'overlap_rules_identical_when_original_eligible':True})
    old_masks.close()
    np.savez_compressed(out/'all_final_region_labels.npz',barcodes=spots.barcode.to_numpy(str),**final_labels)
    np.savez_compressed(out/'selected_rules.npz',**selected_masks)
    pd.DataFrame(candidate_rows).to_csv(out/'candidate_regions.tsv.gz',sep='\t',index=False,na_rep='NA')
    pd.DataFrame(setting_rows).to_csv(out/'selection_coverage.tsv',sep='\t',index=False,na_rep='NA')
    record=dict(resource=resource,sample_id=sid,patient_id=s['patient_id'],identity=identity,counts=members,source_spots_sha256=sha(spot_path),source_v1_measurement_sha256=sha(old_dir/'measurement.npz'),source_v1_selected_sha256=sha(old_dir/'selected_regions.npz'),n_settings=len(checks),n_candidate_regions=len(candidate_rows),structural_verification=checks,files={n:sha(out/n) for n in ['all_final_region_labels.npz','selected_rules.npz','candidate_regions.tsv.gz','selection_coverage.tsv']})
    jsave(done,record)
    print('SECTION_DONE',resource,sid,'old',sum(r['coverage_qualified'] for r in setting_rows if r['selection_rule']==RULES[0]),'new',sum(r['coverage_qualified'] for r in setting_rows if r['selection_rule']==RULES[1]),'candidates',len(candidate_rows),flush=True)
    return record


def assemble(sections,out,identity,records):
    cover=pd.concat([pd.read_csv(out/'sections'/s['resource']/s['sample_id']/'selection_coverage.tsv',sep='\t') for s in sections],ignore_index=True)
    candidates=pd.concat([pd.read_csv(out/'sections'/s['resource']/s['sample_id']/'candidate_regions.tsv.gz',sep='\t') for s in sections],ignore_index=True)
    cover.to_csv(out/'selection_coverage.tsv',sep='\t',index=False,na_rep='NA');candidates.to_csv(out/'all_candidate_regions.tsv.gz',sep='\t',index=False,na_rep='NA')
    assert len(cover)==22*12*2 and cover[['sample_id','setting_id','selection_rule']].duplicated().sum()==0
    resource_rows=[]
    for keys,g in cover.groupby(['resource','method','requested_K','merge_quantile','selection_rule'],sort=True):
        good=g[g.coverage_qualified]
        resource_rows.append(dict(zip(['resource','method','requested_K','merge_quantile','selection_rule'],keys),attempted_sections=len(g),attempted_patients=g.patient_id.nunique(),eligible_sections=len(good),eligible_patients=good.patient_id.nunique()))
    pd.DataFrame(resource_rows).to_csv(out/'resource_coverage.tsv',sep='\t',index=False)
    lock={'locked_utc':stamp(),'selection_coverage_sha256':sha(out/'selection_coverage.tsv'),'candidate_regions_sha256':sha(out/'all_candidate_regions.tsv.gz'),'all_memberships_saved':True,'n_strategy_settings':264,'n_setting_rule_rows':528,'outcomes_known_scope':'Earlier v1 outcomes known; stage2 rule fixed before new coverage-qualified region contrasts. No expression outcome selected any region.'}
    jsave(out/'COVERAGE_LOCK.json',lock)
    contrasts=[]
    for s in sections:
        sid=s['sample_id'];local=cover[cover.sample_id==sid]
        old=OLD/'sections'/s['resource']/sid
        with np.load(old/'measurement.npz') as measurement,np.load(out/'sections'/s['resource']/sid/'selected_rules.npz') as masks:
            values=measurement['values'];near=measurement['near'];far=measurement['far'];spots=pd.read_csv(ROOT/s['source_root']/'spot_inputs.tsv.gz',sep='\t');stroma=spots.coarse_label.eq('stroma').to_numpy()
            readouts={g:values[:,i] for i,g in enumerate(prior.FOUR)};readouts.update(four_gene_mean=values.mean(axis=1),signaling_pair=values[:,:2].mean(axis=1),contractile_pair=values[:,2:].mean(axis=1))
            for _,row in local.iterrows():
                mask=masks[row.selected_mask_key]
                for name,v in readouts.items():
                    full=float(np.median(v[near])-np.median(v[far])) if row.anatomical_evaluable else np.nan
                    selected=float(np.median(v[near&mask])-np.median(v[far&mask])) if row.coverage_qualified else np.nan
                    mad=float(np.median(abs(v[stroma]-np.median(v[stroma])))) if stroma.any() else np.nan
                    contrasts.append({**{k:row[k] for k in ['resource','patient_id','sample_id','method','requested_K','merge_quantile','setting_id','selection_rule','coverage_qualified','unavailable_reason']},'readout':name,'units':'mean_or_single_log1p_CP10K_expression','full_anatomical_evaluable':bool(row.anatomical_evaluable),'full_contrast':full,'selected_contrast':selected,'selected_minus_full':selected-full,'abs_selected_minus_full':abs(selected-full),'section_stroma_mad':mad,'full_contrast_mad':full/mad if mad>0 else np.nan,'selected_minus_full_mad':(selected-full)/mad if mad>0 else np.nan,'zero_or_unavailable_mad':not(mad>0)})
    d=pd.DataFrame(contrasts);d.to_csv(out/'all_setting_rule_readout_contrasts.tsv',sep='\t',index=False,na_rep='NA');assert len(d)==528*7
    good=d[d.coverage_qualified]
    group=['resource','patient_id','method','requested_K','merge_quantile','selection_rule','readout']
    metrics=['full_contrast','selected_contrast','selected_minus_full','abs_selected_minus_full']
    ps=good.groupby(group,as_index=False)[metrics].median();nsection=good.groupby(group).sample_id.nunique().rename('n_sections').reset_index();ps=ps.merge(nsection,on=group,validate='one_to_one');ps.to_csv(out/'patient_by_setting_summaries.tsv',sep='\t',index=False,na_rep='NA')
    sec=good.groupby(['resource','patient_id','sample_id','selection_rule','readout'],as_index=False)[metrics].median();patient=sec.groupby(['resource','patient_id','selection_rule','readout'],as_index=False)[metrics].median();patient.to_csv(out/'patient_hierarchical_summaries.tsv',sep='\t',index=False,na_rep='NA')
    patient_resource=patient.groupby(['resource','selection_rule','readout'],as_index=False)[metrics].median();numbers=patient.groupby(['resource','selection_rule','readout']).patient_id.nunique().rename('n_independent_patients').reset_index();patient_resource.merge(numbers,on=['resource','selection_rule','readout']).to_csv(out/'resource_hierarchical_summaries.tsv',sep='\t',index=False,na_rep='NA')
    keys=['resource','patient_id','sample_id','method','requested_K','merge_quantile','setting_id','readout']
    matched=good[good.selection_rule==RULES[0]].merge(good[good.selection_rule==RULES[1]],on=keys,suffixes=('_original','_coverage'),validate='one_to_one')
    for name in metrics:assert np.allclose(matched[name+'_original'],matched[name+'_coverage'],atol=1e-14,rtol=0)
    matched['contrast_difference_new_minus_original']=matched.selected_contrast_coverage-matched.selected_contrast_original;matched.to_csv(out/'matched_rule_setting_contrasts.tsv',sep='\t',index=False,na_rep='NA')
    mpatient=matched.groupby(['resource','patient_id','readout'],as_index=False)[['contrast_difference_new_minus_original']].median();mpatient.to_csv(out/'matched_rule_patient_summaries.tsv',sep='\t',index=False,na_rep='NA')
    # Same explicit method/K/q contrasts across eligible sections per patient remain visible.
    old=pd.read_csv(OLD/'contrasts.tsv',sep='\t');joined=good[good.selection_rule==RULES[0]].merge(old,on=['resource','patient_id','sample_id','method','requested_K','merge_quantile','readout'],suffixes=('_new','_v1'),validate='one_to_one')
    assert len(joined)==len(old)
    for name in ['full_contrast','selected_contrast','selected_minus_full']:assert np.allclose(joined[name+'_new'],joined[name+'_v1'],atol=1e-14,rtol=0)
    jsave(out/'VERIFICATION.json',{'n_strategy_settings':264,'n_setting_rule_rows':len(cover),'n_candidates':len(candidates),'all_candidate_memberships_saved':True,'all264_original_masks_match_v1':True,'all_v1_original_eligible_readout_rows_reproduced':len(joined),'all_final_candidates_connected':True,'whole_domain_purity_denominator':'entire final candidate region','selected_contrast_population':'annotated stroma intersected with bands','new_eligibility_matches_prior_any_pure_region_flag':True,'overlap_rule_differences_zero':True,'rule_comparison_limit':'On settings where original largest is eligible, the new rule selects the identical region; added availability cannot be described as improved matched contrast accuracy.'})
    return cover,candidates


def main():
    global OLD
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=DEFAULT);parser.add_argument('--sample',action='append');parser.add_argument('--previous',type=Path,required=True);parser.add_argument('--counts',type=Path,required=True);args=parser.parse_args();OLD=args.previous.resolve();os.environ['CRC_COUNTS_ROOT']=str(args.counts.resolve())
    protocol_path=ROOT/'config/regional_profiles.json';protocol=json.loads(protocol_path.read_text());manifest_path=ROOT/protocol['source_manifest'];allsections=json.loads(manifest_path.read_text())['sections'];sections=[s for s in allsections if not args.sample or s['sample_id'] in args.sample]
    assert sha(protocol_path)==json.loads((DOC/'PROTOCOL_LOCK_v1.json').read_text())['sha256']
    stage={'analysis_class':'outcome_informed_coverage_qualified_domain_sensitivity'}
    identity={'protocol_sha256':sha(protocol_path),'source_manifest_sha256':sha(manifest_path),'script_sha256':sha(__file__),'domain_helper_sha256':sha(prior.__file__)}
    args.output.mkdir(parents=True,exist_ok=True);started=stamp();t0=time.monotonic();records=[]
    with threadpool_limits(limits=2):
        for section in sections:
            print('SECTION_START',section['resource'],section['sample_id'],flush=True);records.append(process_section(section,protocol,stage,args.output,identity))
    if args.sample:
        jsave(args.output/'SMOKE_CHECK.json',{'identity':identity,'sample_ids':[s['sample_id'] for s in sections],'n_settings':sum(d['n_settings'] for d in records),'original_masks_match_v1':True,'status':'smoke_section_complete_no_global_contrast_selection'});return
    coverage,candidates=assemble(sections,args.output,identity,records)
    run={'started_utc':started,'completed_utc':stamp(),'elapsed_seconds':time.monotonic()-t0,'identity':identity,'analysis_class':stage['analysis_class'],'status':'completed_coverage_qualified_rule_comparison_not_identity_validation','n_sections':len(sections),'n_strategy_settings':264,'n_setting_rule_rows':len(coverage),'n_candidate_regions':len(candidates),'python':platform.python_version(),'sections':records,'outputs':{p.name:sha(p) for p in args.output.iterdir() if p.is_file() and p.name!='RUN_MANIFEST.json'}};jsave(args.output/'RUN_MANIFEST.json',run)
    print(coverage.groupby(['resource','selection_rule']).coverage_qualified.agg(['sum','count']).to_string(),flush=True)

if __name__=='__main__':main()
