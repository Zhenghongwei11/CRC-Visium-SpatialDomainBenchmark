#!/usr/bin/env python3
"""Compute exploratory broad-gene regional expression profiles."""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='2'
import argparse
import collections
from datetime import datetime, timezone
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import sys
import time

import h5py
import numpy as np
import pandas as pd
import scipy
from scipy.io import mmread
from scipy.sparse import csc_matrix, csr_matrix
from scipy.spatial import cKDTree
from scipy.stats import rankdata

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL=ROOT/'config/regional_profiles.json'
OFFSETS=((0,-2),(0,2),(-1,-1),(-1,1),(1,-1),(1,1))
READOUT=('TGFB1','CXCL12','ACTA2','TAGLN')
METRICS=('spearman','scaled_abs_difference','unscaled_abs_difference')
BANDS=('near','far')


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def seed(*parts):
    return int.from_bytes(hashlib.sha256('|'.join(map(str,parts)).encode()).digest()[:8],'little')


def write_json(path, record):
    Path(path).write_text(json.dumps(record,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def save_table(rows,path):
    frame=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(path,sep='\t',index=False,na_rep='NA')
    return frame


def counts_for(section,root=ROOT):
    directory=Path(os.environ.get('CRC_COUNTS_ROOT',str(root/'original_inputs/counts')))/section['resource']/section['sample_id']/'members'
    members=[]
    for e in section['source_files']:
        p=directory/e['name']; digest=sha(p)
        if p.stat().st_size!=e['size_bytes'] or digest!=e['sha256']:
            raise ValueError('Count source identity mismatch: '+str(p))
        members.append({'path':str(p.relative_to(directory)),'sha256':digest,'bytes':p.stat().st_size})
    sid=section['sample_id']; h5=directory/(sid+'_filtered_feature_bc_matrix.h5')
    if h5.exists():
        with h5py.File(h5,'r') as handle:
            m=handle['matrix']; count=csc_matrix((m['data'][:],m['indices'][:],m['indptr'][:]),shape=tuple(m['shape'][:])).T.tocsr()
            barcode=np.array([x.decode() for x in m['barcodes'][:]],dtype=str)
            ids=np.array([x.decode() for x in m['features/id'][:]],dtype=str)
            symbols=np.array([x.decode() for x in m['features/name'][:]],dtype=str)
    else:
        with gzip.open(directory/(sid+'_matrix.mtx.gz'),'rb') as f:
            count=mmread(f).T.tocsr()
        with gzip.open(directory/(sid+'_barcodes.tsv.gz'),'rt') as f:
            barcode=np.array([line.strip() for line in f if line.strip()],dtype=str)
        features=pd.read_csv(directory/(sid+'_features.tsv.gz'),sep='\t',header=None)
        ids=features.iloc[:,0].astype(str).to_numpy(); symbols=features.iloc[:,1].astype(str).to_numpy()
    if count.shape!=(section['matrix_barcode_count'],section['feature_count']):
        raise ValueError('Unexpected raw matrix dimensions '+sid)
    if len(set(barcode))!=len(barcode) or (count.data<0).any() or not np.equal(count.data,np.round(count.data)).all():
        raise ValueError('Invalid raw matrix/barcodes '+sid)
    source=root/section['source_root']; spots=pd.read_csv(source/'spot_inputs.tsv.gz',sep='\t',float_precision='round_trip')
    if spots.barcode.duplicated().any() or set(spots.barcode)!=set(barcode):
        raise ValueError('Registered barcode universe differs '+sid)
    order=pd.Index(barcode).get_indexer(spots.barcode)
    count=count[order].astype(np.int64).tocsr(); total=np.asarray(count.sum(axis=1)).ravel()
    if (total<=0).any():raise ValueError('Zero library '+sid)
    indices=[np.flatnonzero(symbols==gene)[0] for gene in READOUT]
    score=np.log1p(count[:,indices].toarray()/total[:,None]*1e4).mean(axis=1)
    old=pd.read_csv(source/'spot_scores.tsv.gz',sep='\t',float_precision='round_trip')
    old=old[(old.score_id=='primary_barrier')&(old.status=='success')].set_index('barcode').loc[spots.barcode,'score_value'].to_numpy(float)
    if not np.allclose(score,old,rtol=2e-6,atol=2e-7):raise ValueError('Four-gene recomposition differs '+sid)
    stripped=np.array([re.sub(r'\.\d+$','',s) for s in ids],dtype=str)
    unique,inverse=np.unique(stripped,return_inverse=True)
    if len(unique)!=len(ids):
        mapper=csr_matrix((np.ones(len(ids),dtype=np.int64),(np.arange(len(ids)),inverse)),shape=(len(ids),len(unique)))
        collapsed=count@mapper
    else:
        collapsed=count[:,np.argsort(stripped)].tocsr()
    symbols_for={u:[] for u in unique}
    for u,s in zip(stripped,symbols):
        if s not in symbols_for[u]:symbols_for[u].append(s)
    names=np.array([symbols_for[u][0] for u in unique],dtype=str)
    if not np.array_equal(np.asarray(collapsed.sum(axis=1)).ravel(),total):raise AssertionError('Gene collapse changed totals')
    mito=np.array([bool(re.match(r'^MT-',s,re.I)) for s in symbols])
    mito_counts=np.asarray(count[:,mito].sum(axis=1)).ravel()
    qc={'total':total,'detected':np.diff(collapsed.indptr),'mito':mito_counts}
    return collapsed,spots,unique,names,qc,members,{'four_gene_max_abs_error':float(np.max(abs(score-old))),'n_original_features':len(ids),'n_unique_gene_ids':len(unique),'n_registered':len(spots),'duplicate_gene_ids_collapsed':len(ids)-len(unique)}


def exact_bands(lattice,side,reference):
    distance=np.full(len(lattice),np.inf)
    if side.any() and reference.any():
        xy=np.column_stack((lattice[:,1]/2,np.sqrt(3)*lattice[:,0]/2))*100
        _,j=cKDTree(xy[reference]).query(xy[side],k=1)
        difference=lattice[side]-lattice[reference][j]
        distance[side]=difference[:,1]**2+3*difference[:,0]**2
    near=side&(distance<=4); raw_far=side&(distance>=16)&(distance<=64)
    lookup={tuple(lattice[i]):i for i in np.flatnonzero(side)}
    comp=np.full(len(lattice),-1,int); next_id=0
    for first in np.flatnonzero(side):
        if comp[first]>=0:continue
        queue=collections.deque([first]);comp[first]=next_id
        while queue:
            i=queue.popleft(); r,c=lattice[i]
            for dr,dc in OFFSETS:
                j=lookup.get((r+dr,c+dc))
                if j is not None and comp[j]<0:comp[j]=next_id;queue.append(j)
        next_id+=1
    far=raw_far&np.isin(comp,np.unique(comp[near]))
    return near,far


def geometry_for(section,spots):
    lattice=spots[['array_row','array_col']].to_numpy(float)
    if not np.isfinite(lattice).all() or not np.equal(lattice,np.round(lattice)).all():raise ValueError('Invalid lattice')
    lattice=lattice.astype(int)
    bins=[np.searchsorted(np.linspace(lattice[:,a].min(),lattice[:,a].max(),7)[1:-1],lattice[:,a],side='right') for a in (0,1)]
    blocks=np.asarray([f'x{a}_y{b}' for a,b in zip(*bins)],dtype=str)
    stroma=spots.coarse_label.eq('stroma').to_numpy();tumor=spots.coarse_label.eq('tumor').to_numpy()
    near,far=exact_bands(lattice,stroma,tumor);tn,tf=exact_bands(lattice,tumor,stroma)
    variants={'exacthex:stroma':(near,far),'exacthex:tumor':(tn,tf)}
    geometry=ROOT/'data/reference_inputs/geometry'/section['resource']/section['sample_id']/'geometry_arrays.npz'
    with np.load(geometry,allow_pickle=False) as old:
        assert np.array_equal(blocks,old['historical_raw_6nn|baseline|blocks'])
        assert np.array_equal(near,old['nominal_honeycomb_exact|baseline|near'])
        assert np.array_equal(far,old['nominal_honeycomb_exact|baseline|far'])
        variants['legacy6nn:stroma']=(old['historical_raw_6nn|baseline|near'].copy(),old['historical_raw_6nn|baseline|far'].copy())
    if section['resource']=='GSE294385':
        keep=~spots.raw_label.fillna('').str.contains(r'muscularis|muscle',case=False,regex=True).to_numpy()
        variants['exacthex_no_muscularis:stroma']=(near&keep,far&keep)
    return variants,blocks,geometry


def eligibility(a,b,blocks):
    n1,n2=int(a.sum()),int(b.sum());k1,k2=len(np.unique(blocks[a])),len(np.unique(blocks[b]))
    reasons=[]
    if n1<20:reasons.append('near_spots_below20')
    if n2<20:reasons.append('far_spots_below20')
    if k1<3:reasons.append('near_blocks_below3')
    if k2<3:reasons.append('far_blocks_below3')
    return {'n_near':n1,'n_far':n2,'near_blocks':k1,'far_blocks':k2,'evaluable':not reasons,'unavailable_reasons':';'.join(reasons)}


def profile(x):
    total=float(np.sum(x))
    return np.log1p(x/total*1e6) if total>0 else np.full(x.shape,np.nan)


def measures(left,right,scale):
    a=rankdata(left); b=rankdata(right);a-=a.mean();b-=b.mean()
    denominator=np.sqrt(np.dot(a,a)*np.dot(b,b))
    return {'spearman':float(np.dot(a,b)/denominator) if denominator>0 else np.nan,
            'scaled_abs_difference':float(np.mean(abs(left-right)/scale)),
            'unscaled_abs_difference':float(np.mean(abs(left-right)))}


def bootstrap(values,key):
    x=np.asarray(values,float);x=x[np.isfinite(x)]
    if len(x)<2:return np.nan,np.nan
    rng=np.random.default_rng(seed(20261025,'patient_bootstrap',*key))
    draw=np.median(x[rng.integers(0,len(x),size=(2000,len(x)))],axis=1)
    low,high=np.quantile(draw,[.025,.975]);return float(low),float(high)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,default=DEFAULT_PROTOCOL)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--smoke-sample')
    parser.add_argument('--counts',type=Path,help='Original count members organized by resource and sample')
    args=parser.parse_args();protocol=json.loads(args.protocol.read_text())
    if args.counts:os.environ['CRC_COUNTS_ROOT']=str(args.counts.resolve())
    lock=json.loads((args.protocol.parent/'PROTOCOL_LOCK_v1.json').read_text())
    if sha(args.protocol)!=lock['sha256']:raise ValueError('Protocol changed after lock')
    manifest_path=ROOT/protocol['source_manifest']; manifest=json.loads(manifest_path.read_text())
    sections=manifest['sections']; by_id={s['sample_id']:s for s in sections}
    output=args.output or ROOT/protocol['outputs']; output.mkdir(parents=True,exist_ok=True)
    if args.smoke_sample:
        s=by_id[args.smoke_sample];count,spots,ids,symbols,qc,members,checks=counts_for(s)
        variants,blocks,geometry=geometry_for(s,spots)
        for variant,masks in variants.items():
            for mask in masks:
                vector=np.asarray(count[mask].sum(axis=0)).ravel()
                assert vector.sum()==qc['total'][mask].sum()
        result={'sample_id':s['sample_id'],'checks':checks,'geometry_match':True,'profile_sum_conservation':True,'protocol_sha256':sha(args.protocol),'script_sha256':sha(__file__)}
        write_json(output/'SMOKE_CHECK.json',result);print(json.dumps(result));return
    started=datetime.now(timezone.utc).isoformat(); t0=time.monotonic()
    run={'started_utc':started,'analysis_class':protocol['analysis_class'],'protocol_sha256':sha(args.protocol),'script_sha256':sha(__file__),'source_manifest_sha256':sha(manifest_path),'software':{'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'h5py':h5py.__version__},'sections':[],'failures':[]}
    write_json(output/'STARTED.json',run)
    eligibility_rows=[];qc_rows=[];annotation_rows=[];section_data={};whole={};detections={};resource_ids={};resource_names={}
    profiles_dir=output/'section_profiles';profiles_dir.mkdir(exist_ok=True)
    for s in sections:
        sid=s['sample_id'];resource=s['resource']
        print('SECTION_START',resource,sid,flush=True)
        count,spots,ids,symbols,qc,members,checks=counts_for(s)
        variants,blocks,geometry=geometry_for(s,spots)
        if resource in resource_ids:
            assert np.array_equal(ids,resource_ids[resource]),'Source resource gene-ID universe differs'
            assert np.array_equal(symbols,resource_names[resource]),'Source resource gene-symbol mapping differs'
        else:resource_ids[resource]=ids;resource_names[resource]=symbols
        base={k:s[k] for k in ['resource','patient_id','sample_id','section_id']}
        row={'whole_counts':np.asarray(count.sum(axis=0)).ravel(),'gene_ids':ids,'gene_symbols':symbols,'detected_spot_fraction':np.asarray((count>0).sum(axis=0)).ravel()/len(spots)}
        whole[sid]=profile(row['whole_counts']);detections[sid]=row['detected_spot_fraction']
        section_data[sid]={'source':s,'profiles':{},'eligibility':{}}
        for variant,(near,far) in variants.items():
            geometry_id,compartment=variant.split(':');e=eligibility(near,far,blocks)
            eligibility_rows.append(dict(base,geometry=geometry_id,compartment=compartment,**e));section_data[sid]['eligibility'][variant]=e
            for band,mask in [('near',near),('far',far)]:
                counts_sum=np.asarray(count[mask].sum(axis=0)).ravel()
                assert counts_sum.sum()==qc['total'][mask].sum()
                for j in [0,len(ids)//2,len(ids)-1]:
                    assert counts_sum[j]==sum(int(v) for v in count[mask,j].toarray().ravel())
                row[variant.replace(':','__')+'__'+band]=counts_sum
                section_data[sid]['profiles'][(variant,band)]=profile(counts_sum)
                total=int(counts_sum.sum());n=int(mask.sum())
                qrow=dict(base,geometry=geometry_id,compartment=compartment,band=band,n_spots=n,n_blocks=len(np.unique(blocks[mask])),total_umis=total,median_spot_umis=float(np.median(qc['total'][mask])) if n else np.nan,median_detected_genes=float(np.median(qc['detected'][mask])) if n else np.nan,mitochondrial_umi_fraction=float(qc['mito'][mask].sum()/total) if total else np.nan)
                qc_rows.append(qrow)
                for label,n_label in spots.loc[mask,'raw_label'].fillna('UNLABELED').value_counts().items():
                    annotation_rows.append(dict(base,geometry=geometry_id,compartment=compartment,band=band,raw_label=label,n_spots=int(n_label),spot_fraction=float(n_label/n)))
        np.savez_compressed(profiles_dir/(sid+'.npz'),**row)
        record=dict(base,checks=checks,source_files=members,spot_inputs={'path':str((ROOT/s['source_root']/'spot_inputs.tsv.gz').relative_to(ROOT)),'sha256':sha(ROOT/s['source_root']/'spot_inputs.tsv.gz')},geometry_path=str(geometry.relative_to(ROOT)),geometry_sha256=sha(geometry),stored_profiles_sha256=sha(profiles_dir/(sid+'.npz')))
        run['sections'].append(record);write_json(output/'PROGRESS.json',{'completed_sections':len(run['sections']),'attempted_sections':len(sections),'last_sample':sid})
        del count
        print('SECTION_DONE',sid,flush=True)
    eligible=save_table(eligibility_rows,output/'regional_eligibility.tsv');save_table(qc_rows,output/'regional_qc.tsv');save_table(annotation_rows,output/'regional_annotation_composition.tsv')
    panels={}; scales={}; panel_rows=[];panel_count=[]
    for resource,ids in resource_ids.items():
        samples=[s['sample_id'] for s in sections if s['resource']==resource];symbols=resource_names[resource]
        detection=np.vstack([detections[sid]>=.10 for sid in samples]).sum(axis=0)
        exclude=np.array([bool(re.match(r'^MT-|^RPL|^RPS',x,re.I)) for x in symbols])
        broad=(detection>=math.ceil(len(samples)/2))&~exclude
        x=np.vstack([whole[sid] for sid in samples]);med=np.median(x,axis=0);scale=np.maximum(1.4826*np.median(abs(x-med),axis=0),.25);scales[resource]=scale
        variance=np.var(x,axis=0);available=np.flatnonzero(broad);rank=available[np.argsort(-variance[available],kind='stable')];top=np.zeros(len(ids),bool);top[rank[:2000]]=True
        panels[resource]={'broad':broad,'broad_without_four':broad&~np.isin(symbols,READOUT),'variable2000':top}
        for name,mask in panels[resource].items():panel_count.append({'resource':resource,'panel':name,'n_genes':int(mask.sum())})
        for j,gene in enumerate(ids):
            panel_rows.append({'resource':resource,'gene_id':gene,'gene_symbol':symbols[j],'sections_with_detection_ge10percent':int(detection[j]),'n_sections_for_gene_selection':len(samples),'excluded_mt_ribosomal':bool(exclude[j]),'broad':bool(broad[j]),'broad_without_four':bool(panels[resource]['broad_without_four'][j]),'variable2000':bool(top[j]),'whole_section_variance':variance[j],'fixed_gene_scale':scale[j]})
    save_table(panel_rows,output/'gene_panels.tsv');save_table(panel_count,output/'gene_panel_counts.tsv')
    # The gene selection inputs above are whole registered sections, without band labels.
    common=eligible[(eligible.geometry=='exacthex')].pivot(index=['resource','patient_id','sample_id'],columns='compartment',values='evaluable').fillna(False)
    common_ids=set(common.index[(common.stroma)&(common.tumor)].get_level_values('sample_id'))
    save_table([{'resource':row[0],'patient_id':row[1],'sample_id':row[2],'stroma_evaluable':bool(values.stroma),'tumor_evaluable':bool(values.tumor),'both_evaluable':bool(values.stroma and values.tumor)} for row,values in common.iterrows()],output/'cross_compartment_common_eligibility.tsv')
    within=[];between=[];lopo=[];section_metric=[]
    def analyze_set(frame,scope):
        for (resource,geometry_id,compartment),group in frame[frame.evaluable].groupby(['resource','geometry','compartment'],sort=True):
            variant=geometry_id+':'+compartment
            for patient,pg in group.groupby('patient_id',sort=True):
                for a,b in itertools.combinations(sorted(pg.sample_id),2):
                    for panel,mask in panels[resource].items():
                        values={band:measures(section_data[a]['profiles'][(variant,band)][mask],section_data[b]['profiles'][(variant,band)][mask],scales[resource][mask]) for band in BANDS}
                        for metric in METRICS:
                            within.append({'scope':scope,'resource':resource,'geometry':geometry_id,'compartment':compartment,'patient_id':patient,'section_a':a,'section_b':b,'pair_definition':'same_patient_distinct_registered_sections','panel':panel,'n_genes':int(mask.sum()),'metric':metric,'near_value':values['near'][metric],'far_value':values['far'][metric],'far_minus_near':values['far'][metric]-values['near'][metric]})
            patient_profiles={}
            for patient,pg in group.groupby('patient_id',sort=True):
                patient_profiles[patient]={band:np.mean([section_data[sid]['profiles'][(variant,band)] for sid in pg.sample_id],axis=0) for band in BANDS}
            keys=sorted(patient_profiles)
            for panel,mask in panels[resource].items():
                for a,b in itertools.combinations(keys,2):
                    vals={band:measures(patient_profiles[a][band][mask],patient_profiles[b][band][mask],scales[resource][mask]) for band in BANDS}
                    for metric in METRICS:
                        between.append({'scope':scope,'resource':resource,'geometry':geometry_id,'compartment':compartment,'patient_a':a,'patient_b':b,'panel':panel,'metric':metric,'near_value':vals['near'][metric],'far_value':vals['far'][metric],'far_minus_near':vals['far'][metric]-vals['near'][metric],'independent_patients':len(keys),'patient_pair_count':len(keys)*(len(keys)-1)//2,'interpretation':'descriptive_distances_not_independent_patient_pairs'})
        return
    analyze_set(eligible,'all_eligible')
    analyze_set(eligible[(eligible.geometry=='exacthex')&eligible.sample_id.isin(common_ids)],'joint_compartments')
    within=save_table(within,output/'within_patient_section_pairs.tsv');between=save_table(between,output/'between_patient_profile_pairs.tsv')
    group_keys=['scope','resource','geometry','compartment','panel','metric'];summary=[]
    patient=within.groupby(group_keys+['patient_id'],as_index=False)[['near_value','far_value','far_minus_near']].median()
    save_table(patient,output/'within_patient_summaries.tsv')
    for key,g in patient.groupby(group_keys,sort=True):
        low,high=bootstrap(g.far_minus_near,key)
        summary.append(dict(zip(group_keys,key),n_independent_patients=len(g),near_median=g.near_value.median(),far_median=g.far_value.median(),median_patient_far_minus_near=g.far_minus_near.median(),bootstrap_low=low,bootstrap_high=high,n_far_minus_near_positive=int((g.far_minus_near>0).sum()),n_far_minus_near_negative=int((g.far_minus_near<0).sum()),bootstrap_unit='patient',bootstrap_draws=2000))
    save_table(summary,output/'within_patient_resource_summary.tsv')
    # Pooling resources is descriptive only and keeps patient IDs resource-qualified.
    pooled=[]
    poolkeys=['scope','geometry','compartment','panel','metric']
    for key,g in patient.groupby(poolkeys,sort=True):
        low,high=bootstrap(g.far_minus_near,('pooled',)+key)
        pooled.append(dict(zip(poolkeys,key),n_independent_patients=len(g),median_patient_far_minus_near=g.far_minus_near.median(),bootstrap_low=low,bootstrap_high=high,interpretation='pooled_descriptive_resource_specific_gene_panels'))
    save_table(pooled,output/'within_patient_pooled_descriptive.tsv')
    between_summary=[]
    for key,g in between.groupby(group_keys,sort=True):
        pats=sorted(set(g.patient_a)|set(g.patient_b))
        between_summary.append(dict(zip(group_keys,key),n_independent_patients=len(pats),n_pairwise_distances=len(g),near_pair_median=g.near_value.median(),far_pair_median=g.far_value.median(),difference_of_pair_medians=g.far_value.median()-g.near_value.median(),median_pairwise_far_minus_near=g.far_minus_near.median()))
        for omitted in pats:
            rem=g[(g.patient_a!=omitted)&(g.patient_b!=omitted)]
            lopo.append(dict(zip(group_keys,key),omitted_patient=omitted,remaining_patients=len(pats)-1,near_pair_median=rem.near_value.median(),far_pair_median=rem.far_value.median(),difference_of_pair_medians=rem.far_value.median()-rem.near_value.median()))
    save_table(between_summary,output/'between_patient_resource_summary.tsv');save_table(lopo,output/'between_patient_leave_one_out.tsv')
    # Matched spot counts and then matched library depth; all metric summaries stay within patient.
    sampled=[];matched_qc=[]
    primary=eligible[(eligible.geometry=='exacthex')&eligible.evaluable]
    for (resource,compartment,pat),group in primary.groupby(['resource','compartment','patient_id'],sort=True):
        if len(group)<2:continue
        variant='exacthex:'+compartment
        for a,b in itertools.combinations(sorted(group.sample_id),2):
            arrays=[];masks=[];read_sections=[]
            for sid in (a,b):
                count,spots,ids,symbols,qc,members,checks=counts_for(by_id[sid]); geo,blocks,_=geometry_for(by_id[sid],spots)
                arrays.append(count);masks.append(tuple(np.flatnonzero(mask) for mask in geo[variant]));read_sections.append(sid)
            n=min(len(z) for pair in masks for z in pair)
            for rep in range(50):
                rng=np.random.default_rng(seed(20261025,'matched_spots',resource,compartment,pat,a,b,rep));summed=[]
                for arr,sets in zip(arrays,masks):
                    for indices in sets:
                        selected=rng.choice(indices,size=n,replace=False);assert len(set(selected))==n
                        summed.append(np.asarray(arr[selected].sum(axis=0)).ravel())
                target=min(int(v.sum()) for v in summed)
                thin_rng=np.random.default_rng(seed(20261025,'matched_depth',resource,compartment,pat,a,b,rep))
                thinned=[thin_rng.multivariate_hypergeometric(v.astype(np.int64),target,method='marginals') for v in summed]
                assert all(int(v.sum())==target for v in thinned)
                assert all(np.all(v<=original) for v,original in zip(thinned,summed))
                matched_qc.append({'resource':resource,'compartment':compartment,'patient_id':pat,'section_a':a,'section_b':b,'repeat':rep,'n_spots_each_region':n,'minimum_total_umis':target,'original_sampled_region_totals':';'.join(str(int(v.sum())) for v in summed)})
                for control,counts4 in [('spot_matched',summed),('spot_and_depth_matched',thinned)]:
                    logs=[profile(v) for v in counts4]
                    for panel,mask in panels[resource].items():
                        vals={'near':measures(logs[0][mask],logs[2][mask],scales[resource][mask]),'far':measures(logs[1][mask],logs[3][mask],scales[resource][mask])}
                        for metric in METRICS:
                            sampled.append({'resource':resource,'compartment':compartment,'patient_id':pat,'section_a':a,'section_b':b,'repeat':rep,'control':control,'panel':panel,'metric':metric,'near_value':vals['near'][metric],'far_value':vals['far'][metric],'far_minus_near':vals['far'][metric]-vals['near'][metric]})
            del arrays
            print('SAMPLING_DONE',resource,compartment,pat,a,b,flush=True)
    sampled=save_table(sampled,output/'matched_sampling_draws.tsv');save_table(matched_qc,output/'matched_sampling_qc.tsv')
    skeys=['resource','compartment','patient_id','control','panel','metric']
    sm=sampled.groupby(skeys,as_index=False)[['near_value','far_value','far_minus_near']].median()
    save_table(sm,output/'matched_sampling_patient_summaries.tsv');ss=[]
    for key,g in sm.groupby(['resource','compartment','control','panel','metric'],sort=True):
        low,high=bootstrap(g.far_minus_near,('matched',)+key)
        ss.append(dict(zip(['resource','compartment','control','panel','metric'],key),n_independent_patients=len(g),near_median=g.near_value.median(),far_median=g.far_value.median(),median_patient_far_minus_near=g.far_minus_near.median(),bootstrap_low=low,bootstrap_high=high))
    save_table(ss,output/'matched_sampling_resource_summary.tsv')
    # In the joint set, compare far-minus-near metrics between the two compartments within patient.
    joint=patient[(patient.scope=='joint_compartments')&(patient.geometry=='exacthex')]
    cross=joint.pivot(index=['resource','patient_id','panel','metric'],columns='compartment',values='far_minus_near').dropna().reset_index()
    cross['tumor_minus_stroma_far_minus_near']=cross.tumor-cross.stroma
    save_table(cross,output/'cross_compartment_patient_comparison.tsv')
    write_json(output/'COMPOSITION_HOOK.json',{'status':'not_integrated','required_input':'author composition CSV, verified barcode join and resource-wide category definition before examining regional associations','inference_boundary':'same-count modeled composition is not independent cell-type validation','no_values_imputed':True})
    run.update({'completed_utc':datetime.now(timezone.utc).isoformat(),'seconds':time.monotonic()-t0,'n_sections':len(sections),'n_regions_attempted':len(eligible),'source_counts_recomposition_passed':True,'geometry_checks_passed':True,'sampling_repeats':50,'patient_bootstrap_draws':2000,'independent_units':'patients, not genes/spots/maps/section pairs','status':'completed_exploratory_profiles_without_author_composition'})
    run['outputs']={str(p.relative_to(output)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(output.rglob('*')) if p.is_file() and p.name not in ('RUN_MANIFEST.json','STARTED.json','PROGRESS.json')}
    write_json(output/'RUN_MANIFEST.json',run)
    print(json.dumps({'status':run['status'],'seconds':run['seconds'],'panel_counts':panel_count,'sections':len(sections)}),flush=True)

if __name__=='__main__':main()
