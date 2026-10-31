#!/usr/bin/env python3
"""Summarize reference-inferred cell abundances across regions and domains."""
from pathlib import Path
import argparse, hashlib, itertools, json, os, re
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from regional_profiles import counts_for, geometry_for, profile, measures

ROOT=Path(__file__).resolve().parents[1]
DOC=ROOT/'config'
OUT=ROOT/'results/inferred_composition'
PROFILES=ROOT/'results/regional_profiles'
DOMAIN_ROOT=ROOT/'results/pure_domains'

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    global OUT, PROFILES, DOMAIN_ROOT
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--counts',type=Path,default=ROOT/'original_inputs/counts')
    p.add_argument('--profiles',type=Path,default=PROFILES)
    p.add_argument('--domains',type=Path,default=DOMAIN_ROOT)
    p.add_argument('--output',type=Path,default=OUT)
    args=p.parse_args();os.environ['CRC_COUNTS_ROOT']=str(args.counts.resolve());PROFILES=args.profiles.resolve();DOMAIN_ROOT=args.domains.resolve();OUT=args.output.resolve()
    plan=json.loads((DOC/'COMPOSITION_PLAN_v1.json').read_text());lock=json.loads((DOC/'COMPOSITION_LOCK_v1.json').read_text())
    assert sha(DOC/'COMPOSITION_PLAN_v1.json')==lock['sha256']
    OUT.mkdir(parents=True,exist_ok=True)
    source=ROOT/'data/reference_inputs/cell2location'
    frames={arm:pd.read_csv(source/name).set_index('spot_id') for arm,name in [('mean','W_cell_density.csv'),('q05','W_cell_density_q05.csv')]}
    labels=[x for v in plan['groups'].values() for x in v]
    assert all(not x.index.duplicated().any() for x in frames.values())
    assert set(frames['mean'].index)==set(frames['q05'].index)
    for arm,frame in frames.items():
        frame.columns=[re.sub(r'^(mean|q05)_spot_factors','',x) for x in frame.columns]
        assert set(frame.columns)==set(labels)
        assert np.isfinite(frame.to_numpy()).all() and (frame.to_numpy()>=0).all()
    keys=pd.read_csv(ROOT/'data/reference_inputs/cell2location/barcode_map.tsv',sep='\t')
    assert not keys[['sample_id','barcode']].duplicated().any()
    manifest=json.loads((ROOT/'config/source_pipeline/source_counts_manifest.json').read_text())
    sections=[s for s in manifest['sections'] if s['resource']=='Valdeolivas']
    eligibility=pd.read_csv(PROFILES/'regional_eligibility.tsv',sep='\t')
    panels=pd.read_csv(PROFILES/'gene_panels.tsv',sep='\t');panel=panels[panels.resource.eq('Valdeolivas')].copy()
    broad=panel.broad.to_numpy(bool);scale=panel.fixed_gene_scale.to_numpy(float)
    groups=list(plan['groups']);rows=[];subtypes=[];coverage=[];domains=[];region_shares={};matched_profiles={};provenance=[]
    for section in sections:
        sid=section['sample_id'];counts,spots,ids,symbols,qc,members,checks=counts_for(section)
        assert np.array_equal(ids,panel.gene_id.to_numpy(str))
        variants,blocks,_=geometry_for(section,spots)
        local=keys[keys.sample_id.eq(sid)].set_index('barcode').reindex(spots.barcode)
        source_ids=local.spot_id.to_numpy()
        values={arm:frame.reindex(source_ids).to_numpy(float) for arm,frame in frames.items()}
        matched=np.isfinite(values['mean']).all(axis=1)
        assert np.array_equal(matched,np.isfinite(values['q05']).all(axis=1))
        base={'resource':section['resource'],'patient_id':section['patient_id'],'sample_id':sid}

        def summarize(mask,metadata,kind):
            good=mask&matched;n=int(mask.sum());nm=int(good.sum())
            coverage.append(dict(base,kind=kind,**metadata,n_region_spots=n,n_matched_spots=nm,n_missing_spots=n-nm))
            answer={}
            for arm,matrix in values.items():
                sums=matrix[good].sum(axis=0) if nm else np.full(len(labels),np.nan)
                all_total=float(sums.sum())
                shares=[]
                for group,types in plan['groups'].items():
                    idx=[list(frames[arm].columns).index(t) for t in types]
                    total=float(sums[idx].sum());share=total/all_total if all_total>0 else np.nan
                    rec=dict(base,**metadata,arm=arm,group=group,n_matched_spots=nm,mean_inferred_abundance=total/nm if nm else np.nan,relative_model_share=share)
                    (rows if kind=='anatomical_band' else domains).append(rec)
                    shares.append(share)
                for i,label in enumerate(frames[arm].columns):
                    if kind=='anatomical_band':
                        subtypes.append(dict(base,**metadata,arm=arm,subtype=label,n_matched_spots=nm,mean_inferred_abundance=sums[i]/nm if nm else np.nan,relative_model_share=sums[i]/all_total if all_total>0 else np.nan))
                if np.isfinite(shares).all():assert np.isclose(sum(shares),1)
                answer[arm]=np.array(shares)
            return answer

        for compartment in ['stroma','tumor']:
            near,far=variants['exacthex:'+compartment]
            e=eligibility[eligibility.sample_id.eq(sid)&eligibility.geometry.eq('exacthex')&eligibility.compartment.eq(compartment)].iloc[0]
            for band,mask in [('near',near),('far',far)]:
                meta={'compartment':compartment,'band':band,'evaluable':bool(e.evaluable)}
                answer=summarize(mask,meta,'anatomical_band')
                region_shares[(sid,compartment,band)]=answer
                vector=np.asarray(counts[mask&matched].sum(axis=0)).ravel()
                matched_profiles[(sid,compartment,band)]=profile(vector)
        # Original maps are source labels; whole domain and its annotated stroma
        # remain separate populations in the model-composition tables.
        src=ROOT/section['source_root']
        try:
            maps=pd.read_csv(src/'input_maps.tsv.gz',sep='\t');mapping=pd.read_csv(src/'domain_mapping.tsv.gz',sep='\t')
        except pd.errors.EmptyDataError:
            maps=pd.DataFrame();mapping=pd.DataFrame()
        declared=mapping[mapping.selection_status.eq('success')].drop_duplicates(['method_id','K','partition_id']) if not mapping.empty else pd.DataFrame()
        for _,r in declared.iterrows():
            m=maps[maps.method_id.eq(r.method_id)&maps.K.eq(r.K)&maps.replicate_id.eq(r.partition_id)]
            if m.empty:continue
            assert not m.barcode.duplicated().any()
            assignments=m.set_index('barcode').domain_label.reindex(spots.barcode)
            mask=assignments.to_numpy()==r.selected_stroma_domain
            for population,selected in [('whole_domain',mask),('annotated_stroma',mask&spots.coarse_label.eq('stroma').to_numpy())]:
                summarize(selected,{'strategy':'original_lowK','domain_id':f'{r.method_id}|K{r.K}|{r.partition_id}','population':population},'domain')
        pilot=DOMAIN_ROOT/'sections'/section['resource']/sid/'selected_regions.npz'
        with np.load(pilot) as saved:
            for key in saved.files:
                mask=saved[key]
                for population,selected in [('whole_domain',mask),('annotated_stroma',mask&spots.coarse_label.eq('stroma').to_numpy())]:
                    summarize(selected,{'strategy':'highK_merged','domain_id':key,'population':population},'domain')
        provenance.append(dict(base,source_files=members,keymap_rows=len(local),matched=int(matched.sum()),missing=int((~matched).sum())))
        print('COMPOSITION_SECTION',sid,'matched',int(matched.sum()),'of',len(spots),flush=True)
        del counts
    def save(items,name):pd.DataFrame(items).to_csv(OUT/name,sep='\t',index=False,na_rep='NA')
    save(rows,'regional_group_composition.tsv');save(subtypes,'regional_subtype_composition.tsv');save(coverage,'join_and_region_coverage.tsv');save(domains,'domain_group_composition.tsv')
    full_pairs=pd.read_csv(PROFILES/'within_patient_summaries.tsv',sep='\t')
    pair_rows=[]
    for compartment in ['stroma','tumor']:
        eligible=eligibility[eligibility.resource.eq('Valdeolivas')&eligibility.geometry.eq('exacthex')&eligibility.compartment.eq(compartment)&eligibility.evaluable]
        for patient,g in eligible.groupby('patient_id'):
            for a,b in itertools.combinations(sorted(g.sample_id),2):
                for arm in frames:
                    distances={band:float(np.abs(region_shares[(a,compartment,band)][arm]-region_shares[(b,compartment,band)][arm]).sum()) for band in ['near','far']}
                    matched={band:measures(matched_profiles[(a,compartment,band)][broad],matched_profiles[(b,compartment,band)][broad],scale[broad])['scaled_abs_difference'] for band in ['near','far']}
                    r=full_pairs[full_pairs.scope.eq('all_eligible')&full_pairs.resource.eq('Valdeolivas')&full_pairs.geometry.eq('exacthex')&full_pairs.compartment.eq(compartment)&full_pairs.patient_id.eq(patient)&full_pairs.panel.eq('broad')&full_pairs.metric.eq('scaled_abs_difference')].iloc[0]
                    pair_rows.append(dict(resource='Valdeolivas',patient_id=patient,compartment=compartment,arm=arm,section_a=a,section_b=b,composition_L1_near=distances['near'],composition_L1_far=distances['far'],composition_L1_far_minus_near=distances['far']-distances['near'],rna_full_far_minus_near=r.far_minus_near,rna_matched_spots_far_minus_near=matched['far']-matched['near']))
    save(pair_rows,'within_patient_composition_and_rna.tsv')
    associations=[];pr=pd.DataFrame(pair_rows)
    for (compartment,arm),g in pr.groupby(['compartment','arm']):
        x=g.composition_L1_far_minus_near.to_numpy();y=g.rna_matched_spots_far_minus_near.to_numpy()
        rho=float(spearmanr(x,y).statistic) if len(g)>=3 and np.std(x)>0 and np.std(y)>0 else np.nan
        associations.append({'compartment':compartment,'arm':arm,'n_independent_patients':len(g),'spearman_delta_composition_vs_delta_rna':rho,'interpretation':'descriptive_same_RNA_model_context_not_causal_or_independent_validation'})
    save(associations,'descriptive_composition_associations.tsv')
    receipt={'completed_utc':datetime.now(timezone.utc).isoformat(),'status':'completed_same_RNA_author_composition_context','source_csv':{n:{'sha256':sha(source/n),'bytes':(source/n).stat().st_size} for n in ['W_cell_density.csv','W_cell_density_q05.csv']},'plan_sha256':sha(DOC/'COMPOSITION_PLAN_v1.json'),'script_sha256':sha(__file__),'sections':provenance,'groups':plan['groups'],'q05_interpretation':'lower-summary weights, not a composition credible interval','outputs':{p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='RUN_MANIFEST.json'}}
    (OUT/'RUN_MANIFEST.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(pd.DataFrame(associations).to_string(index=False))

if __name__=='__main__':main()
