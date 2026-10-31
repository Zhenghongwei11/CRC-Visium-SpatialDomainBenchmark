#!/usr/bin/env python3
"""Outcome-informed S/L/C/LC references for fixed original domain contrasts."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[key]='2'
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,hashlib,re,time,warnings
import numpy as np
import pandas as pd
from scipy.optimize import linprog,OptimizeWarning
from scipy.sparse import csr_matrix,hstack,vstack,eye
from regional_profiles import counts_for,sha
from scientific_seeds import residual_seed_vectors
ROOT=Path(__file__).resolve().parents[1]
DOC=ROOT/'config'
OUT=ROOT/'results/residual_composition'
READOUT=['four_gene_mean','TGFB1','CXCL12','ACTA2','TAGLN','ACTA2_TAGLN_mean','TGFB1_CXCL12_mean']
CONTROLS=['S','L','C','LC']
COMPATIBLE_CHECKPOINT_SCRIPT_SHA256=()
def js(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def calibrate(x,selected,poolblocks,local):
 n=len(selected);N=len(x);target=x[selected].mean(axis=0);P=x.shape[1]
 eq=[np.r_[np.ones(N),np.zeros(N)]];rhs=[1.]
 for j in range(P):eq.append(np.r_[x[:,j],np.zeros(N)]);rhs.append(target[j])
 if local:
  for b in np.unique(poolblocks):
   eq.append(np.r_[(poolblocks==b).astype(float),np.zeros(N)]);rhs.append(np.mean(poolblocks[selected]==b))
 uniform=np.full(N,1/N);ident=eye(N,format='csr')
 aub=vstack([hstack([ident,-ident]),hstack([-ident,-ident])],format='csr')
 with warnings.catch_warnings():
  warnings.simplefilter('ignore',OptimizeWarning)
  fit=linprog(np.r_[np.zeros(N),np.ones(N)],A_ub=aub,b_ub=np.r_[uniform,-uniform],A_eq=csr_matrix(eq),b_eq=rhs,bounds=[(0,1/n)]*N+[(0,None)]*N,method='highs',options={'threads':2,'time_limit':120,'primal_feasibility_tolerance':1e-8,'dual_feasibility_tolerance':1e-8})
 if not fit.success:return None,{'status':int(fit.status),'message':str(fit.message)}
 w=np.maximum(fit.x[:N],0);w/=w.sum();err=float(np.max(abs(w@x-target)))
 if err>1e-6 or w.max()>1/n+1e-7:return None,{'status':-1,'message':'postsolve_constraint_failure','max_mean_error':err}
 return w,{'status':int(fit.status),'message':str(fit.message),'max_mean_error':err,'weight_ess':float(1/np.sum(w*w)),'max_weight':float(w.max()),'positive_support':int((w>1e-12).sum())}
def draw_band(rng,control,poolblocks,selected,weights):
 n=len(selected);N=len(poolblocks)
 if control in ['L','LC']:
  pieces=[]
  for b in np.unique(poolblocks[selected]):
   ix=np.flatnonzero(poolblocks==b);k=int(np.sum(poolblocks[selected]==b))
   if weights is None:p=None
   else:
    p=weights[ix].copy();p[p<1e-12]=0
    if (p>0).sum()<k:return None
    p/=p.sum()
   pieces.append(rng.choice(ix,k,replace=False,p=p))
  return np.concatenate(pieces)
 p=None if weights is None else np.where(weights>=1e-12,weights,0)
 if p is not None:
  if (p>0).sum()<n:return None
  p=p/p.sum()
 return rng.choice(N,n,replace=False,p=p)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=OUT);ap.add_argument('--only',nargs='*');ap.add_argument('--counts',type=Path,required=True);ap.add_argument('--maps-results',type=Path,default=ROOT/'data/table_inputs/current_per_map_results.tsv');args=ap.parse_args();os.environ['CRC_COUNTS_ROOT']=str(args.counts.resolve());out=args.output;out.mkdir(parents=True,exist_ok=True)
 protocol=DOC/'regional_profiles.json';lock=json.loads((DOC/'PROTOCOL_LOCK_v1.json').read_text());assert sha(protocol)==lock['sha256']
 amendment=DOC/'reference_groups.json';tax=json.loads(amendment.read_text())
 groups=tax['author_model_summary_groups'];names=list(groups)
 source=ROOT/'data/reference_inputs/cell2location/W_cell_density.csv';m=pd.read_csv(source).set_index('spot_id');m.columns=[re.sub('^mean_spot_factors','',c) for c in m.columns]
 abundance=pd.DataFrame({k:m[v].sum(axis=1) for k,v in groups.items()});assert np.allclose(abundance.sum(axis=1),m.sum(axis=1))
 keys=pd.read_csv(ROOT/'data/reference_inputs/cell2location/barcode_map.tsv',sep='\t')
 manifest=ROOT/'config/source_pipeline/source_counts_manifest.json';sections=[s for s in json.loads(manifest.read_text())['sections'] if s['resource']=='Valdeolivas']
 originals=pd.read_csv(args.maps_results,sep='\t')
 originals=originals[(originals.resource=='Valdeolivas')&(originals.estimator_id=='stromal_retention')]
 run={'started_utc':datetime.now(timezone.utc).isoformat(),'script_sha256':sha(__file__),'protocol_sha256':sha(protocol),'taxonomy_sha256':sha(amendment),'composition_sha256':sha(source),'source_manifest_sha256':sha(manifest),'outcome_class':'post_outcome_conditional_reference_comparison_not_causal_attribution','fixed_attempts':200,'minimum_accepted':100,'complete_model_band_requirement':True,'source_sections':[]};js(out/'STARTED.json',run)
 rows=[];availability=[];calibrations=[];drawdiag=[];memberships=[]
 for section in sections:
  sid=section['sample_id']
  if args.only and sid not in args.only:continue
  sd=out/'sections'/sid;sd.mkdir(parents=True,exist_ok=True);done=sd/'DONE.json'
  if done.exists():
   old=json.loads(done.read_text());assert old['script_sha256'] in (sha(__file__),*COMPATIBLE_CHECKPOINT_SCRIPT_SHA256)
   for n,e in old['files'].items():assert sha(sd/n)==e
   for n,target in [('contrasts.tsv',rows),('availability.tsv',availability),('calibration.tsv',calibrations),('draw_diagnostics.tsv',drawdiag)]:
    try:target.extend(pd.read_csv(sd/n,sep='\t').to_dict('records'))
    except pd.errors.EmptyDataError:pass
   run['source_sections'].append(old);continue
  count,spots,ids,symbols,qc,sources,checks=counts_for(section)
  with np.load(ROOT/'data/reference_inputs/geometry'/section['resource']/sid/'geometry_arrays.npz') as z:
   near=z['historical_raw_6nn|baseline|near'];far=z['historical_raw_6nn|baseline|far'];blocks=z['historical_raw_6nn|baseline|blocks']
  local=keys[keys.sample_id==sid].set_index('barcode').reindex(spots.barcode);X=abundance.reindex(local.spot_id).to_numpy(float);available=np.isfinite(X).all(axis=1)
  try:maps=pd.read_csv(ROOT/section['source_root']/'input_maps.tsv.gz',sep='\t')
  except pd.errors.EmptyDataError:maps=pd.DataFrame()
  try:mapping=pd.read_csv(ROOT/section['source_root']/'domain_mapping.tsv.gz',sep='\t')
  except pd.errors.EmptyDataError:mapping=pd.DataFrame()
  sr=originals[originals.sample_id==sid]
  selection_arrays={};case_data=[];localavailability=[];localcal=[];localdraw=[]
  # Match selection/eligibility uses X/labels/geometry only; no new random contrast is evaluated before lock.
  for _,record in sr.iterrows():
   key=f'{record.method_id}|K{int(record.K)}|{record.partition_id}';meta={'resource':section['resource'],'patient_id':section['patient_id'],'sample_id':sid,'method_id':record.method_id,'K':int(record.K),'partition_id':record.partition_id,'case_id':key}
   reason=''
   if record.independent_status!='success':reason='original_domain_comparison_unavailable'
   if not available[near|far].all():reason=reason or 'incomplete_author_model_full_band'
   match=mapping[(mapping.method_id==record.method_id)&(mapping.K==record.K)&(mapping.partition_id==record.partition_id)&(mapping.selection_status=='success')] if not mapping.empty else pd.DataFrame()
   if match.empty:reason=reason or 'source_domain_mapping_unavailable'
   if reason:
    for control in CONTROLS:localavailability.append(dict(meta,control=control,accepted_draws=0,attempted_draws=0,evaluable=False,reason=reason))
    continue
   label=match.iloc[0].selected_stroma_domain;maprows=maps[(maps.method_id==record.method_id)&(maps.K==record.K)&(maps.replicate_id==record.partition_id)]
   selected=maprows.set_index('barcode').domain_label.reindex(spots.barcode).to_numpy()==label
   assert int((selected&near).sum())==int(record.n_selected_near) and int((selected&far).sum())==int(record.n_selected_far)
   banddata=[]
   for band,mask in [('near',near),('far',far)]:
    pool=np.flatnonzero(mask);sel=np.flatnonzero(selected[pool]);base=X[pool];sdx=np.maximum(base.std(axis=0),1e-6);features=base/sdx;target=features[sel].mean(axis=0);weights={}
    for c in ['C','LC']:
     weights[c],diagnostic=calibrate(features,sel,blocks[pool],c=='LC');localcal.append(dict(meta,band=band,control=c,**diagnostic))
    banddata.append((band,pool,sel,features,target,weights))
   for control in CONTROLS:
    arrays=[];accepted=np.ones(200,bool);maxerrs=[]
    failed=control in ['C','LC'] and any(d[-1][control] is None for d in banddata)
    if failed:
     localavailability.append(dict(meta,control=control,accepted_draws=0,attempted_draws=0,evaluable=False,reason='composition_calibration_failed'));continue
    for band,pool,sel,features,target,weights in banddata:
     matrix=np.full((200,len(sel)),-1,int)
     for rep in range(200):
      rng=np.random.default_rng(residual_seed_vectors()[(sid,key,control,band)][rep]);chosen=draw_band(rng,control,blocks[pool],sel,None if control in ['S','L'] else weights[control])
      ok=chosen is not None;err=float(np.max(abs(features[chosen].mean(axis=0)-target))) if ok else np.inf
      if control in ['C','LC']:ok=ok and err<=.10+1e-12
      if chosen is not None:
       assert len(np.unique(chosen))==len(sel)
       if control in ['L','LC']:assert dict(zip(*np.unique(blocks[pool][chosen],return_counts=True)))==dict(zip(*np.unique(blocks[pool][sel],return_counts=True)))
       matrix[rep]=pool[chosen]
      accepted[rep]&=ok
      localdraw.append(dict(meta,control=control,band=band,repeat=rep,accepted_band=ok,max_standardized_abundance_mean_error=err if np.isfinite(err) else None,n_spots=len(sel)))
     arrays.append(matrix)
    tag=hashlib.sha256((key+'|'+control).encode()).hexdigest()[:20]
    selection_arrays[tag+'_near']=arrays[0];selection_arrays[tag+'_far']=arrays[1];selection_arrays[tag+'_accepted']=accepted
    evaluable=int(accepted.sum())>=100
    localavailability.append(dict(meta,control=control,accepted_draws=int(accepted.sum()),attempted_draws=200,evaluable=evaluable,reason='' if evaluable else 'accepted_composition_draws_below100'))
    case_data.append((meta,record,selected,control,tag,evaluable))
  np.savez_compressed(sd/'reference_selections.npz',**selection_arrays)
  pd.DataFrame(localavailability).to_csv(sd/'availability.tsv',sep='\t',index=False)
  pd.DataFrame(localcal).to_csv(sd/'calibration.tsv',sep='\t',index=False)
  pd.DataFrame(localdraw).to_csv(sd/'draw_diagnostics.tsv',sep='\t',index=False)
  js(sd/'SELECTION_LOCK.json',{'locked_utc':datetime.now(timezone.utc).isoformat(),'script_sha256':sha(__file__),'protocol_sha256':sha(protocol),'selections_sha256':sha(sd/'reference_selections.npz'),'availability_sha256':sha(sd/'availability.tsv'),'new_reference_outcomes_computed':False})
  # Outcome computation now consumes the immutable sampled-index arrays.
  genes=[np.flatnonzero(symbols==g)[0] for g in ['TGFB1','CXCL12','ACTA2','TAGLN']];v=np.log1p(count[:,genes].toarray()/qc['total'][:,None]*1e4);v=np.column_stack([v.mean(axis=1),v,v[:,2:].mean(axis=1),v[:,:2].mean(axis=1)])
  localrows=[]
  for meta,record,selected,control,tag,evaluable in case_data:
   if not evaluable:continue
   a=selection_arrays[tag+'_accepted'];fn=np.median(v[near],axis=0)-np.median(v[far],axis=0);ds=np.median(v[near&selected],axis=0)-np.median(v[far&selected],axis=0)
   assert np.isclose(fn[0],record.full_band_contrast,rtol=2e-6,atol=2e-7) and np.isclose(ds[0],record.domain_selected_contrast,rtol=2e-6,atol=2e-7)
   contrast=np.median(v[selection_arrays[tag+'_near'][a]],axis=1)-np.median(v[selection_arrays[tag+'_far'][a]],axis=1)
   for j,name in enumerate(READOUT):
    fullscale=float(np.median(abs(v[spots.coarse_label.eq('stroma'),j]-np.median(v[spots.coarse_label.eq('stroma'),j]))));rand=abs(contrast[:,j]-fn[j]);median=float(np.median(rand));excess=float(abs(ds[j]-fn[j])-median)
    localrows.append(dict(meta,control=control,readout=name,accepted_draws=int(a.sum()),full_band_contrast=fn[j],domain_selected_contrast=ds[j],absolute_domain_departure=abs(ds[j]-fn[j]),reference_median_absolute_departure=median,excess_expression_units=excess,section_stroma_mad=fullscale,excess_mad_units=excess/fullscale if fullscale>0 else np.nan,reference_odd_even_median_difference=abs(np.median(rand[::2])-np.median(rand[1::2]))))
  pd.DataFrame(localrows).to_csv(sd/'contrasts.tsv',sep='\t',index=False)
  checkpoint={'sample_id':sid,'script_sha256':sha(__file__),'source_files':sources,'count_checks':checks,'files':{p.name:sha(p) for p in sd.iterdir() if p.is_file() and p.name!='DONE.json'}};js(done,checkpoint)
  rows.extend(localrows);availability.extend(localavailability);calibrations.extend(localcal);drawdiag.extend(localdraw);run['source_sections'].append(checkpoint)
  print('RESIDUAL_SECTION',sid,'evaluated',len(localrows),'reference_attempts',len(localavailability),flush=True)
  del count,v,maps
 for name,data in [('contrasts.tsv',rows),('availability.tsv',availability),('calibration.tsv',calibrations),('draw_diagnostics.tsv',drawdiag)]:pd.DataFrame(data).to_csv(out/name,sep='\t',index=False)
 frame=pd.DataFrame(rows)
 if not frame.empty:
  ids=['resource','patient_id','sample_id','method_id','K','partition_id','case_id'];good=frame.groupby(ids,dropna=False).control.nunique();common=good[good==4].reset_index()[ids]
  common.to_csv(out/'all_control_common_cases.tsv',sep='\t',index=False)
  result=[];patient=[]
  for scope,f in [('control_specific_available',frame),('all_control_common',frame.merge(common,on=ids,validate='many_to_one'))]:
   if f.empty:continue
   values=['excess_expression_units','excess_mad_units','reference_median_absolute_departure']
   a=f.groupby(['resource','patient_id','sample_id','method_id','K','control','readout'],as_index=False)[values].median();b=a.groupby(['resource','patient_id','sample_id','control','readout'],as_index=False)[values].median();c=b.groupby(['resource','patient_id','control','readout'],as_index=False)[values].median();c['scope']=scope;patient.append(c)
   for (resource,control,readout),g in c.groupby(['resource','control','readout']):result.append(dict(scope=scope,resource=resource,control=control,readout=readout,n_patients=len(g),**{x:float(g[x].median()) for x in values}))
  if patient:pd.concat(patient).to_csv(out/'patient_summaries.tsv',sep='\t',index=False)
  pd.DataFrame(result).to_csv(out/'resource_summaries.tsv',sep='\t',index=False)
 run.update(completed_utc=datetime.now(timezone.utc).isoformat(),status='completed_conditional_references',n_contrast_rows=len(rows),n_attempted_setting_controls=len(availability),successful_setting_controls=int(pd.DataFrame(availability).evaluable.sum()),outputs={p.name:sha(p) for p in out.glob('*.tsv')});js(out/'RUN_MANIFEST.json',run);print(json.dumps({k:run[k] for k in ['status','n_contrast_rows','n_attempted_setting_controls','successful_setting_controls']}),flush=True)
if __name__=='__main__':main()
