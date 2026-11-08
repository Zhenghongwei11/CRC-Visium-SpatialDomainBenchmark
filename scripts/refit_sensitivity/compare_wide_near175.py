"""Independent, checkpointed four-gene sensitivity of saved versus source-refit maps.

Computes the matched-location reference from the fixed protocol; source map
and assay inputs are never overwritten. Output includes both unavailable and
new-only maps and separates old archived from new refit results.
"""
from pathlib import Path
import argparse,hashlib,json,sys,time
import numpy as np,pandas as pd

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts/analysis_pipeline'))
from regional_helpers import domain_selection,block_ids,nearest_indices,connected_stroma_components,computational_groups
from run_jbcb_corrected_statistics import exact_neighbors,control_indices,context

WORK=ROOT/'expected/refit_sensitivity'
PARSER=argparse.ArgumentParser();PARSER.add_argument('--sample');PARSER.add_argument('--settings',default='near175');PARSER.add_argument('--graphs',default='historical_raw_6nn,nominal_honeycomb_exact');PARSER.add_argument('--output',type=Path,required=True);args=PARSER.parse_args()
protocol=json.loads((ROOT/'config/analysis_pipeline/regional.json').read_text());settings=[{'id':'near175','grid':6,'near_dnum_max':12,'far_dnum_min':16,'far_dnum_max':64}];graphs=args.graphs.split(',')
table=pd.read_csv(WORK/'map_comparisons.tsv',sep='\t')
archive=pd.read_csv(ROOT/'expected/tables/wider_near_map_results.tsv',sep='\t');archive=archive[archive.readout_id.eq('four_gene_mean')].set_index(['resource','sample_id','graph_id','setting_id','method_id','K','partition_id','estimator_id'])
pool=pd.read_csv(ROOT/'data/reference_inputs/nominal_geometry/nominal_pool_all_registered.tsv.gz',sep='\t')
done=set()
if args.output.exists():
 for line in args.output.read_text().splitlines():
  z=json.loads(line);done.add(tuple(z[x] for x in ('resource','sample','graph','setting','method','K','replicate','estimator')))
started=time.monotonic();total=0
for (resource,sample),rows in table.groupby(['resource','sample'],sort=True):
 if args.sample and sample!=args.sample:continue
 section=next(z for z in protocol['source_sections'] if z['resource']==resource and z['sample_id']==sample);p=ROOT/'data/reference_inputs/sections'/resource/sample
 spots=pd.read_csv(p/'spot_inputs.tsv.gz',sep='\t').set_index('barcode');old=pd.read_csv(p/'input_maps.tsv.gz',sep='\t')
 score=pd.read_csv(p/'spot_scores.tsv.gz',sep='\t');values=score[score.score_id.eq('primary_barrier')].set_index('barcode').loc[spots.index].score_value.to_numpy(float)
 lattice=spots[['array_row','array_col']].to_numpy(int);coarse=spots.coarse_label.to_numpy(str);stroma=coarse=='stroma'
 b=pool[pool.sample_id.eq(sample)].set_index('barcode').loc[spots.index];distance=b.nearest_pure_tumor_dnum_delta_col2_plus_3_delta_row2.to_numpy(float)
 mad=float(np.median(np.abs(values[stroma]-np.median(values[stroma])))) if stroma.any() else np.nan
 geometry={}
 for graph in graphs:
  neighbors=nearest_indices(lattice) if graph=='historical_raw_6nn' else exact_neighbors(lattice)[0]
  components=connected_stroma_components(stroma,neighbors)
  for setting in settings:
   near=stroma&np.isfinite(distance)&(distance<=setting['near_dnum_max']);farraw=stroma&(distance>=setting['far_dnum_min'])&(distance<=setting['far_dnum_max']);far=farraw&np.isin(components,np.unique(components[near]));blocks=block_ids(lattice,grid=setting['grid']).astype(str)
   geometry[(graph,setting['id'])]=(neighbors,near,far,blocks)
 refit=pd.read_csv(ROOT/'data/refit_domains'/resource/(sample+'.tsv.gz'),sep='\t')
 for method,mrows in rows.groupby('method',sort=True):
  maps=refit[refit.method_id.eq(method)]
  for row in mrows.itertuples(index=False):
   oldmap=old[(old.method_id==method)&(old.K==row.K)&(old.replicate_id==row.replicate)];newmap=maps[(maps.K==row.K)&(maps.replicate_id==row.replicate)]
   variants=[]
   for t in (oldmap,newmap):
    if t.empty:variants.append(None);continue
    assert not t.barcode.duplicated().any() and set(t.barcode)==set(spots.index)
    labels=t.set_index('barcode').loc[spots.index].domain_label.to_numpy(int);selection,_=domain_selection(labels,coarse)
    variants.append((labels,selection) if selection['status']=='success' else None)
   for graph in graphs:
    for setting in settings:
     sid=setting['id'];neighbors,near,far,blocks=geometry[(graph,sid)]
     def eligible(n,f):return n.sum()>=20 and f.sum()>=20 and len(np.unique(blocks[n]))>=3 and len(np.unique(blocks[f]))>=3
     anchor_ok=eligible(near,far);anchor=float(np.median(values[near])-np.median(values[far])) if anchor_ok else None
     for est,alias in [('anchor_retention','stromal_retention'),('strict_computational_interface','stricter_adjacency')]:
      ident=(resource,sample,graph,sid,method,int(row.K),row.replicate,est)
      if ident in done:continue
      rec=dict(zip(('resource','sample','graph','setting','method','K','replicate','estimator'),ident));rec.update(patient_id=section['patient_id'],old_map=not oldmap.empty,new_map=not newmap.empty)
      key=(resource,sample,graph,sid,method,int(row.K),row.replicate,alias);previous=archive.loc[key] if key in archive.index else None
      rec['old_status']=('eligible' if previous is not None and previous.independent_status=='success' else 'not_eligible') if not oldmap.empty else 'map_absent'
      oldval=previous.block_matched_excess_abs_deviation_mad if previous is not None else np.nan;rec['old_block_excess_mad']=float(oldval) if pd.notna(oldval) else None
      selected=[]
      for variant in variants:
       if variant is None:selected.append(None);continue
       labels,sel=variant;selected.append(computational_groups(near,far,labels,sel['tumor_domain'],sel['stroma_domain'],neighbors,est))
      if previous is None:raise ValueError('archived configuration missing '+str(key))
      oldpair=selected[0];old_n=int(oldpair[0].sum()) if oldpair is not None else 0;old_f=int(oldpair[1].sum()) if oldpair is not None else 0
      if old_n!=int(previous.n_selected_near) or old_f!=int(previous.n_selected_far):raise ValueError('archived membership count mismatch '+str(key))
      old_ok=bool(anchor_ok and oldpair is not None and eligible(*oldpair))
      if old_ok!=(previous.independent_status=='success'):raise ValueError('archived eligibility mismatch '+str(key))
      newpair=selected[1];new_ok=bool(anchor_ok and newpair is not None and eligible(*newpair))
      rec['new_status']='eligible' if new_ok else ('map_absent' if newmap.empty else 'not_eligible')
      rec['new_near_n']=int(newpair[0].sum()) if newpair is not None else None;rec['new_far_n']=int(newpair[1].sum()) if newpair is not None else None
      if new_ok and np.isfinite(mad) and mad>0:
       same=selected[0] is not None and all(np.array_equal(x,y) for x,y in zip(*selected))
       if same and rec['old_block_excess_mad'] is not None:rec['new_block_excess_mad']=rec['old_block_excess_mad'];rec['calculation']='identical_selected_membership'
       else:
        sn,sf=newpair;delta=float(np.median(values[sn])-np.median(values[sf]));seedkey='|'.join(map(str,('jbcb-reader-revision-20261006-methods-v1',section['dataset_id'],sample,graph,sid,est,method,int(row.K),row.replicate,'control','block_matched')))
        rng=np.random.default_rng(int.from_bytes(hashlib.sha256(seedkey.encode()).digest()[:8],'little'));indices=[]
        for full,subset in ((near,sn),(far,sf)):
         pieces=[]
         for block in np.unique(blocks[subset]):
          candidates=np.flatnonzero(full&(blocks==block));n=int((subset&(blocks==block)).sum())
          pieces.append(np.broadcast_to(candidates,(1000,n)) if len(candidates)==n else np.stack([rng.choice(candidates,n,replace=False) for _ in range(1000)]))
         indices.append(np.concatenate(pieces,axis=1))
        ni,fi=indices;d=np.median(values[ni],axis=1)-np.median(values[fi],axis=1)
        rec['new_block_excess_mad']=(abs(delta-anchor)-float(np.median(np.abs(d-anchor))))/mad;rec['calculation']='fresh_reference_draws'
      else:rec['new_block_excess_mad']=None;rec['calculation']='new_ineligible_or_zero_mad'
      with args.output.open('a') as f:f.write(json.dumps(rec,allow_nan=False)+'\n')
      total+=1
 print('DONE',resource,sample,'written',total,'seconds',round(time.monotonic()-started),flush=True)
print('FINISHED',total,flush=True)
