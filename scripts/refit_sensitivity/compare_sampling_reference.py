from pathlib import Path
import argparse,json,sys,time
import numpy as np,pandas as pd
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'scripts/analysis_pipeline'))
from regional_helpers import domain_selection,block_ids,nearest_indices,computational_groups
from run_jbcb_corrected_statistics import control_indices,context
parser=argparse.ArgumentParser();parser.add_argument('--sample');parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
work=root/'expected/refit_sensitivity';protocol=json.loads((root/'config/analysis_pipeline/regional.json').read_text())
table=pd.read_csv(work/'map_comparisons.tsv',sep='\t')
archive=pd.read_csv(root/'expected/tables/regional_map_sensitivity.tsv',sep='\t')
archive=archive[(archive.graph_id=='historical_raw_6nn')&(archive.setting_id=='baseline')&(archive.readout_id=='four_gene_mean')]
archive=archive.set_index(['resource','sample_id','method_id','K','partition_id','estimator_id'])
pool=pd.read_csv(root/'data/reference_inputs/nominal_geometry/nominal_pool_all_registered.tsv.gz',sep='\t')
out=args.output;done=set()
if out.exists():
 for line in out.read_text().splitlines():
  z=json.loads(line);done.add((z['resource'],z['sample'],z['method'],z['K'],z['replicate'],z['estimator']))
started=time.monotonic()
for (resource,sample),sample_rows in table[table.status.eq('matched')].groupby(['resource','sample'],sort=True):
 if args.sample and sample!=args.sample:continue
 p=root/'data/reference_inputs/sections'/resource/sample
 spots=pd.read_csv(p/'spot_inputs.tsv.gz',sep='\t').set_index('barcode')
 scores=pd.read_csv(p/'spot_scores.tsv.gz',sep='\t');values=scores[scores.score_id.eq('primary_barrier')].set_index('barcode').loc[spots.index].score_value.to_numpy()
 b=pool[pool.sample_id.eq(sample)].set_index('barcode').loc[spots.index];near=b.nominal_near_pure_stroma_dnum_le_4.to_numpy(bool);far=b.nominal_far_final_after_near_component_rule.to_numpy(bool)
 lattice=spots[['array_row','array_col']].to_numpy(int);blocks=block_ids(lattice,grid=6).astype(str);neighbors=nearest_indices(lattice);coarse=spots.coarse_label.to_numpy(str)
 mad=float(np.median(np.abs(values[coarse=='stroma']-np.median(values[coarse=='stroma']))));anchor=float(np.median(values[near])-np.median(values[far]))
 section=next(z for z in protocol['source_sections'] if z['resource']==resource and z['sample_id']==sample)
 refit=pd.read_csv(root/'data/refit_domains'/resource/(sample+'.tsv.gz'),sep='\t')
 for method,method_rows in sample_rows.groupby('method',sort=True):
  maps=refit[refit.method_id.eq(method)]
  groups={(int(k),str(r)):g for (k,r),g in maps.groupby(['K','replicate_id'])}
  for row in method_rows.itertuples(index=False):
   labels=None;selection=None
   for est,alias,oldstatus,newstatus in [('anchor_retention','stromal_retention',row.old_status,row.new_status),('strict_computational_interface','stricter_adjacency',row.strict_old_status,row.strict_new_status)]:
    ident=(resource,sample,method,int(row.K),row.replicate,est)
    if ident in done:continue
    rec=dict(zip(['resource','sample','method','K','replicate','estimator'],ident));rec.update(old_status=oldstatus,new_status=newstatus,ARI=float(row.ARI))
    refkey=(resource,sample,method,int(row.K),row.replicate,alias)
    expected=archive.loc[refkey] if refkey in archive.index else None
    for kind in protocol['control_kinds']:
     col=kind+'_excess_abs_deviation_mad';rec['old_'+kind+'_excess_mad']=float(expected[col]) if expected is not None and pd.notna(expected[col]) else None
    if newstatus=='eligible':
     if float(row.ARI)==1.0:
      for kind in protocol['control_kinds']:rec['new_'+kind+'_excess_mad']=rec['old_'+kind+'_excess_mad']
      rec['calculation']='same_partition_as_archive'
     else:
      if labels is None:
       tab=groups[(int(row.K),row.replicate)];labels=tab.set_index('barcode').loc[spots.index].domain_label.to_numpy(int);selection,_=domain_selection(labels,coarse)
      sn,sf=computational_groups(near,far,labels,selection['tumor_domain'],selection['stroma_domain'],neighbors,est)
      assert sn.sum()>=20 and sf.sum()>=20 and len(np.unique(blocks[sn]))>=3 and len(np.unique(blocks[sf]))>=3 and mad>0
      delta=float(np.median(values[sn])-np.median(values[sf]))
      for kind in protocol['control_kinds']:
       key=context(protocol,section,'historical_raw_6nn','baseline',est,method,int(row.K),row.replicate,'control',kind)
       ni,fi=control_indices(blocks,near,far,sn,sf,kind,key,protocol);d=np.median(values[ni],axis=1)-np.median(values[fi],axis=1)
       rec['new_'+kind+'_excess_mad']=(abs(delta-anchor)-float(np.median(np.abs(d-anchor))))/mad
      rec['calculation']='fresh_reference_draws'
    else:
     for kind in protocol['control_kinds']:rec['new_'+kind+'_excess_mad']=None
     rec['calculation']='new_not_eligible'
    with out.open('a') as f:f.write(json.dumps(rec,allow_nan=False)+'\n')
 print('DONE',resource,sample,'elapsed',round(time.monotonic()-started),flush=True)
print('FINISHED',flush=True)
