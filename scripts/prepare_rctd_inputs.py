#!/usr/bin/env python3
"""Sparse original counts for a frozen RCTD reference-model pilot."""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
 os.environ[_key]='2'
from pathlib import Path
import json,hashlib,gzip,argparse
import numpy as np,pandas as pd
from scipy.sparse import csr_matrix
from scipy.io import mmwrite
from regional_profiles import counts_for,geometry_for,sha,seed
ROOT=Path(__file__).resolve().parents[1]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--all',action='store_true');ap.add_argument('--section');ap.add_argument('--counts',type=Path,default=ROOT/'original_inputs/counts');ap.add_argument('--output',type=Path,default=ROOT/'results/rctd');a=ap.parse_args();os.environ['CRC_COUNTS_ROOT']=str(a.counts.resolve());doc=ROOT/'config';p=doc/'rctd_model.json';design=json.loads(p.read_text());assert sha(p)==json.loads((doc/'rctd_model_lock.json').read_text())['sha256']
 output=a.output.resolve()/'inputs';output.mkdir(parents=True,exist_ok=True)
 manifest=json.loads((ROOT/'config/source_pipeline/source_counts_manifest.json').read_text())['sections']
 if a.section is not None:assert a.section in {s['sample_id'] for s in manifest}
 genes=pd.read_csv(ROOT/'data/reference_inputs/single_cell/GSE132465_real_cell_subset_genes.tsv.gz',sep='\t',header=None).iloc[:,0].to_numpy(str);refset=set(genes)-set(design['excluded_genes'])
 for section in manifest:
  sid=section['sample_id']
  if a.section is not None and sid!=a.section:continue
  if not a.all and sid not in design['pilot_sections']:continue
  mode='all' if a.all else 'pilot';out=output/mode/sid;out.mkdir(parents=True,exist_ok=True)
  prior=json.loads((out/'INPUT_MANIFEST.json').read_text()) if (out/'INPUT_MANIFEST.json').exists() else None
  if prior is not None and (out/'counts.mtx.gz').exists():
   assert sha(out/'counts.mtx.gz')==prior['files']['counts.mtx.gz']['sha256']
   continue
  counts,spots,ids,names,qc,source,checks=counts_for(section);variants,blocks,_=geometry_for(section,spots)
  near,far=variants['exacthex:stroma'];strata=np.where(near,'stroma_near',np.where(far,'stroma_far',np.where(spots.coarse_label.eq('stroma'),'stroma_other',np.where(spots.coarse_label.eq('tumor'),'tumor','other'))))
  selected=[]
  for st in sorted(set(strata)):
   ix=np.flatnonzero(strata==st);rng=np.random.default_rng(seed('stage2RCTDpilot',sid,st));selected.extend(ix if a.all or len(ix)<=160 else rng.choice(ix,160,replace=False))
  selected=np.array(sorted(selected));common=sorted(set(names)&refset);lookup={g:i for i,g in enumerate(common)};orig=[j for j,g in enumerate(names) if g in lookup];dest=[lookup[names[j]] for j in orig]
  mapper=csr_matrix((np.ones(len(orig),dtype=np.int64),(orig,dest)),shape=(counts.shape[1],len(common)))
  subset=(counts[selected]@mapper).T.tocsc();assert np.equal(subset.data,np.round(subset.data)).all()
  with (out/'counts.mtx.gz').open('wb') as raw:
   with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as h:mmwrite(h,subset,field='integer',symmetry='general')
  if prior is not None:
   assert sha(out/'counts.mtx.gz')==prior['files']['counts.mtx.gz']['sha256'],'Restored derived counts differ from frozen input'
   print('RCTD_INPUT_RESTORED',mode,sid,subset.shape,flush=True);del counts,subset
   continue
  pd.DataFrame({'gene':common}).to_csv(out/'genes.tsv.gz',sep='\t',index=False)
  meta=spots.iloc[selected][['barcode','array_row','array_col','coarse_label','raw_label']].copy();meta['original_row_index']=selected;meta['x']=meta.array_col*50;meta['y']=meta.array_row*np.sqrt(3)*50;meta['nUMI_full_feature']=qc['total'][selected];meta['fit_gene_UMIs']=np.asarray(subset.sum(axis=0)).ravel();meta['stratum']=strata[selected];meta['near_stroma']=near[selected];meta['far_stroma']=far[selected];meta['blocks']=blocks[selected]
  meta['detected_genes_full_feature']=qc['detected'][selected]
  meta['mitochondrial_UMIs_full_feature']=qc['mito'][selected]
  meta.to_csv(out/'spots.tsv.gz',sep='\t',index=False)
  rec={'sample_id':sid,'resource':section['resource'],'patient_id':section['patient_id'],'mode':mode,'protocol_sha256':sha(p),'reference_sha256':design['reference_input_sha256'],'script_sha256':sha(__file__),'original_files':source,'original_count_checks':checks,'registered_spots':len(spots),'selected_spots':len(selected),'common_fit_genes':len(common),'excluded_genes':design['excluded_genes'],'original_feature_library_offsets_preserved':True,'spatial_symbol_duplicate_columns_aggregated':len(orig)-len(common),'files':{f.name:{'bytes':f.stat().st_size,'sha256':sha(f)} for f in out.iterdir() if f.is_file()}}
  (out/'INPUT_MANIFEST.json').write_text(json.dumps(rec,ensure_ascii=False,indent=2)+'\n');print('RCTD_INPUT',mode,sid,subset.shape,flush=True);del counts,subset
if __name__=='__main__':main()
