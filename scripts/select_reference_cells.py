#!/usr/bin/env python3
"""Select the recorded 1,400 real CRC reference cells from source annotations."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]
ANNOTATION_SHA256='4bf2ebdadc88da53b0c46e654a6576e841925addc25c981d7cdbc0d988b4677c'
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--annotation',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 with a.annotation.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 if h!=ANNOTATION_SHA256:raise ValueError('Annotation does not match the original source')
 cfg=json.loads((ROOT/'config/reference_groups.json').read_text());df=pd.read_csv(a.annotation,sep='\t',dtype=str)
 known={t:g for g,ts in cfg['common_reference_groups'].items() for t in ts};allgroups={t:g for g,ts in cfg['author_model_summary_groups'].items() for t in ts}
 df['stage2_author_group']=df.Cell_subtype.map(allgroups);df['stage2_reference_group']=df.Cell_subtype.map(known)
 if df.stage2_author_group.isna().any() or set(df.loc[df.stage2_reference_group.isna(),'Cell_subtype'])!={'Unknown'}:raise ValueError('Unexpected source label set')
 rng=np.random.default_rng(20261026);ids=[]
 for group in cfg['common_reference_groups']:
  subset=df[df.stage2_reference_group.eq(group)];pools={pat:list(rng.permutation(sorted(v.Index))) for pat,v in sorted(subset.groupby('Patient'))};picked=[];cycle=0;target=min(200,len(subset))
  while len(picked)<target:
   for pool in pools.values():
    if cycle<len(pool):picked.append(pool[cycle])
    if len(picked)==target:break
   cycle+=1
  ids.extend(picked)
 result=df.set_index('Index').loc[ids].reset_index();result['selection_order']=np.arange(1,len(result)+1)
 a.output.parent.mkdir(parents=True,exist_ok=True);result.to_csv(a.output,sep='\t',index=False);print('Selected cells:',len(result))
if __name__=='__main__':main()
