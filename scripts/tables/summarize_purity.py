#!/usr/bin/env python3
"""Summarize domain purity and regional eligibility by source."""
import argparse
from pathlib import Path
import pandas as pd

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    data=pd.read_csv(args.input,sep='\t');data=data[data.estimator_id.eq('stromal_retention')];rows=[]
    for resource,g in data.groupby('resource'):
        valid=g[g.selected_stroma_purity.notna()];good=g[g.status.eq('success')];keep=good[good.selected_stroma_purity>=.9]
        rows.append(dict(resource=resource,n_declared_maps=len(g),n_maps_with_purity=len(valid),n_purity_below90=int((valid.selected_stroma_purity<.9).sum()),n_maps_purity_ge90=int((valid.selected_stroma_purity>=.9).sum()),n_regional_evaluable=len(good),n_regional_evaluable_purity_ge90=len(keep),n_retained_sections=keep.sample_id.nunique(),n_retained_patients=keep.patient_id.nunique()))
    args.output.parent.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_csv(args.output,sep='\t',index=False)
if __name__=='__main__':main()
