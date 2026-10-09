#!/usr/bin/env python3
"""Derive a patient-aware description of regional selection across map families."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from jbcb_presentation_paths import read
from jbcb_presentation_paths import default_package

METHODS={'M0_expr_kmeans':'Expression k-means','M1_spatial_concat_kmeans':'Expression and coordinate k-means',
         'M2_spatial_ward':'Spatial Ward clustering','M3_spatial_leiden':'Spatial Leiden clustering',
         'M4_spagcn_official':'SpaGCN (configured implementation)','M5_stagate_official':'STAGATE embedding with k-means',
         'M6_bayesspace_official':'BayesSpace (1,000 iterations)'}

def patient_median(data,column):
    finite=data[np.isfinite(pd.to_numeric(data[column],errors='coerce'))]
    if finite.empty:return np.nan
    method_k=finite.groupby(['patient_id','sample_id','K'])[column].median()
    section=method_k.groupby(level=['patient_id','sample_id']).median()
    patient=section.groupby(level='patient_id').median()
    return float(patient.median())

def build_method_comparison(sensitivity,effects):
    data=sensitivity[(sensitivity.graph_id=='historical_raw_6nn') & (sensitivity.setting_id=='baseline') & (sensitivity.readout_id=='primary_barrier')]
    effects=effects[(effects.graph_id=='historical_raw_6nn') & (effects.setting_id=='baseline')]
    rows=[]
    metrics=['near_retention','far_retention','selected_stroma_purity','abs_deviation_score',
             'count_only_excess_abs_deviation_score','block_matched_excess_abs_deviation_score',
             'abs_deviation_mad','count_only_excess_abs_deviation_mad','block_matched_excess_abs_deviation_mad']
    for keys,g in data.groupby(['resource','estimator_id','method_id'],sort=True):
        eligible=g[g.independent_status=='success']
        available=g[~g.independent_reason.fillna('').eq('saved_partition_unavailable')]
        geom=effects[(effects.resource==keys[0]) & (effects.estimator_id==keys[1]) & (effects.method_id==keys[2]) & ~effects.is_reference]
        geom=geom[np.isfinite(geom.stroma_domain_jaccard_vs_reference) & np.isfinite(geom.stroma_domain_centroid_shift_um)]
        row=dict(zip(['resource','estimator_id','method_id'],keys));row['method_name']=METHODS.get(keys[2],keys[2])
        row.update(n_declared_configurations=len(g),n_saved_map_configurations=len(available),
                   n_evaluable_configurations=len(eligible),n_evaluable_sections=eligible.sample_id.nunique(),
                   n_evaluable_patients=eligible.patient_id.nunique(),
                   n_standardized_configurations=int(eligible.standardized_status.eq('success').sum()),
                   n_standardized_patients=eligible.loc[eligible.standardized_status.eq('success'),'patient_id'].nunique(),
                   n_geometry_alternatives=len(geom),n_geometry_sections=geom.sample_id.nunique(),
                   n_geometry_patients=geom.patient_id.nunique())
        for metric in metrics:row['median_'+metric]=patient_median(eligible,metric)
        row['median_domain_jaccard_vs_reference']=patient_median(geom,'stroma_domain_jaccard_vs_reference')
        row['median_domain_centroid_shift_um']=patient_median(geom,'stroma_domain_centroid_shift_um')
        rows.append(row)
    return pd.DataFrame(rows)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pack-dir',type=Path,default=default_package(__file__))
    args=parser.parse_args(); si=args.pack_dir/'tables'
    build_method_comparison(read(si/'regional_map_sensitivity.tsv',sep='\t'),read(si/'current_per_map_results.tsv',sep='\t')).to_csv(si/'method_selection_comparison.tsv',sep='\t',index=False,na_rep='NA')
