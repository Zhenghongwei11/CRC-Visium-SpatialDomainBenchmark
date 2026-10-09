#!/usr/bin/env python3
"""Assemble the study's patient summaries and use the published field names."""
import argparse,csv,gzip,io,shutil,json
from pathlib import Path
import pandas as pd
import numpy as np

def read(p):return pd.read_csv(p,sep='\t',float_precision='round_trip')
def write(data,folder,name):data.to_csv(folder/name,sep='\t',index=False)
def assemble(spatial,tables,config,precision,markers,withholding):
    write(read(spatial/'main_baseline_all12patients_2estimators.tsv'),tables,'primary_patient_summary.tsv')
    purity=read(spatial/'main_baseline_purity_summary.tsv').rename(columns={'purity_threshold':'purity_min','n_eligible_maps':'n_retained_maps','n_eligible_sections':'n_retained_sections','n_eligible_patients':'n_retained_patients','map_median_block_matched_excess_abs_deviation_mad':'median_block_matched_excess_abs_deviation_mad'})
    write(purity,tables,'primary_stromal_fraction_summary.tsv')
    joint=read(spatial/'anchor_joint_sensitivity.tsv')
    joint['independent_width']=joint.independent_high-joint.independent_low
    joint['joint_width']=joint.joint_high-joint.joint_low
    joint['joint_to_independent_width_ratio']=joint.joint_width/joint.independent_width
    write(joint[joint.setting_id.ne('near175')],tables,'common_block_intervals.tsv')
    write(joint[joint.setting_id.eq('near175')],tables,'wider_near_anchor_intervals.tsv')
    for old,new in [('near175_map_sensitivity.tsv','wider_near_map_results.tsv'),
                    ('near175_independent_patient_summary.tsv','wider_near_patient_summary.tsv'),
                    ('near175_independent_resource_summary.tsv','wider_near_resource_summary.tsv')]:
        shutil.copy2(spatial/old,tables/new)
    workspace=spatial.parent.parent
    protocol=json.loads((config/'regional.json').read_text());membership=[]
    for section in protocol['source_sections']:
        resource,sample=section['resource'],section['sample_id']
        n=len(pd.read_csv(workspace/section['source_root']/'spot_inputs.tsv.gz',sep='\t',usecols=['barcode']))
        with np.load(spatial/'sections'/resource/sample/'near175_geometry.npz') as expanded,np.load(workspace/'analysis/regional/sections'/resource/sample/'geometry_arrays.npz') as baseline:
            for graph in protocol['graph_ids']:
                row={'sample_id':sample,'graph_id':graph}
                for band in ['near','far']:
                    new=np.unpackbits(expanded[graph+'|near175|'+band])[:n].astype(bool)
                    old=baseline[graph+'|baseline|'+band].astype(bool)
                    row[band+'_gained']=int((new & ~old).sum());row[band+'_lost']=int((old & ~new).sum())
                membership.append(row)
    write(pd.DataFrame(membership),tables,'wider_near_membership.tsv')
    for old,new in [('near100_near175_common_map_resource_summary.tsv','near_band_common_map_summary.tsv'),('near100_near175_all_map_comparison.tsv','near_band_map_comparison.tsv')]:shutil.copy2(spatial/old,tables/new)
    for folder,target,names in [(precision,'control_reference_precision',['control_reference_map_precision.tsv','control_reference_patient_precision.tsv','control_reference_resource_precision.tsv']),
                                (withholding,'readout_withholding',['availability.tsv','map_results.tsv','paired_map_results.tsv','patient_summary.tsv','resource_summary.tsv','section_summary.tsv'])]:
        destination=tables/target;destination.mkdir(exist_ok=True)
        for name in names:shutil.copy2(folder/name,destination/name)
    for folder in sorted((withholding/'sections').glob('*/*')):
        destination=tables/'readout_withholding/sections'/folder.parent.name/folder.name
        destination.mkdir(parents=True,exist_ok=True)
        for name in ['map_results.tsv','feature_universe.tsv.gz']:shutil.copy2(folder/name,destination/name)
    for name in ['contractile_marker_section_summary.tsv','contractile_marker_block_summary.tsv']:shutil.copy2(markers/name,tables/name)
    shutil.copy2(markers/'contractile_marker_spots.tsv.gz',tables/'contractile_marker_spots.tsv.gz')
    shutil.copy2(config/'map_parameters.tsv',tables/'map_producer_parameters.tsv')
    shutil.copy2(config/'TABLE_FIELD_DICTIONARY.tsv',tables/'TABLE_FIELD_DICTIONARY.tsv')
    # Table 1 uses independently eligible domain selections in the main analysis.
    maps=read(tables/'regional_map_sensitivity.tsv');maps=maps[maps.graph_id.eq('historical_raw_6nn') & maps.setting_id.eq('baseline') & maps.readout_id.eq('primary_barrier') & maps.estimator_id.eq('anchor_retention')]
    anchors=read(tables/'anchor_section_summary.tsv');rows=[]
    for resource,g in maps.groupby('resource'):
        a=anchors[anchors.resource.eq(resource)];eligible=g[g.independent_status.eq('success')];complete=a[a.anchor_status.eq('evaluable')]
        rows.append({'resource':resource,'Sections analyzed':g.sample_id.nunique(),'Patients represented':g.patient_id.nunique(),'Sections with complete stromal bands':complete.sample_id.nunique(),'Patients with complete stromal bands':complete.patient_id.nunique(),'Sections with domain-selected comparisons':eligible.sample_id.nunique(),'Patients with domain-selected comparisons':eligible.patient_id.nunique(),'Map settings examined':len(g),'Map settings with domain-selected comparisons':len(eligible)})
    counts=pd.DataFrame(rows).set_index('resource').T[['Valdeolivas','GSE294385']].reset_index(names='Regional comparison')
    write(counts,tables,'current_table1.tsv')

def published_names(folder,mapping):
    with mapping.open(newline='') as h:rows=list(csv.DictReader(h,delimiter='\t'))
    fields={r['original_name']:r['submission_name'] for r in rows if r['name_type']=='field'}
    values={r['original_name']:r['submission_name'] for r in rows if r['name_type']=='category'}
    for p in sorted(folder.rglob('*')):
        if not (p.name.endswith('.tsv') or p.name.endswith('.tsv.gz')) or p.name in ['TABLE_FIELD_NAME_MAP.tsv','TABLE_FIELD_DICTIONARY.tsv']:continue
        data=gzip.open(p,'rt',newline='').read() if p.name.endswith('.gz') else p.read_text()
        original=list(csv.reader(io.StringIO(data),delimiter='\t'))
        if not original:continue
        converted=[[fields.get(v,v) for v in original[0]]]+[[values.get(v,v) for v in r] for r in original[1:]]
        output=io.StringIO(newline='');csv.writer(output,delimiter='\t',lineterminator='\n').writerows(converted)
        if p.name.endswith('.gz'):
            p.write_bytes(gzip.compress(output.getvalue().encode(),mtime=0))
        else:p.write_text(output.getvalue())
    shutil.copy2(mapping,folder/'TABLE_FIELD_NAME_MAP.tsv')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['spatial','tables','config','precision','markers','withholding']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();assemble(**vars(a))
