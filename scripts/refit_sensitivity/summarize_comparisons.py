#!/usr/bin/env python3
"""Summarize baseline and 175-micrometer domain-refit comparisons by patient."""
from pathlib import Path
import argparse, json
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
KEY = ['resource', 'sample', 'method', 'K', 'replicate', 'estimator']


def read_records(path):
    if path.name.endswith(('.tsv', '.tsv.gz')):
        return pd.read_csv(path, sep='\t')
    return pd.read_json(path, lines=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--wide', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    x = read_records(args.baseline)
    if len(x) != 1666 or x.duplicated(KEY).any():
        raise ValueError('Baseline input must contain all 1,666 matched-map/estimator records')
    sections = json.loads((ROOT / 'config/source_pipeline/source_counts_manifest.json').read_text())['sections']
    ids = {(z['resource'], z['sample_id']): z['patient_id'] for z in sections}
    x['patient_id'] = [ids[(r, s)] for r, s in zip(x.resource, x['sample'])]
    patients = []
    for origin in ('old', 'new'):
        cols = [origin + '_count_only_excess_mad', origin + '_block_matched_excess_mad']
        d = x[x[origin + '_status'].eq('eligible')].copy()
        if not d[cols].notna().all().all():
            raise ValueError('Eligible map is missing an excess estimate')
        setting = d.groupby(['resource', 'patient_id', 'sample', 'method', 'K', 'estimator'], as_index=False)[cols].median()
        section = setting.groupby(['resource', 'patient_id', 'sample', 'estimator'], as_index=False)[cols].median()
        patient = section.groupby(['resource', 'patient_id', 'estimator'], as_index=False)[cols].median()
        counts = d.groupby(['resource', 'patient_id', 'estimator']).size().rename('n_evaluable_maps').reset_index()
        patient = patient.merge(counts, on=['resource', 'patient_id', 'estimator'], validate='one_to_one')
        patient = patient.rename(columns={cols[0]: 'size_only_mad', cols[1]: 'location_matched_mad'})
        patient['origin'] = origin
        patients.append(patient)
    patient = pd.concat(patients, ignore_index=True)
    patient['location_reduction'] = patient.size_only_mad - patient.location_matched_mad
    resource = patient.groupby(['resource', 'estimator', 'origin']).agg(
        n_patients=('patient_id', 'size'), size_only_mad=('size_only_mad', 'median'),
        location_matched_mad=('location_matched_mad', 'median'),
        patients_decreased=('location_reduction', lambda values: int((values > 0).sum()))).reset_index()
    wide = read_records(args.wide)
    if len(wide) != 3360 or wide.duplicated(KEY + ['graph', 'setting']).any():
        raise ValueError('Wide-band input must contain all 3,360 configuration records')
    wide = wide[wide.graph.eq('historical_raw_6nn') & wide.estimator.eq('anchor_retention')]
    base = x[x.estimator.eq('anchor_retention')]
    joined = base.merge(wide, on=KEY, suffixes=('_baseline', '_wide'), validate='one_to_one')
    if not joined.patient_id_baseline.eq(joined.patient_id_wide).all():
        raise ValueError('Baseline and wider-band patient identifiers do not match')
    cols = ['old_block_matched_excess_mad', 'old_block_excess_mad',
            'new_block_matched_excess_mad', 'new_block_excess_mad']
    joint = joined.dropna(subset=cols).copy()
    if not (joint.old_status_baseline.eq('eligible') & joint.new_status_baseline.eq('eligible') &
            joint.old_status_wide.eq('eligible') & joint.new_status_wide.eq('eligible')).all():
        raise ValueError('Four-way common set contains an ineligible estimate')
    setting = joint.groupby(['resource', 'patient_id_baseline', 'sample', 'method', 'K'], as_index=False)[cols].median()
    section = setting.groupby(['resource', 'patient_id_baseline', 'sample'], as_index=False)[cols].median()
    within = section.groupby(['resource', 'patient_id_baseline'], as_index=False)[cols].median()
    wide_resource = within.groupby('resource', as_index=False)[cols].median()
    wide_counts = joint.groupby('resource').agg(common_maps_all_four=('sample', 'size'), patients=('patient_id_baseline', 'nunique')).reset_index()
    wide_resource = wide_counts.merge(wide_resource, on='resource', validate='one_to_one').rename(columns={
        cols[0]: 'old_block_excess_mad_baseline', cols[1]: 'old_block_excess_mad_near175',
        cols[2]: 'new_block_excess_mad_baseline', cols[3]: 'new_block_excess_mad_near175'})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    patient.to_csv(args.output_dir / 'baseline_patient_summaries.tsv', sep='\t', index=False, na_rep='NA')
    resource.to_csv(args.output_dir / 'baseline_resource_summaries.tsv', sep='\t', index=False, na_rep='NA')
    wide_resource.to_csv(args.output_dir / 'wide_band_common_resource_summaries.tsv', sep='\t', index=False, na_rep='NA')
    joint.to_csv(args.output_dir / 'wide_band_four_way_common_maps.tsv', sep='\t', index=False, na_rep='NA')
    print('Summarized', len(patient), 'patient rows and', len(joint), 'four-way common maps')


if __name__ == '__main__':
    main()
