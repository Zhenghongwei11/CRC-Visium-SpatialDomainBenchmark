#!/usr/bin/env python3
"""Extract all independently eligible figure/purity/component summaries.

Post-outcome reporting view of fully checked fixed-map results; no new fitting.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

from configuration import parse_configuration
C = parse_configuration(__doc__)
ROOT = C.workspace
PROTOCOL = C.protocol
PLAN = C.plan
OUTPUT = C.results
BASE = C.results
P = json.loads(PROTOCOL.read_text())
METRICS = ['abs_deviation_score', 'abs_deviation_mad', 'near_retention', 'far_retention',
           'selected_stroma_purity', 'count_only_excess_abs_deviation_score',
           'block_matched_excess_abs_deviation_score', 'count_only_excess_abs_deviation_mad',
           'block_matched_excess_abs_deviation_mad']


def main():
    source = ROOT / P['output_dir'] / 'map_sensitivity.tsv'
    table = pd.read_csv(source, sep='\t', float_precision='round_trip')
    keys = ['resource', 'patient_id', 'sample_id', 'graph_id', 'estimator_id', 'setting_id', 'readout_id']
    valid = table[table.independent_status.eq('success')]
    cells = valid.groupby(keys + ['method_id', 'K'])[METRICS].median().reset_index()
    section_values = cells.groupby(keys)[METRICS].median().reset_index()
    counts = table.assign(included=table.independent_status.eq('success'),
                          standardized=table.independent_status.eq('success') & table.standardized_status.eq('success')) \
        .groupby(keys).agg(n_declared_maps=('included', 'size'), n_evaluable_maps=('included', 'sum'),
                          n_standardized_maps=('standardized', 'sum')).reset_index()
    sections = counts.merge(section_values, on=keys, how='left', validate='one_to_one')
    patient_keys = [x for x in keys if x != 'sample_id']
    patient_values = sections.groupby(patient_keys)[METRICS].median().reset_index()
    patient_counts = sections.assign(evaluable_section=sections.n_evaluable_maps.gt(0)).groupby(patient_keys).agg(
        n_attempted_sections=('sample_id', 'size'), n_evaluable_sections=('evaluable_section', 'sum'),
        n_declared_maps=('n_declared_maps', 'sum'), n_evaluable_maps=('n_evaluable_maps', 'sum'),
        n_standardized_maps=('n_standardized_maps', 'sum')).reset_index()
    patients = patient_counts.merge(patient_values, on=patient_keys, how='left', validate='one_to_one')
    resources = patients.groupby(['resource', 'graph_id', 'estimator_id', 'setting_id', 'readout_id'])[METRICS].median().reset_index()
    for name, data in [('method_k', cells), ('section', sections), ('patient', patients), ('resource', resources)]:
        data.insert(0, 'eligibility_frame', 'independent_eligible')
        data.to_csv(OUTPUT / f'figure_main_{name}_summary.tsv', sep='\t', index=False)
    primary_patients = patients[patients.graph_id.eq('historical_raw_6nn') & patients.setting_id.eq('baseline') &
                                patients.readout_id.eq('primary_barrier')]
    if len(primary_patients) != 24:
        raise ValueError('Expected all 12 attempted patients x two estimators at baseline')
    primary_patients.to_csv(OUTPUT / 'main_baseline_all12patients_2estimators.tsv', sep='\t', index=False)
    components = resources[resources.graph_id.eq('historical_raw_6nn') & resources.setting_id.eq('baseline')]
    if len(components) != 28:
        raise ValueError('Expected two resources x two estimators x seven readouts')
    components.to_csv(OUTPUT / 'main_baseline_components_resource_summary.tsv', sep='\t', index=False)
    component_counts = table[table.graph_id.eq('historical_raw_6nn') & table.setting_id.eq('baseline')].assign(
        eligible=lambda d: d.independent_status.eq('success'),
        zero_mad=lambda d: d.independent_status.eq('success') & d.standardized_status.eq('zero_mad')) \
        .groupby(['resource', 'estimator_id', 'readout_id']).agg(
            declared_maps=('eligible', 'size'), eligible_maps=('eligible', 'sum'), zero_mad_eligible_maps=('zero_mad', 'sum')).reset_index()
    component_counts.to_csv(OUTPUT / 'main_baseline_component_availability.tsv', sep='\t', index=False)
    method_primary = table[table.graph_id.eq('historical_raw_6nn') & table.setting_id.eq('baseline') &
                           table.estimator_id.eq('anchor_retention') & table.readout_id.eq('primary_barrier')]
    method_rows = []
    for (resource, method), attempted in method_primary.groupby(['resource', 'method_id']):
        selected = attempted[attempted.independent_status.eq('success')]
        k_cells = selected.groupby(['patient_id', 'sample_id', 'K'])[METRICS].median()
        method_sections = k_cells.groupby(['patient_id', 'sample_id']).median()
        method_patients = method_sections.groupby('patient_id').median()
        row = {'resource': resource, 'method_id': method, 'n_declared_maps': len(attempted),
               'n_eligible_maps': len(selected), 'n_attempted_sections': attempted.sample_id.nunique(),
               'n_eligible_sections': selected.sample_id.nunique(), 'n_attempted_patients': attempted.patient_id.nunique(),
               'n_eligible_patients': selected.patient_id.nunique()}
        row.update({metric: float(method_patients[metric].median()) for metric in METRICS})
        method_rows.append(row)
    pd.DataFrame(method_rows).to_csv(OUTPUT / 'main_baseline_by_method_resource_summary.tsv', sep='\t', index=False)
    purity_rows = []
    primary = table[table.readout_id.eq('primary_barrier')]
    for key, attempted in primary.groupby(['resource', 'graph_id', 'estimator_id', 'setting_id']):
        for cutoff in [0, .5, .75]:
            eligible = attempted[attempted.independent_status.eq('success') & attempted.selected_stroma_purity.ge(cutoff)]
            method_k = eligible.groupby(['patient_id', 'sample_id', 'method_id', 'K'])[METRICS].median()
            section = method_k.groupby(['patient_id', 'sample_id']).median()
            patient = section.groupby('patient_id').median()
            row = dict(zip(['resource', 'graph_id', 'estimator_id', 'setting_id'], key))
            row.update(eligibility_frame='independent_eligible', purity_threshold=cutoff,
                       n_declared_maps=len(attempted), n_eligible_maps=len(eligible),
                       n_attempted_sections=attempted.sample_id.nunique(), n_eligible_sections=eligible.sample_id.nunique(),
                       n_attempted_patients=attempted.patient_id.nunique(), n_eligible_patients=eligible.patient_id.nunique())
            row.update({metric: float(patient[metric].median()) for metric in METRICS})
            for metric in ['abs_deviation_score', 'abs_deviation_mad', 'count_only_excess_abs_deviation_mad',
                           'block_matched_excess_abs_deviation_mad']:
                row['map_median_' + metric] = float(eligible[metric].median())
            purity_rows.append(row)
    purity = pd.DataFrame(purity_rows)
    purity.to_csv(OUTPUT / 'figure_main_purity_summary.tsv', sep='\t', index=False)
    purity[purity.graph_id.eq('historical_raw_6nn') & purity.setting_id.eq('baseline')].to_csv(
        OUTPUT / 'main_baseline_purity_summary.tsv', sep='\t', index=False)
    manifest = {'input': str(source.relative_to(ROOT)), 'input_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                'extract_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'patient_baseline_rows': len(primary_patients), 'component_baseline_rows': len(components),
                'purity_rows': len(purity), 'analysis': 'post_outcome_reporting_view; independent qualification only'}
    (OUTPUT / 'main_frame_extraction_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest))


if __name__ == '__main__':
    main()
