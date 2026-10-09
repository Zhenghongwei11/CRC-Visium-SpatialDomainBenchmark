#!/usr/bin/env python3
"""Derive scientific supporting tables from the complete corrected rerun."""
from pathlib import Path
import argparse
import json
import shutil
import numpy as np
import pandas as pd
from run_jbcb_corrected_statistics import read, vectors_for, unique, METRICS, sha256

ROOT = Path(__file__).resolve().parents[2] / "results/workspace"
GRAPH = 'historical_raw_6nn'


def write(df, si, name):
    # Local paths belong in reproducibility records, not scientific tables.
    df = df.drop(columns=['source_root'], errors='ignore')
    df.to_csv(si / name, sep='\t', index=False, na_rep='NA')


def support(df, prefix, margin):
    point, low, high = [df[prefix + suffix] for suffix in ['delta', 'low', 'high']]
    return df.primary_scale_mad.gt(0) & point.abs().ge(margin * df.primary_scale_mad) & ((low > 0) | (high < 0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--results', type=Path, default=ROOT / 'analysis/regional')
    parser.add_argument('--protocol', type=Path, default=ROOT / 'config/analysis_pipeline/regional.json')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--auxiliary', type=Path, required=True, help='Fresh simulation and original-count summaries')
    args = parser.parse_args(); root = args.root.resolve(); result = args.results.resolve()
    si = args.output_dir / 'tables'; si.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((result / 'run_manifest.json').read_text())
    for name, item in manifest['files'].items():
        if sha256(result / name) != item['sha256']:
            raise ValueError('Corrected output hash mismatch: ' + name)
    protocol = json.loads(args.protocol.read_text())
    sensitivity = read(result / 'map_sensitivity.tsv')
    effects = read(result / 'real_cohort_effects.tsv')
    if len(sensitivity) != 36960 or len(effects) != 16800:
        raise ValueError('Incomplete corrected output grid')
    baseline = effects[(effects.graph_id == GRAPH) & (effects.setting_id == 'baseline')].copy()
    main_maps = sensitivity[(sensitivity.graph_id == GRAPH) & (sensitivity.setting_id == 'baseline') &
                            (sensitivity.readout_id == 'primary_barrier')].copy()
    main_maps['status'] = np.where(main_maps.independent_status.eq('success') & main_maps.historical_status.eq('success'), 'success', 'not_evaluable')
    main_maps['status_reason'] = np.where(main_maps.historical_status.ne('success'), 'historical_eligibility_restriction', main_maps.independent_reason)
    main_maps['diagnostic_status'] = main_maps.status
    main_maps['historical_geometry_status'] = main_maps.historical_status
    flags = baseline[['resource', 'sample_id', 'estimator_id', 'method_id', 'K', 'partition_id', 'is_reference',
                      'anchor_supported', 'additional_support_loss', 'paired_change_flag', 'supported_reversal_vs_anchor',
                      'supported_reversal_vs_reference', 'opposite_point_sign_vs_reference']]
    main_maps = main_maps.merge(flags, on=['resource', 'sample_id', 'estimator_id', 'method_id', 'K', 'partition_id'], validate='one_to_one')
    write(main_maps, si, 'diagnostic_map_diagnostic.tsv')
    write(baseline, si, 'current_per_map_results.tsv')
    write(effects, si, 'corrected_full_effects.tsv')
    from build_jbcb_reader_tables import build_method_comparison
    write(build_method_comparison(sensitivity, baseline), si, 'method_selection_comparison.tsv')
    write(read(result / 'historical_reconciliation.tsv'), si, 'geometry_result_reconciliation.tsv')
    anchors = read(result / 'anchor_section_summary.tsv')
    anchors['anchor_supported'] = anchors.anchor_status.eq('evaluable') & anchors.anchor_conclusion.isin(['near_enrichment', 'near_depletion'])
    anchor_main = anchors[(anchors.graph_id == GRAPH) & (anchors.setting_id == 'baseline')]
    write(anchor_main, si, 'anchor_section_summary.tsv')
    write(anchors, si, 'corrected_anchor_all_settings.tsv')
    summaries = {}
    for frame in protocol['eligibility_frames']:
        for level in ['method_k', 'section', 'patient']:
            data = read(result / f'{frame}_{level}_summary.tsv')
            summaries[(frame, level)] = data
            write(data, si, f'regional_{frame}_{level}_sensitivity.tsv')
    patients = pd.concat([summaries[(f, 'patient')] for f in protocol['eligibility_frames']], ignore_index=True)
    write(patients, si, 'regional_patient_sensitivity.tsv')
    patient_main = patients[(patients.eligibility_frame == 'historical_restricted') & (patients.graph_id == GRAPH)
                            & (patients.setting_id == 'baseline') & (patients.readout_id == 'primary_barrier')].copy()
    patient_main['n_eligible_maps'] = patient_main.n_evaluable_maps
    patient_main['n_eligible_sections'] = patient_main.n_evaluable_sections
    write(patient_main, si, 'diagnostic_patient_summary.tsv')
    section_main = summaries[('historical_restricted', 'section')]
    section_main = section_main[(section_main.graph_id == GRAPH) & (section_main.setting_id == 'baseline') & (section_main.readout_id == 'primary_barrier')].copy()
    section_main['n_eligible_maps'] = section_main.n_evaluable_maps
    write(section_main, si, 'diagnostic_section_summary.tsv')
    write(summaries[('historical_restricted', 'method_k')], si, 'diagnostic_method_k_summary.tsv')
    write(main_maps.groupby(['resource', 'estimator_id', 'status', 'status_reason'], dropna=False).size().reset_index(name='n_maps'), si, 'diagnostic_status_summary.tsv')
    purity = read(result / 'purity_summary.tsv')
    write(purity, si, 'regional_purity_sensitivity.tsv')
    purity_main = purity[(purity.graph_id == GRAPH) & (purity.setting_id == 'baseline') & (purity.eligibility_frame == 'historical_restricted')].copy()
    purity_main['purity_min'] = purity_main.purity_threshold
    purity_main['n_retained_maps'] = purity_main.n_eligible_maps
    # Main figure's purity summaries are over retained map configurations, as in
    # the original diagnostic. Patient summaries remain separately available.
    for index, r in purity_main.iterrows():
        g = main_maps[(main_maps.resource == r.resource) & (main_maps.estimator_id == r.estimator_id)
                      & main_maps.status.eq('success') & main_maps.selected_stroma_purity.ge(r.purity_threshold)]
        for metric in METRICS:
            purity_main.loc[index, metric] = g[metric].median()
    purity_main['median_block_matched_excess_abs_deviation_mad'] = purity_main.block_matched_excess_abs_deviation_mad
    write(purity_main, si, 'diagnostic_purity_sensitivity.tsv')
    resources = []
    for keys, g in patients.groupby(['eligibility_frame', 'resource', 'graph_id', 'estimator_id', 'setting_id', 'readout_id']):
        row = dict(zip(['eligibility_frame', 'resource', 'graph_id', 'estimator_id', 'setting_id', 'readout_id'], keys))
        row.update(n_attempted_patients=len(g), n_evaluable_patients=int(g.n_evaluable_maps.gt(0).sum()),
                   n_standardized_patients=int(g.n_standardized_maps.gt(0).sum()), n_declared_maps=int(g.n_declared_maps.sum()),
                   n_evaluable_maps=int(g.n_evaluable_maps.sum()), n_standardized_maps=int(g.n_standardized_maps.sum()),
                   n_attempted_sections=int(g.n_attempted_sections.sum()), n_evaluable_sections=int(g.n_evaluable_sections.sum()))
        row.update({metric: g[metric].median() for metric in METRICS}); resources.append(row)
    resources = pd.DataFrame(resources)
    write(resources, si, 'regional_resource_summary.tsv')
    write(resources[resources.readout_id.eq('primary_barrier')], si, 'spatial_scale_summary.tsv')
    write(sensitivity, si, 'regional_map_sensitivity.tsv')
    geometry = effects[effects.estimator_id.eq('anchor_retention') & ~effects.is_reference
                       & effects.membership_change_scope.isin(['disjoint_selected_stromal_domains', 'stromal_domain_change_extent_reported'])].copy()
    write(geometry, si, 'domain_repartitioning_summary.tsv')
    thresholds = []
    for margin in [0., .25, .5, .75]:
        for keys, g in effects.groupby(['resource', 'graph_id', 'setting_id', 'estimator_id', 'is_reference']):
            a = support(g, 'morphology_anchor_', margin) & g.anchor_status.eq('evaluable') & g.status.eq('success')
            b = support(g, 'computational_', margin)
            thresholds.append({**dict(zip(['resource', 'graph_id', 'setting_id', 'estimator_id', 'is_reference'], keys)),
                               'map_arm': 'reference' if keys[-1] else 'alternative', 'margin_mad': margin,
                               'anchor_pass': int(a.sum()), 'map_pass': int((a & b).sum()),
                               'map_pass_fraction': (a & b).sum() / a.sum() if a.any() else np.nan})
    write(pd.DataFrame(thresholds), si, 'real_threshold_sensitivity.tsv')
    write(baseline.groupby(['resource', 'estimator_id', 'status']).size().reset_index(name='n_maps'), si, 'setting_status_summary.tsv')
    table1 = []
    for resource, g in main_maps[main_maps.estimator_id.eq('anchor_retention')].groupby('resource'):
        a = anchor_main[anchor_main.resource.eq(resource)]; ok = g.status.eq('success')
        table1.append({'resource': resource, 'attempted_sections': g.sample_id.nunique(), 'attempted_patients': g.patient_id.nunique(),
                       'anchor_eligible_sections': int(a.anchor_status.eq('evaluable').sum()),
                       'anchor_eligible_patients': a.loc[a.anchor_status.eq('evaluable'), 'patient_id'].nunique(),
                       'diagnostic_eligible_sections': g.loc[ok, 'sample_id'].nunique(),
                       'diagnostic_eligible_patients': g.loc[ok, 'patient_id'].nunique(),
                       'declared_primary_maps': len(g), 'eligible_primary_maps': int(ok.sum())})
    counts = pd.DataFrame(table1).set_index('resource')
    labels = {'attempted_sections': 'Attempted sections', 'attempted_patients': 'Attempted patients',
              'anchor_eligible_sections': 'Sections with eligible tissue bands', 'anchor_eligible_patients': 'Patients with eligible tissue bands',
              'diagnostic_eligible_sections': 'Sections with evaluable primary diagnostic maps',
              'diagnostic_eligible_patients': 'Patients with evaluable primary diagnostic maps',
              'declared_primary_maps': 'Declared primary map settings', 'eligible_primary_maps': 'Evaluable primary map settings'}
    counts = counts[['attempted_sections', 'attempted_patients', 'anchor_eligible_sections', 'anchor_eligible_patients',
                     'diagnostic_eligible_sections', 'diagnostic_eligible_patients', 'declared_primary_maps', 'eligible_primary_maps']].T
    counts.index = counts.index.map(labels)
    counts = counts[['Valdeolivas', 'GSE294385']].reset_index(names='Analysis availability')
    write(counts, si, 'current_table1.tsv')
    expressions, cases, graph_membership = [], [], []
    for section in protocol['source_sections']:
        base = result / 'sections' / section['resource'] / section['sample_id']
        spots = unique(read(root / section['source_root'] / 'spot_inputs.tsv.gz'), 'spots')
        vectors = vectors_for(root, section, spots, protocol)
        stroma = spots.coarse_label.eq('stroma').to_numpy()
        geometry_arrays = np.load(base / 'geometry_arrays.npz', allow_pickle=False)
        graph_membership.append({**{k: section[k] for k in ['resource', 'patient_id', 'sample_id', 'section_id']},
            'n_near_members_different': int(np.count_nonzero(geometry_arrays['historical_raw_6nn|baseline|near'] !=
                                                           geometry_arrays['nominal_honeycomb_exact|baseline|near'])),
            'n_far_members_different': int(np.count_nonzero(geometry_arrays['historical_raw_6nn|baseline|far'] !=
                                                          geometry_arrays['nominal_honeycomb_exact|baseline|far']))})
        for graph in protocol['graph_ids']:
            near, far = geometry_arrays[graph + '|baseline|near'], geometry_arrays[graph + '|baseline|far']
            blocks = geometry_arrays[graph + '|baseline|blocks']; components = geometry_arrays[graph + '|components']
            for readout, v in vectors.items():
                scale = float(np.median(np.abs(v[stroma] - np.median(v[stroma])))) if stroma.any() else np.nan
                expressions.append({**{k: section[k] for k in ['resource', 'patient_id', 'sample_id', 'section_id']},
                                    'graph_id': graph, 'readout_id': readout, 'n_stroma': int(stroma.sum()),
                                    'n_near': int(near.sum()), 'n_far': int(far.sum()), 'stroma_mad': scale,
                                    'zero_mad': bool(scale == 0), 'near_median': np.median(v[near]) if near.any() else np.nan,
                                    'far_median': np.median(v[far]) if far.any() else np.nan,
                                    'stroma_nonzero_fraction': float((v[stroma] > 0).mean()) if stroma.any() else np.nan})
            if section['patient_id'] in protocol['case_patients']:
                cases.append({**{k: section[k] for k in ['resource', 'patient_id', 'sample_id', 'section_id']}, 'graph_id': graph,
                              'n_stroma': int(stroma.sum()), 'n_near': int(near.sum()), 'n_far': int(far.sum()),
                              'n_near_blocks': len(np.unique(blocks[near])), 'n_far_blocks': len(np.unique(blocks[far])),
                              'n_near_components': len(np.unique(components[near])), 'n_far_components': len(np.unique(components[far]))})
    write(pd.DataFrame(expressions), si, 'component_expression_summary.tsv')
    write(pd.DataFrame(cases), si, 'case_section_geometry.tsv')
    write(pd.DataFrame(graph_membership), si, 'graph_band_membership_summary.tsv')
    write(read(result / 'case_block_contributions.tsv'), si, 'case_block_contributions.tsv')
    original = args.auxiliary
    for name in ['simulation_summary.tsv', 'simulation_cell_metrics.tsv', 'simulation_calibration_by_map.tsv',
                 'simulation_threshold_sensitivity.tsv', 'component_source_recovery.tsv']:
        if (original / name).resolve() != (si / name).resolve():
            shutil.copy2(original / name, si / name)
    print('Built corrected scientific tables:', si)


if __name__ == '__main__':
    main()
