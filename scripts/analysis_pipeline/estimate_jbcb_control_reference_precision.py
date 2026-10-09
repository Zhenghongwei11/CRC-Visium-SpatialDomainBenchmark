#!/usr/bin/env python3
"""Finite-draw precision of the fixed matched-subset references.

Reads the supplied corrected-statistics workspace and creates a new result
directory. Bounds concern Monte Carlo reference medians conditional on fixed
scores, bands and maps; they do not estimate patient-population uncertainty.
"""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import hashlib
import json

import numpy as np
import pandas as pd
import scipy
from scipy.stats import binom


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def run(workspace, output, resume=False):
    protocol = json.loads((workspace / 'config/analysis_pipeline/regional.json').read_text())
    base = workspace / protocol['output_dir']
    source = base / 'map_sensitivity.tsv'
    table = pd.read_csv(source, sep='\t', float_precision='round_trip', low_memory=False)
    baseline = table[(table.graph_id == 'historical_raw_6nn') & (table.setting_id == 'baseline') &
                     (table.estimator_id == 'anchor_retention') & (table.readout_id == 'primary_barrier')]
    selected = baseline[baseline.independent_status == 'success'].copy()
    identity = ['resource', 'sample_id', 'method_id', 'K', 'partition_id']
    if len(baseline) != 840 or selected.empty or selected.duplicated(identity).any():
        raise ValueError('Expected the complete configured baseline and unique eligible selections')
    if not protocol.get('source_fit_manifest_sha256') and len(selected) != 407:
        raise ValueError('Saved publication fits must reproduce their 407 primary selections')
    if not (selected.stroma_mad > 0).all():
        raise ValueError('Primary MAD must be positive')
    family_n, n = len(selected) * 2, 1000
    alpha = .05 / family_n
    k = int(binom.ppf(alpha / 2, n, .5))
    if not binom.cdf(k - 1, n, .5) <= alpha / 2 < binom.cdf(k, n, .5):
        raise ValueError('Incorrect binomial tail index')
    output.mkdir(parents=True, exist_ok=resume)
    inputs = [{'workspace_path': str(source.relative_to(workspace)), 'sha256': digest(source)}]
    records = []
    for (resource, sample), group in selected.groupby(['resource', 'sample_id']):
        path = base / 'sections' / resource / sample / 'control_draws.npz'
        inputs.append({'workspace_path': str(path.relative_to(workspace)), 'sha256': digest(path)})
        with np.load(path) as arrays:
            for row in group.to_dict('records'):
                key = '|'.join(str(row[c]) for c in ['graph_id', 'setting_id', 'estimator_id', 'method_id', 'K', 'partition_id', 'readout_id'])
                observed = abs(row['selected_delta_score'] - row['anchor_delta_score'])
                scale = row['stroma_mad']
                rec = {c: row[c] for c in ['resource', 'patient_id', 'sample_id', 'method_id', 'K', 'partition_id']}
                for kind in ['count_only', 'block_matched']:
                    values = arrays[key + '|' + kind]
                    if len(values) != n or not np.isfinite(values).all() or row[kind + '_control_used'] != n:
                        raise ValueError('A reference does not contain 1,000 finite draws')
                    absolute = np.sort(np.abs(values - row['anchor_delta_score']))
                    median = float(np.median(absolute))
                    if not np.isclose(median, row[kind + '_control_abs_deviation_median_score'], rtol=1e-12, atol=1e-12):
                        raise ValueError('Reference does not reproduce the frozen median')
                    low, high = float(absolute[k - 1]), float(absolute[n - k])
                    rec[kind + '_median_reference'] = median / scale
                    rec[kind + '_reference_mc_low'] = low / scale
                    rec[kind + '_reference_mc_high'] = high / scale
                    rec[kind + '_excess'] = (observed - median) / scale
                    rec[kind + '_excess_mc_low'] = (observed - high) / scale
                    rec[kind + '_excess_mc_high'] = (observed - low) / scale
                records.append(rec)
    maps = pd.DataFrame(records)
    columns = [f'{kind}_excess{suffix}' for kind in ['count_only', 'block_matched'] for suffix in ['', '_mc_low', '_mc_high']]
    settings = maps.groupby(['resource', 'patient_id', 'sample_id', 'method_id', 'K'])[columns].median()
    sections = settings.groupby(['resource', 'patient_id', 'sample_id'])[columns].median()
    patients = sections.groupby(['resource', 'patient_id'])[columns].median()
    resources = patients.groupby('resource')[columns].median()
    for data, name in [(patients, 'patient_gap'), (resources, 'resource_median_gap')]:
        data[name] = data.count_only_excess - data.block_matched_excess
        data['gap_mc_low'] = data.count_only_excess_mc_low - data.block_matched_excess_mc_high
        data['gap_mc_high'] = data.count_only_excess_mc_high - data.block_matched_excess_mc_low
    maps.to_csv(output / 'control_reference_map_precision.tsv', sep='\t', index=False)
    patients.to_csv(output / 'control_reference_patient_precision.tsv', sep='\t')
    resources.to_csv(output / 'control_reference_resource_precision.tsv', sep='\t')
    record = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'analysis': 'retrospective finite-draw precision conditional on fixed tissue, readout and maps',
        'primary_configurations': len(selected), 'reference_medians': family_n,
        'draws_per_reference': n, 'joint_nominal_coverage': .95,
        'single_reference_alpha': alpha, 'one_based_order_positions': [k, n - k + 1],
        'joint_coverage_lower_bound': float(1 - family_n * 2 * binom.cdf(k - 1, n, .5)),
        'replication': 'partition to method/K to section to patient; resource median over patients',
        'source_sha256': digest(Path(__file__).resolve()), 'inputs': inputs,
        'software': {'numpy': np.__version__, 'pandas': pd.__version__, 'scipy': scipy.__version__},
    }
    (output / 'analysis_record.json').write_text(json.dumps(record, indent=2) + '\n')
    print(resources[['resource_median_gap', 'gap_mc_low', 'gap_mc_high']].to_string())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--output', '--results', dest='results', type=Path, required=True, help='New output directory')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    run(args.workspace.resolve(), args.results.resolve(), args.resume)


if __name__ == '__main__':
    main()
