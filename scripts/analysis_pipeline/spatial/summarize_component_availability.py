#!/usr/bin/env python3
"""Attach map and patient availability to every baseline component summary."""
from pathlib import Path
import pandas as pd

from configuration import parse_configuration
C = parse_configuration(__doc__)
ROOT = C.workspace
BASE = C.results
PROTOCOL = C.protocol



def read(name):
    return pd.read_csv(BASE / name, sep='\t', float_precision='round_trip')


def main():
    values = read('main_baseline_components_resource_summary.tsv')
    counts = read('main_baseline_component_availability.tsv')
    patients = read('figure_main_patient_summary.tsv')
    patients = patients[patients.graph_id.eq('historical_raw_6nn') & patients.setting_id.eq('baseline')].copy()
    for kind, metric in [('abs', 'abs_deviation_mad'), ('count', 'count_only_excess_abs_deviation_mad'),
                         ('block', 'block_matched_excess_abs_deviation_mad')]:
        patients['n_finite_' + kind + '_std'] = patients[metric].notna()
    keys = ['resource', 'estimator_id', 'readout_id']
    available = patients.groupby(keys)[['n_finite_abs_std', 'n_finite_count_std', 'n_finite_block_std']].sum().reset_index()
    result = values.merge(counts, on=keys, validate='one_to_one').merge(available, on=keys, validate='one_to_one')
    assert len(result) == 28
    result.to_csv(BASE / 'main_baseline_components_with_availability.tsv', sep='\t', index=False)


if __name__ == '__main__':
    main()
