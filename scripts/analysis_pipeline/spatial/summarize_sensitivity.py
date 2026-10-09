#!/usr/bin/env python3
"""Report complete continuous joint intervals and common-map ring sensitivity."""
from pathlib import Path
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
METRICS = ['abs_deviation_score', 'abs_deviation_mad', 'count_only_excess_abs_deviation_score',
           'block_matched_excess_abs_deviation_score', 'count_only_excess_abs_deviation_mad',
           'block_matched_excess_abs_deviation_mad', 'near_retention', 'far_retention', 'selected_stroma_purity']


def main():
    anchors = pd.read_csv(BASE / 'anchor_joint_sensitivity.tsv', sep='\t', float_precision='round_trip')
    valid = anchors[anchors.status.eq('evaluable')].copy()
    valid['independent_width'] = valid.independent_high - valid.independent_low
    valid['joint_width'] = valid.joint_high - valid.joint_low
    valid['joint_to_independent_width_ratio'] = valid.joint_width / valid.independent_width
    valid['lower_endpoint_change'] = valid.joint_low - valid.independent_low
    valid['upper_endpoint_change'] = valid.joint_high - valid.independent_high
    valid.to_csv(BASE / 'joint_interval_continuous_comparison.tsv', sep='\t', index=False)
    old_valid = valid[valid.setting_id.ne('near175')]
    stats = {'original_attempted_anchors': int(anchors.setting_id.ne('near175').sum()),
             'original_eligible_anchors': len(old_valid),
             'original_eligible_distinct_sections': old_valid.sample_id.nunique(),
             'joint_used': int(old_valid.joint_used.sum()), 'joint_attempted': int(old_valid.joint_attempted.sum()),
             'joint_empty_band_rejections': int(old_valid.joint_rejected_empty_band.sum()),
             'maximum_empty_band_rejections_per_anchor': int(old_valid.joint_rejected_empty_band.max()),
             'median_joint_to_independent_width_ratio': float(old_valid.joint_to_independent_width_ratio.median()),
             'min_joint_to_independent_width_ratio': float(old_valid.joint_to_independent_width_ratio.min()),
             'max_joint_to_independent_width_ratio': float(old_valid.joint_to_independent_width_ratio.max()),
             'near175_attempted_anchors': int(anchors.setting_id.eq('near175').sum()),
             'near175_eligible_anchors': int(valid.setting_id.eq('near175').sum())}
    (BASE / 'joint_interval_summary.json').write_text(json.dumps(stats, indent=2) + '\n')
    old = pd.read_csv(ROOT / P['output_dir'] / 'map_sensitivity.tsv', sep='\t', float_precision='round_trip')
    old = old[old.readout_id.eq('primary_barrier') & old.setting_id.eq('baseline')]
    new = pd.read_csv(BASE / 'near175_map_sensitivity.tsv', sep='\t', float_precision='round_trip')
    keys = ['resource', 'sample_id', 'graph_id', 'estimator_id', 'method_id', 'K', 'partition_id']
    paired = old.merge(new, on=keys, suffixes=('_100', '_175'), validate='one_to_one')
    paired['common_eligible'] = paired.independent_status_100.eq('success') & paired.independent_status_175.eq('success')
    paired.to_csv(BASE / 'near100_near175_all_map_comparison.tsv', sep='\t', index=False)
    rows = []
    for key, group in paired.groupby(['resource', 'graph_id', 'estimator_id']):
        common = group[group.common_eligible]
        for ring in ['100', '175']:
            columns = [x + '_' + ring for x in METRICS]
            patient_key = 'patient_id_' + ring
            cells = common.groupby([patient_key, 'sample_id', 'method_id', 'K'])[columns].median()
            sections = cells.groupby([patient_key, 'sample_id']).median()
            patients = sections.groupby(patient_key).median()
            row = dict(zip(['resource', 'graph_id', 'estimator_id'], key))
            row.update(near_limit_um=int(ring), n_declared_maps=len(group), n_common_eligible_maps=len(common),
                       n_common_eligible_sections=common.sample_id.nunique(), n_common_eligible_patients=common[patient_key].nunique())
            row.update({metric: float(patients[metric + '_' + ring].median()) for metric in METRICS})
            rows.append(row)
    pd.DataFrame(rows).to_csv(BASE / 'near100_near175_common_map_resource_summary.tsv', sep='\t', index=False)
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
