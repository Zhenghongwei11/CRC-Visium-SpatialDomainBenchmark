#!/usr/bin/env python3
"""Compare archived and refitted partitions on the same registered barcodes."""
from pathlib import Path
import argparse, sys
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/analysis_pipeline'))
from regional_helpers import domain_selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample', help='Optional one-section comparison')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    grid = pd.read_csv(ROOT / 'expected/refit_sensitivity/map_comparisons.tsv', sep='\t')
    if args.sample:
        grid = grid[grid['sample'].eq(args.sample)]
    if grid.empty:
        raise ValueError('No matching section')
    results = []
    for (resource, sample), rows in grid.groupby(['resource', 'sample'], sort=True):
        base = ROOT / 'data/reference_inputs/sections' / resource / sample
        spots = pd.read_csv(base / 'spot_inputs.tsv.gz', sep='\t').set_index('barcode')
        if spots.index.duplicated().any():
            raise ValueError('Duplicate registered barcode: ' + sample)
        old = pd.read_csv(base / 'input_maps.tsv.gz', sep='\t')
        if 'status' in old:
            old = old[old['status'].eq('success')]
        new = pd.read_csv(ROOT / 'data/refit_domains' / resource / (sample + '.tsv.gz'), sep='\t')
        old_groups = dict(tuple(old.groupby(['method_id', 'K', 'replicate_id'], sort=False)))
        new_groups = dict(tuple(new.groupby(['method_id', 'K', 'replicate_id'], sort=False)))
        coarse = spots['coarse_label'].to_numpy(str)
        for row in rows.itertuples(index=False):
            key = (row.method, int(row.K), row.replicate)
            groups = [old_groups.get(key), new_groups.get(key)]
            labels = []
            for group in groups:
                if group is None:
                    labels.append(None)
                    continue
                if group.barcode.duplicated().any() or set(group.barcode) != set(spots.index):
                    raise ValueError(f'Barcode set mismatch: {resource}/{sample}/{key}')
                labels.append(group.set_index('barcode').loc[spots.index, 'domain_label'].to_numpy(int))
            if labels[0] is None and labels[1] is None:
                status = 'both_maps_absent'
            elif labels[0] is None:
                status = 'old_map_absent'
            elif labels[1] is None:
                status = 'new_map_absent'
            else:
                status = 'matched'
            ari = None
            jaccard = None
            if status == 'matched':
                ari = float(adjusted_rand_score(*labels))
                a, _ = domain_selection(labels[0], coarse)
                b, _ = domain_selection(labels[1], coarse)
                if a['status'] == 'success' and b['status'] == 'success':
                    left = labels[0] == a['stroma_domain']
                    right = labels[1] == b['stroma_domain']
                    jaccard = float(np.count_nonzero(left & right) / np.count_nonzero(left | right))
            results.append(dict(resource=resource, patient_id=row.patient_id, sample=sample,
                                method=key[0], K=key[1], replicate=key[2], status=status,
                                ARI=ari, domain_jaccard=jaccard))
        print(resource, sample, 'completed', flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(args.output, sep='\t', index=False, na_rep='NA')


if __name__ == '__main__':
    main()
