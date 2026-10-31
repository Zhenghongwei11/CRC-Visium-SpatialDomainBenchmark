"""Fixed random seeds for the published conditional-reference analyses."""
import csv
import gzip
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'data/reference_inputs/random_seeds'


@lru_cache(maxsize=1)
def residual_seed_vectors():
    with gzip.open(ROOT / 'residual_seed_vectors.tsv.gz', 'rt', newline='') as handle:
        rows = csv.DictReader(handle, delimiter='\t')
        return {(r['sample_id'], r['case_id'], r['control'], r['band']):
                tuple(map(int, r['seeds'].split(','))) for r in rows}


@lru_cache(maxsize=1)
def conditional_seeds():
    with gzip.open(ROOT / 'conditional_chain_seeds.tsv.gz', 'rt', newline='') as handle:
        rows = csv.DictReader(handle, delimiter='\t')
        return {(r['sample_id'], r['case_id'], r['control'], r['band'], int(r['chain'])):
                int(r['seed']) for r in rows}
