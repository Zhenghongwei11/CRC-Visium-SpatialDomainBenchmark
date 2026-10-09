#!/usr/bin/env python3
"""Describe MYH11, CNN1 and DES in the unchanged annotated stromal bands."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.io import mmread
from scipy.sparse import csc_matrix
from threadpoolctl import threadpool_limits

MARKERS = ('MYH11', 'CNN1', 'DES')
READOUT = ('TGFB1', 'CXCL12', 'ACTA2', 'TAGLN')
GRAPH = 'historical_raw_6nn'


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read(path):
    return pd.read_csv(path, sep='\t', float_precision='round_trip')


def section_job(section, count_root, workspace):
    resource, sample = section['resource'], section['sample_id']
    folder = Path(count_root) / resource / sample
    source_files = {}
    for expected in section['source_files']:
        candidates = list(folder.rglob(expected['name']))
        if len(candidates) != 1:
            raise ValueError(f'Expected one original file: {sample}/{expected["name"]}')
        path = candidates[0]
        if path.stat().st_size != expected['size_bytes'] or digest(path) != expected['sha256']:
            raise ValueError(f'Original source file differs: {path}')
        source_files[path.name] = path
    h5name = sample + '_filtered_feature_bc_matrix.h5'
    with threadpool_limits(limits=1):
        if h5name in source_files:
            with h5py.File(source_files[h5name]) as handle:
                matrix = handle['matrix']
                counts = csc_matrix((matrix['data'][:], matrix['indices'][:], matrix['indptr'][:]),
                                    shape=tuple(matrix['shape'][:])).astype(np.float64)
                barcodes = [x.decode() for x in matrix['barcodes'][:]]
                symbols = [x.decode() for x in matrix['features/name'][:]]
        else:
            with gzip.open(source_files[sample + '_matrix.mtx.gz'], 'rb') as handle:
                counts = mmread(handle).tocsc().astype(np.float64)
            with gzip.open(source_files[sample + '_barcodes.tsv.gz'], 'rt') as handle:
                barcodes = [line.strip() for line in handle if line.strip()]
            features = pd.read_csv(source_files[sample + '_features.tsv.gz'], sep='\t', header=None)
            symbols = features.iloc[:, 1].astype(str).tolist()
    if counts.shape != (section['feature_count'], section['matrix_barcode_count']):
        raise ValueError('Original matrix dimensions differ')
    if len(barcodes) != len(set(barcodes)) or not np.isfinite(counts.data).all() or (counts.data < 0).any():
        raise ValueError('Invalid original counts or barcodes')
    spots = read(Path(workspace) / section['source_root'] / 'spot_inputs.tsv.gz')
    saved = read(Path(workspace) / section['source_root'] / 'spot_scores.tsv.gz')
    saved = saved.loc[saved.score_id.eq('primary_barrier')].copy()
    if spots.barcode.duplicated().any() or saved.barcode.duplicated().any():
        raise ValueError('Duplicate registered barcode')
    index = pd.Index(barcodes).get_indexer(spots.barcode)
    if (index < 0).any():
        raise ValueError('Registered spot absent from original count matrix')
    totals = np.asarray(counts.sum(axis=0)).ravel()[index]
    values = {}
    for gene in MARKERS + READOUT:
        locations = [i for i, symbol in enumerate(symbols) if symbol == gene]
        if len(locations) != 1:
            raise ValueError(f'{gene} must have one original source feature in {sample}')
        raw = counts[locations[0], :].toarray().ravel()[index]
        values[gene] = np.log1p(np.divide(raw * 10000., totals, out=np.zeros_like(raw), where=totals > 0))
    frozen = saved.set_index('barcode').loc[spots.barcode, 'score_value'].to_numpy()
    recomposed = np.mean([values[g] for g in READOUT], axis=0)
    if not np.allclose(recomposed, frozen, rtol=2e-6, atol=2e-7):
        raise ValueError('Original full-feature normalization differs from the published readout')
    geometry_path = Path(workspace) / json.loads((Path(workspace) / 'config/analysis_pipeline/regional.json').read_text())['output_dir'] / 'sections' / resource / sample / 'geometry_arrays.npz'
    with np.load(geometry_path, allow_pickle=False) as geometry:
        near = geometry[GRAPH + '|baseline|near'].astype(bool)
        far = geometry[GRAPH + '|baseline|far'].astype(bool)
        blocks = geometry[GRAPH + '|baseline|blocks'].copy()
    if len(near) != len(spots) or len(far) != len(spots) or (near & far).any():
        raise ValueError('Incorrect registered band arrays')
    stroma = spots.coarse_label.eq('stroma').to_numpy()
    if ((near | far) & ~stroma).any():
        raise ValueError('A measured band contains a nonstromal source label')
    frame = spots[['barcode', 'coarse_label', 'x_fullres', 'y_fullres', 'array_row', 'array_col']].copy()
    frame.insert(0, 'sample_id', sample); frame.insert(0, 'patient_id', section['patient_id']); frame.insert(0, 'resource', resource)
    frame['near_band'] = near; frame['far_band'] = far; frame['spatial_block'] = blocks
    for gene in MARKERS:
        frame[gene + '_log_cp10k'] = values[gene]
    summary, block_rows = [], []
    for gene in MARKERS:
        vector = values[gene]
        record = dict(resource=resource, patient_id=section['patient_id'], sample_id=sample, gene=gene)
        for name, mask in [('stroma', stroma), ('near', near), ('far', far)]:
            selected = vector[mask]
            record['n_' + name] = len(selected)
            record[name + '_detected_fraction'] = float(np.mean(selected > 0)) if len(selected) else np.nan
            record[name + '_median'] = float(np.median(selected)) if len(selected) else np.nan
            record[name + '_q25'] = float(np.quantile(selected, .25)) if len(selected) else np.nan
            record[name + '_q75'] = float(np.quantile(selected, .75)) if len(selected) else np.nan
        record['near_minus_far'] = record['near_median'] - record['far_median']
        record['bands_meet_original_requirements'] = bool(min(near.sum(), far.sum()) >= 20 and min(len(np.unique(blocks[near])), len(np.unique(blocks[far]))) >= 3)
        summary.append(record)
        for band, mask in [('near', near), ('far', far)]:
            for block in np.unique(blocks[mask]):
                selected = vector[mask & (blocks == block)]
                block_rows.append(dict(resource=resource, patient_id=section['patient_id'], sample_id=sample,
                                       gene=gene, band=band, spatial_block=str(block), n_spots=len(selected),
                                       detected_fraction=float(np.mean(selected > 0)), median=float(np.median(selected))))
    return frame, summary, block_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['workspace', 'count-root', 'source-manifest', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    sections = json.loads(args.source_manifest.read_text())['sections']
    if len(sections) != 22:
        raise ValueError('All original 22 sections are required')
    args.output.mkdir(parents=True, exist_ok=True)
    frames, rows, block_rows = [], [], []
    with ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(section_job, section, args.count_root, args.workspace) for section in sections]
        for future in as_completed(futures):
            frame, summary, per_block = future.result()
            frames.append(frame); rows.extend(summary); block_rows.extend(per_block)
            print(f'{frame.resource.iloc[0]} {frame.sample_id.iloc[0]}: {len(frames)}/22 sections', flush=True)
    pd.DataFrame(rows).sort_values(['resource', 'sample_id', 'gene']).to_csv(args.output / 'contractile_marker_section_summary.tsv', sep='\t', index=False)
    pd.DataFrame(block_rows).sort_values(['resource', 'sample_id', 'gene', 'band', 'spatial_block']).to_csv(args.output / 'contractile_marker_block_summary.tsv', sep='\t', index=False)
    pd.concat(frames).sort_values(['resource', 'sample_id', 'barcode']).to_csv(args.output / 'contractile_marker_spots.tsv.gz', sep='\t', index=False, compression={'method': 'gzip', 'mtime': 0})
    record={'source_manifest_sha256':digest(args.source_manifest),'script_sha256':digest(__file__),
            'files':{p.name:{'sha256':digest(p),'bytes':p.stat().st_size} for p in args.output.glob('*.tsv*')}}
    (args.output/'analysis_record.json').write_text(json.dumps(record,indent=2)+'\n')
    print('Completed fixed-marker descriptions for all 22 sections.', flush=True)


if __name__ == '__main__':
    main()
