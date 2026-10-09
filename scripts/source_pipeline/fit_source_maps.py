#!/usr/bin/env python3
"""Fit the study's configured domain maps from original registered counts."""
from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

import scientific_functions as science
import official_methods as official
from fixed_scores import read_counts, fixed_scores
from source_inputs import digest

METHODS = tuple(science.METHOD_LABELS)


def source_sample(flat, annotation, row, loader):
    """Prepare the source normalization/PCA and a separately normalized outcome."""
    counts, barcodes, genes, members = read_counts(flat, row['sample_id'])
    positions = science.read_positions(next(p for p in [
        flat / (row['sample_id'] + '_tissue_positions_list.csv.gz'),
        flat / (row['sample_id'] + '_tissue_positions.csv.gz')] if p.exists()))
    if len(set(barcodes)) != len(barcodes) or positions.barcode.duplicated().any():
        raise ValueError('Duplicate matrix or position barcode')
    if not set(barcodes) <= set(positions.barcode):
        raise ValueError('A matrix barcode lacks original coordinates')
    positions = positions.set_index('barcode').loc[barcodes].reset_index()
    keep = positions.in_tissue.astype(int).eq(1).to_numpy()
    positions = positions.loc[keep].reset_index(drop=True)
    selected = positions.barcode.astype(str).tolist()
    original_annotation = science.read_annotation(annotation)
    if original_annotation.barcode.duplicated().any() or not set(original_annotation.barcode) <= set(barcodes):
        raise ValueError('Original annotation barcode mismatch')
    annotations = original_annotation.set_index('barcode').raw_label
    raw_labels = np.array([str(annotations.get(b, '')) for b in selected], object)
    scalepath = next(p for p in [flat / (row['sample_id'] + '_scalefactors_json.json.gz'),
                               flat / (row['sample_id'] + '_scalefactors.json.gz')] if p.exists())
    diameter = float(science.read_scalefactors(scalepath)['spot_diameter_fullres'])
    if not np.isfinite(diameter) or diameter <= 0:
        raise ValueError('Invalid original spot diameter')
    # Fit representation follows the producer's float32 CP10K/log1p arithmetic.
    # Outcome recovery below follows the separately recorded float64 contract.
    x = counts[keep, :].astype(np.float32)
    totals = np.asarray(x.sum(axis=1)).ravel()
    denominator = np.where(totals == 0, 1, totals)
    x = x.multiply((np.float32(10000) / denominator)[:, None]).tocsr()
    x.data = np.log1p(x.data)
    means = np.asarray(x.mean(axis=0)).ravel()
    sq_means = np.asarray(x.power(2).mean(axis=0)).ravel()
    variance = np.maximum(sq_means - means ** 2, 0)
    feature_idx = np.argsort(variance)[::-1][:min(2000, max(2, x.shape[1] - 1))]
    dense = x[:, feature_idx].toarray().astype(np.float32)
    pcs = PCA(n_components=max(2, min(20, dense.shape[0] - 1, dense.shape[1] - 1)),
              random_state=0).fit_transform(dense).astype(np.float32)
    sample = {'sample_id': row['sample_id'], 'patient_id': row['patient_id'],
              'section_id': row['section_id'], 'barcodes': np.array(selected),
              'coords': positions[['pxl_col_in_fullres', 'pxl_row_in_fullres']].to_numpy(float),
              'array_coords': positions[['array_row', 'array_col']].to_numpy(float),
              'raw_labels': raw_labels,
              'coarse_labels': np.array([science.map_label(v) for v in raw_labels], object),
              'pcs': pcs, 'spot_diameter_fullres': diameter,
              'microns_per_pixel': 55.0 / diameter}
    if loader == 'scanpy':
        original = science.load_expression(flat, row['sample_id'], annotation)
        if not np.array_equal(original['barcodes'], sample['barcodes']):
            raise ValueError('Native and original producer registration differ')
        sample['pcs'] = original['pcs']
    scores = fixed_scores(counts, barcodes, genes, row['dataset_id'], row['sample_id'])
    scores = scores.set_index('barcode').loc[selected].reset_index()
    normalized = counts[keep, :].astype(np.float64)
    totals64 = np.asarray(normalized.sum(axis=1)).ravel()
    normalized = normalized.multiply(np.divide(10000., totals64, out=np.zeros_like(totals64),
                                               where=totals64 > 0)[:, None]).tocsr()
    normalized.data = np.log1p(normalized.data)
    gene_index = {g: i for i, g in enumerate(genes)}
    components = pd.DataFrame({'barcode': selected})
    for gene in science.PRIMARY_GENES:
        components[gene] = normalized[:, gene_index[gene]].toarray().ravel()
    spots = pd.DataFrame({'dataset_id': row['dataset_id'], 'sample_id': row['sample_id'],
                          'patient_id': row['patient_id'], 'barcode': selected,
                          'raw_label': raw_labels, 'coarse_label': sample['coarse_labels'],
                          'x_fullres': sample['coords'][:, 0], 'y_fullres': sample['coords'][:, 1],
                          'array_row': sample['array_coords'][:, 0], 'array_col': sample['array_coords'][:, 1],
                          'spot_diameter_fullres': diameter})
    features = pd.DataFrame({'source_feature_index': feature_idx,
                             'feature_name': np.asarray(genes)[feature_idx], 'variance': variance[feature_idx]})
    return sample, spots, scores, components, features


def environment(methods, loader, rscript):
    required = ['igraph', 'leidenalg'] if 'M3_spatial_leiden' in methods else []
    if loader == 'scanpy' or set(methods) & {'M4_spagcn_official', 'M5_stagate_official'}:
        required += ['scanpy', 'anndata']
    if 'M4_spagcn_official' in methods:
        required += ['SpaGCN', 'torch', 'cv2']
    if 'M5_stagate_official' in methods:
        required += ['STAGATE_pyG', 'torch', 'torch_geometric', 'torch_sparse', 'torch_scatter']
    missing = [m for m in set(required) if importlib.util.find_spec(m) is None]
    if missing:
        raise RuntimeError('Install the documented scientific environment first; missing: ' + ', '.join(sorted(missing)))
    if 'M6_bayesspace_official' in methods:
        subprocess.run([rscript, '-e', 'invisible(lapply(c("Matrix","SingleCellExperiment","BayesSpace","mclust","hdf5r"),function(x) library(x,character.only=TRUE)))'], check=True)
    versions = {'python': sys.version.split()[0]}
    for package in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'h5py', 'igraph', 'leidenalg',
                    'scanpy', 'anndata', 'torch', 'torch-geometric', 'torch-sparse', 'torch-scatter', 'SpaGCN']:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    if 'M6_bayesspace_official' in methods:
        rversions=subprocess.run([rscript,'-e','cat(paste(c(paste0("R=",getRversion()),sapply(c("Matrix","SingleCellExperiment","BayesSpace","mclust","hdf5r"),function(x) paste0(x,"=",packageVersion(x)))),collapse="\\n"))'],check=True,capture_output=True,text=True)
        versions.update(dict(line.split('=',1) for line in rversions.stdout.splitlines()))
    return versions


def fit_baselines(sample, methods, rows):
    from sklearn.cluster import KMeans, AgglomerativeClustering
    from sklearn.neighbors import NearestNeighbors
    pcs, coords = sample['pcs'], sample['coords']
    spatial = np.concatenate([pcs, .5 * ((coords - coords.mean(axis=0)) / (coords.std(axis=0) + 1e-6))], axis=1)
    frames = []
    for method, features in [('M0_expr_kmeans', pcs), ('M1_spatial_concat_kmeans', spatial)]:
        if method not in methods:
            continue
        for k in [4, 6]:
            for seed in [11, 23, 37]:
                labels = KMeans(n_clusters=k, n_init=50, random_state=seed).fit_predict(features)
                frames.append(science.make_map(sample, labels, method, k, f'seed_{seed}', 'seed', 'fixed-K baseline'))
    if 'M2_spatial_ward' in methods:
        for k in [4, 6]:
            for n in [4, 6, 8]:
                nearest = NearestNeighbors(n_neighbors=min(len(coords), n + 1)).fit(coords)
                graph = nearest.kneighbors_graph(coords, mode='connectivity')
                labels = AgglomerativeClustering(n_clusters=k, linkage='ward', connectivity=graph.maximum(graph.T)).fit_predict(pcs)
                frames.append(science.make_map(sample, labels, 'M2_spatial_ward', k, f'neighbors_{n}', 'graph_neighbors', 'fixed-K spatial Ward graph specification'))
    if 'M3_spatial_leiden' in methods:
        graph = science.weighted_leiden_graph(pcs, coords, neighbors=6)
        for k in [4, 6]:
            for seed in [11, 23, 37]:
                labels, resolution, exact = science.exact_leiden(graph, k, seed)
                rows.append({'method': 'M3_spatial_leiden', 'K': k, 'seed': seed,
                             'status': 'success' if exact else 'not_evaluable',
                             'reason': '' if exact else 'no_exact_K_resolution'})
                if exact:
                    frames.append(science.make_map(sample, labels, 'M3_spatial_leiden', k, f'seed_{seed}', 'seed', f'fixed-K spatial Leiden six-neighbor graph;resolution={resolution:.8g}'))
    return frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New map-fitting output directory')
    parser.add_argument('--methods', default='all', help='all, baseline, or comma-separated original method IDs')
    parser.add_argument('--samples', default='')
    parser.add_argument('--loader', choices=['scanpy', 'native'], default='scanpy',
                        help='scanpy uses the original producer loader; native uses explicit float32 CP10K/PCA arithmetic')
    parser.add_argument('--rscript', default='Rscript')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--resume', action='store_true', help='Continue the same fits from complete section outputs')
    args = parser.parse_args()
    if args.output.exists() and not args.resume:
        raise FileExistsError('Select a new output directory for the new fits')
    methods = METHODS if args.methods == 'all' else METHODS[:4] if args.methods == 'baseline' else tuple(args.methods.split(','))
    if not methods or set(methods) - set(METHODS):
        raise ValueError('Unknown or empty method list')
    versions = environment(methods, args.loader, args.rscript)
    rows = pd.read_csv(args.inputs / 'sample_manifest.tsv', sep='\t').to_dict('records')
    requested = set(args.samples.split(',')) if args.samples else set()
    rows = [r for r in rows if not requested or r['sample_id'] in requested]
    if not rows or requested - {r['sample_id'] for r in rows}:
        raise ValueError('Unknown sample identifiers')
    identity = json.loads((args.inputs / 'source_identity.json').read_text())
    for name, expected in identity['extracted_files'].items():
        p = args.inputs / name
        if digest(p) != expected['sha256'] or p.stat().st_size != expected['bytes']:
            raise ValueError('Registered original input changed: ' + name)
    job_identity={'input_sha256':digest(args.inputs/'source_identity.json'), 'sources':rows,
                  'methods':list(methods), 'loader':args.loader, 'software':versions,
                  'code':{p.name:digest(p) for p in [Path(__file__),Path(science.__file__),Path(official.__file__),
                                                   Path(__file__).with_name('fixed_scores.py'),Path(__file__).with_name('bayesspace.R')]}}
    started=args.output/'started_inputs.json'
    if args.output.exists() and (not started.exists() or json.loads(started.read_text())!=job_identity):
        raise ValueError('Source inputs, code or fitting environment changed; use a new output directory')
    args.output.mkdir(parents=True,exist_ok=args.resume)
    started.write_text(json.dumps(job_identity,indent=2)+'\n')
    attempts = []
    for row in rows:
        sample = row['sample_id']
        section = args.output / 'sections' / row['resource'] / sample
        checkpoint=section/'manifest.json'
        if args.resume and checkpoint.is_file():
            saved=json.loads(checkpoint.read_text())
            for name,r in saved['files'].items():
                p=section/name
                if digest(p)!=r['sha256'] or p.stat().st_size!=r['bytes']:
                    raise ValueError('Completed source fit changed: '+sample+'/'+name)
            attempts.extend(saved['attempts'])
            print('Resumed complete section '+sample,flush=True)
            continue
        section_attempts=[]
        science.DATASET_ID = row['dataset_id']
        science.SAMPLE_META = {sample: row}
        flat = args.inputs / row['bundle_root']
        with threadpool_limits(limits=args.threads):
            representation, spots, scores, components, features = source_sample(flat, args.inputs / row['annotation_file'], row, args.loader)
            frames = fit_baselines(representation, methods, section_attempts)
        section.mkdir(parents=True,exist_ok=args.resume)
        if 'M4_spagcn_official' in methods:
            official.run_spagcn(row['dataset_id'], sample, flat, frames, section_attempts)
        if 'M5_stagate_official' in methods:
            official.run_stagate(row['dataset_id'], sample, flat, frames, section_attempts)
        if 'M6_bayesspace_official' in methods and row['resource'] == 'GSE294385':
            rmap, rbench = section / 'bayesspace_maps.tsv', section / 'bayesspace_benchmark.tsv'
            subprocess.run([args.rscript, str(Path(__file__).with_name('bayesspace.R')),
                            '--dataset-id', row['dataset_id'], '--dataset-root', str(flat),
                            '--sample-id', sample, '--k-grid', '4,6', '--seeds', '11,23,37',
                            '--nrep', '1000', '--gamma', '3', '--output-domain-map-tsv', str(rmap),
                            '--output-tsv', str(rbench), '--note', 'configured-source-recomputation'], check=True)
            bayes = pd.read_csv(rmap, sep='\t')
            bayes['method_id'] = 'M6_bayesspace_official'
            bayes['replicate_id'] = bayes.seed.map(lambda seed: 'seed_' + str(int(seed)))
            bayes['replicate_type'] = 'seed'
            bayes['status'] = 'success'
            frames.append(bayes)
        maps = pd.concat(frames, ignore_index=True)
        maps.method_id = maps.method_id.replace({'Official_SpaGCN_v1_2_7': 'M4_spagcn_official',
                                                 'Official_STAGATE_pyG': 'M5_stagate_official'})
        for frame, name in [(spots, 'spot_inputs.tsv.gz'), (scores, 'spot_scores.tsv.gz'),
                            (components, 'spot_components.tsv.gz'), (maps, 'input_maps.tsv.gz'),
                            (features, 'fitting_features.tsv.gz')]:
            frame.to_csv(section / name, sep='\t', index=False, compression='gzip')
        np.save(section / 'fitting_pca.npy', representation['pcs'])
        record = {'schema_version': 'new_source_fit_section_v1', 'source': row,
                  'methods_requested': list(methods), 'loader': args.loader,
                  'attempts':section_attempts,
                  'n_spots': len(spots), 'fitted_configurations': len(maps.groupby(['method_id', 'K', 'replicate_id'])),
                  'files': {p.name: {'sha256': digest(p), 'bytes': p.stat().st_size} for p in section.iterdir() if p.is_file()}}
        (section / 'manifest.json').write_text(json.dumps(record, indent=2) + '\n')
        attempts.extend(section_attempts)
        print(f'Fit {record["fitted_configurations"]} configurations for {sample}', flush=True)
    (args.output / 'run_manifest.json').write_text(json.dumps({
        'schema_version': 'new_source_map_fit_v1', 'run_type': 'new_fits_from_public_counts',
        'section_sources': rows, 'software': versions, 'loader': args.loader,
        'methods_requested': list(methods), 'K': [4, 6], 'seeds': [11, 23, 37],
        'source_identity_sha256': digest(args.inputs / 'source_identity.json'),
        'scientific_function_sha256': digest(Path(science.__file__)),
        'official_function_sha256': digest(Path(official.__file__)),
        'equivalence_to_published_fits': 'Requires separate comparison; new fits are not substituted silently for saved publication maps',
    }, indent=2) + '\n')
    pd.DataFrame(attempts).to_csv(args.output / 'fit_availability.tsv', sep='\t', index=False)


if __name__ == '__main__':
    main()
