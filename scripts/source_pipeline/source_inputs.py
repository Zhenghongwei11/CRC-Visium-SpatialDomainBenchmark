#!/usr/bin/env python3
"""Obtain and register the original count, coordinate and annotation inputs."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import subprocess
import tarfile
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

CONFIG = Path(__file__).resolve().parents[2] / 'config/source_pipeline'


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def obtain(record, cache, offline=False):
    if Path(record['filename']).name != record['filename'] or not record['url'].startswith('https://'):
        raise ValueError('Source records require a basename and HTTPS URL')
    path = Path(cache) / record['filename']
    if not path.exists():
        if offline:
            raise FileNotFoundError(f'Place the original file in the cache: {path}')
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + '.partial')
        try:
            if shutil.which('curl'):
                subprocess.run(['curl', '--fail', '--location', '--silent', '--show-error',
                                '--retry', '8', '--retry-all-errors', '--continue-at', '-',
                                '--connect-timeout', '30', '--max-time', '3600',
                                '--output', str(temporary), record['url']], check=True)
            else:
                request = urllib.request.Request(record['url'], headers={'User-Agent': 'CRC-Visium-research/1.0'})
                with urllib.request.urlopen(request, timeout=300) as source, temporary.open('wb') as target:
                    shutil.copyfileobj(source, target, 1024 * 1024)
            verify(temporary, record)
            temporary.replace(path)
        except Exception:
            # An interrupted transfer can be resumed and must pass the source hash.
            raise
    verify(path, record)
    return path


def verify(path, record):
    if path.stat().st_size != record['bytes'] or digest(path) != record['sha256']:
        raise ValueError(f'Original source identity mismatch: {path.name}')


def write_member(root, sample, name, data):
    basename = Path(name).name
    plain = basename.removesuffix('.gz')
    accepted = {'filtered_feature_bc_matrix.h5', 'matrix.mtx', 'barcodes.tsv', 'features.tsv',
                'genes.tsv', 'tissue_positions.csv', 'tissue_positions_list.csv',
                'scalefactors_json.json', 'scalefactors.json', 'tissue_lowres_image.png',
                'tissue_hires_image.png', 'detected_tissue_image.jpg'}
    if plain not in accepted:
        return None
    output = root / (sample + '_' + basename)
    if not basename.endswith(('.gz', '.h5')):
        output = output.with_name(output.name + '.gz')
        data = gzip.compress(data, mtime=0)
    if output.exists() and output.read_bytes() != data:
        raise ValueError(f'Multiple different archive members for {output.name}')
    output.write_bytes(data)
    return output


def extract(package, section, output):
    output.mkdir(parents=True, exist_ok=True)
    sample = section['sample_id']
    if package.suffix == '.zip':
        with zipfile.ZipFile(package) as archive:
            for member in archive.infolist():
                if not member.is_dir():
                    write_member(output, sample, member.filename, archive.read(member))
    else:
        with tarfile.open(package, 'r:*') as archive:
            for member in archive.getmembers():
                if member.isfile():
                    with archive.extractfile(member) as stream:
                        write_member(output, sample, member.name, stream.read())
    for expected in section['source_files']:
        verify(output / expected['name'], {'bytes': expected['size_bytes'], 'sha256': expected['sha256']})


def section_inputs(section, sources, cache, output, offline, discard_archives):
    sample = section['sample_id']
    source = sources['packages'][sample]
    preexisting = (cache / source['filename']).exists()
    package = obtain(source, cache, offline)
    flat = output / 'data' / section['dataset_id'] / 'extracted'
    extract(package, section, flat)
    members = output / 'counts' / section['resource'] / sample / 'members'
    members.mkdir(parents=True, exist_ok=True)
    for expected in section['source_files']:
        target = members / expected['name']
        if not target.exists():
            try:
                target.hardlink_to(flat / expected['name'])
            except OSError:
                shutil.copy2(flat / expected['name'], target)
    annotation = output / 'annotations' / section['resource'] / ('Pathologist_Annotations_' + sample + '.csv')
    row = {'dataset_id': section['dataset_id'], 'resource': section['resource'],
           'sample_id': sample, 'patient_id': section['patient_id'], 'section_id': section['section_id'],
           'bundle_root': flat.relative_to(output).as_posix(),
           'annotation_file': annotation.relative_to(output).as_posix(),
           'package_sha256': digest(package), 'package_bytes': package.stat().st_size,
           'annotation_sha256': digest(annotation)}
    if discard_archives and not preexisting:
        package.unlink()
    print(f'Obtained and verified {sample}', flush=True)
    return row


def annotations(resource, sections, cache, output, sources, offline, rscript):
    output.mkdir(parents=True, exist_ok=True)
    record = sources['annotations'][resource]
    original = obtain(record, cache, offline)
    if resource == 'Valdeolivas':
        with zipfile.ZipFile(original) as archive:
            for section in sections:
                name = 'Pathologist_Annotations_' + section['sample_id'] + '.csv'
                matches = [p for p in archive.namelist() if Path(p).name == name]
                if len(matches) != 1:
                    raise ValueError(f'Expected one original annotation member: {name}')
                (output / name).write_bytes(archive.read(matches[0]))
    else:
        subprocess.run([rscript, str(Path(__file__).with_name('export_gse_annotations.R')),
                        str(original), str(output), str(obtain(sources["annotations"]["GSE294385_raw"], cache, offline))], check=True)
    return original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New original-input extraction directory')
    parser.add_argument('--resource', choices=['Valdeolivas', 'GSE294385', 'all'], default='all')
    parser.add_argument('--samples', default='', help='Comma-separated sample identifiers; empty means all')
    parser.add_argument('--offline', action='store_true', help='Use verified original packages already in cache')
    parser.add_argument('--resume', action='store_true', help='Resume source acquisition; all existing source files are reverified')
    parser.add_argument('--rscript', default='Rscript')
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--workers', type=int, choices=[1, 2, 3, 4], default=2)
    parser.add_argument('--discard-archives', action='store_true',
                        help='Remove only archives downloaded by this invocation after successful extraction')
    args = parser.parse_args()
    if args.output.exists() and not args.resume:
        raise FileExistsError('Select a new output directory; existing extraction is not overwritten')
    sources = json.loads((args.config / 'sources.json').read_text())
    counts = json.loads((args.config / 'source_counts_manifest.json').read_text())
    requested = set(args.samples.split(',')) if args.samples else set()
    sections = [s for s in counts['sections'] if args.resource in ['all', s['resource']]
                and (not requested or s['sample_id'] in requested)]
    if not sections or requested - {s['sample_id'] for s in sections}:
        raise ValueError('No sections or unknown sample identifiers')
    args.output.mkdir(parents=True, exist_ok=args.resume)
    files, manifest = [], []
    for resource in sorted({s['resource'] for s in sections}):
        resource_sections = [s for s in sections if s['resource'] == resource]
        annpath = args.output / 'annotations' / resource
        original_ann = annotations(resource, resource_sections, args.cache, annpath, sources,
                                   args.offline, args.rscript)
        files.append({'kind': 'annotation', 'resource': resource, 'filename': original_ann.name,
                      'sha256': digest(original_ann), 'bytes': original_ann.stat().st_size})
        if resource == 'GSE294385':
            original_raw = obtain(sources['annotations']['GSE294385_raw'], args.cache, args.offline)
            files.append({'kind': 'annotation_raw', 'resource': resource, 'filename': original_raw.name,
                          'sha256': digest(original_raw), 'bytes': original_raw.stat().st_size})
        with ThreadPoolExecutor(args.workers) as pool:
            jobs = [pool.submit(section_inputs, s, sources, args.cache, args.output,
                                args.offline, args.discard_archives) for s in resource_sections]
            for job in as_completed(jobs):
                manifest.append(job.result())
    manifest.sort(key=lambda row: (row['resource'], row['sample_id']))
    pd.DataFrame(manifest).to_csv(args.output / 'sample_manifest.tsv', sep='\t', index=False)
    (args.output / 'source_identity.json').write_text(json.dumps({
        'schema_version': 'public_original_input_v1', 'scope': 'Original public count matrices, coordinates, images and tissue annotations',
        'sources_config_sha256': digest(args.config / 'sources.json'),
        'count_manifest_sha256': digest(args.config / 'source_counts_manifest.json'),
        'source_files': files, 'sections': manifest,
        'extracted_files': {p.relative_to(args.output).as_posix(): {'sha256': digest(p), 'bytes': p.stat().st_size}
                            for p in args.output.rglob('*') if p.is_file()},
    }, indent=2) + '\n')
    print(f'Registered original inputs for {len(manifest)} sections')


if __name__ == '__main__':
    main()
