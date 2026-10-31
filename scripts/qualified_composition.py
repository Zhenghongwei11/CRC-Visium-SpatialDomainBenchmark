#!/usr/bin/env python3
"""Describe source-qualified cell2location group composition in frozen domains.

This is a saved-input summary only. It does not open raw counts, fit a model,
choose domains by composition, perform inference, or change any earlier result.
"""
from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_info, threadpool_limits

import marker_context as marker
from marker_context import (
    ANATOMICAL, ROOT, fixed_populations, load_plan, read, saved_inputs,
    save, sha, write_json,
)

# The imported helper sets its own thread defaults; enforce one native thread.
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
_THREAD_LIMIT = threadpool_limits(limits=1)

DOC = ROOT / 'config'
TAXONOMY = DOC / 'reference_groups.json'
TAXONOMY_LOCK = DOC / 'reference_groups_lock.json'
CONTEXT = DOC / 'qualified_composition.json'
CONTEXT_LOCK = DOC / 'qualified_composition_lock.json'
W_PATH = ROOT / 'data/reference_inputs/cell2location/W_cell_density.csv'
KEYMAP_PATH = ROOT / 'data/reference_inputs/cell2location/barcode_map.tsv'
DEFAULT_OUTPUT = ROOT / 'results/qualified_composition'
RESOURCE_WITH_MODEL = 'Valdeolivas'
SHARE_TOLERANCE = 1e-12
AGGREGATION_TOLERANCE = 1e-10
THREAD_KEYS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
               'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')


def file_sha(path: Path) -> str:
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def check_lock(payload_path: Path, lock_path: Path) -> dict:
    payload = json.loads(payload_path.read_text())
    lock = json.loads(lock_path.read_text())
    assert file_sha(payload_path) == lock['sha256']
    return payload


def load_author_model():
    taxonomy = check_lock(TAXONOMY, TAXONOMY_LOCK)
    contract = check_lock(CONTEXT, CONTEXT_LOCK)
    groups = taxonomy['author_model_summary_groups']
    assert len(groups) == 8
    type_to_group = {}
    for group_name, subtypes in groups.items():
        assert subtypes
        for subtype in subtypes:
            assert subtype not in type_to_group
            type_to_group[subtype] = group_name
    assert len(type_to_group) == 36
    return taxonomy, contract, groups, type_to_group


def read_model_inputs(type_to_group: dict[str, str], groups: dict[str, list[str]]):
    w = pd.read_csv(W_PATH, dtype={'spot_id': str}, low_memory=False)
    assert w.columns[0] == 'spot_id'
    assert w.spot_id.notna().all() and not w.spot_id.duplicated().any()
    prefix = 'mean_spot_factors'
    w_type_columns = {column[len(prefix):]: column for column in w.columns[1:]
                      if column.startswith(prefix)}
    assert len(w_type_columns) == 36 and set(w_type_columns) == set(type_to_group)
    assert len(w.columns) == 37

    ordered_types = list(type_to_group)
    ordered_columns = [w_type_columns[subtype] for subtype in ordered_types]
    source_values = w[ordered_columns].to_numpy(dtype=np.float64)
    model_valid = np.isfinite(source_values).all(axis=1) & (source_values >= 0).all(axis=1)
    group_names = list(groups)
    group_values = np.full((len(w), len(group_names)), np.nan, dtype=np.float64)
    type_position = {subtype: i for i, subtype in enumerate(ordered_types)}
    for j, group_name in enumerate(group_names):
        indices = [type_position[subtype] for subtype in groups[group_name]]
        group_values[:, j] = source_values[:, indices].sum(axis=1)

    # Independent pandas sums over the source subtype columns check the 8-group
    # aggregation for every complete, finite, nonnegative W row.
    valid_values = source_values[model_valid]
    valid_frame = pd.DataFrame(valid_values, columns=ordered_types)
    group_errors = []
    for j, group_name in enumerate(group_names):
        expected = valid_frame[groups[group_name]].sum(axis=1).to_numpy(dtype=np.float64)
        actual = group_values[model_valid, j]
        error = float(np.max(np.abs(actual - expected))) if len(expected) else 0.0
        group_errors.append({'group': group_name, 'source_subtype_N': len(groups[group_name]),
                             'n_complete_W_rows': int(model_valid.sum()),
                             'max_abs_error': error})
        assert error <= AGGREGATION_TOLERANCE
    source_total = valid_values.sum(axis=1)
    grouped_total = group_values[model_valid].sum(axis=1)
    total_error = float(np.max(np.abs(source_total - grouped_total))) if len(source_total) else 0.0
    assert total_error <= AGGREGATION_TOLERANCE

    keymap = pd.read_csv(KEYMAP_PATH, sep='\t', dtype=str, low_memory=False)
    required = {'patient_id', 'sample_id', 'spot_id', 'barcode', '_merge'}
    assert required <= set(keymap.columns)
    assert keymap[list(required)].notna().all().all()
    assert keymap['_merge'].eq('both').all()
    assert not keymap.spot_id.duplicated().any()
    assert not keymap.duplicated(['sample_id', 'barcode']).any()
    assert set(w.spot_id) == set(keymap.spot_id)
    assert len(w) == len(keymap) == 20654
    assert keymap.sample_id.nunique() == 14
    assert set(keymap.groupby('sample_id').patient_id.nunique()) == {1}

    group_frame = pd.DataFrame(group_values, index=pd.Index(w.spot_id, name='spot_id'),
                               columns=group_names)
    valid_series = pd.Series(model_valid, index=pd.Index(w.spot_id, name='spot_id'))
    keymap_by_sample = {sid: frame.set_index('barcode').spot_id.to_dict()
                        for sid, frame in keymap.groupby('sample_id', sort=False)}
    input_hashes = {
        str(W_PATH.relative_to(ROOT)): file_sha(W_PATH),
        str(KEYMAP_PATH.relative_to(ROOT)): file_sha(KEYMAP_PATH),
        str(TAXONOMY.relative_to(ROOT)): file_sha(TAXONOMY),
        str(TAXONOMY_LOCK.relative_to(ROOT)): file_sha(TAXONOMY_LOCK),
        str(CONTEXT.relative_to(ROOT)): file_sha(CONTEXT),
        str(CONTEXT_LOCK.relative_to(ROOT)): file_sha(CONTEXT_LOCK),
    }
    verification = {
        'w_rows': len(w), 'w_spot_id_unique': True, 'w_subtype_columns': len(w_type_columns),
        'keymap_rows': len(keymap), 'keymap_spot_id_unique': True,
        'keymap_sample_barcode_unique': True, 'keymap_all_merge_both': True,
        'w_keymap_spot_id_sets_equal': True, 'keymap_samples': keymap.sample_id.nunique(),
        'keymap_patients': keymap.patient_id.nunique(),
        'complete_nonnegative_W_rows': int(model_valid.sum()),
        'incomplete_or_invalid_W_rows': int((~model_valid).sum()),
        'group_N': len(group_names), 'source_subtype_N': len(type_to_group),
        'every_source_subtype_assigned_once': True,
        'maximum_group_aggregation_absolute_error': max(x['max_abs_error'] for x in group_errors),
        'maximum_all_group_total_absolute_error': total_error,
        'aggregation_tolerance': AGGREGATION_TOLERANCE,
    }
    assert input_hashes[str(TAXONOMY.relative_to(ROOT))] == json.loads(TAXONOMY_LOCK.read_text())['sha256']
    assert input_hashes[str(CONTEXT.relative_to(ROOT))] == json.loads(CONTEXT_LOCK.read_text())['sha256']
    checks = pd.DataFrame(group_errors)
    return group_names, group_frame, valid_series, keymap_by_sample, keymap, input_hashes, verification, checks


def _base_identity(helper_identity: dict, input_hashes: dict) -> dict:
    return {
        **helper_identity,
        'summary_script_sha256': file_sha(Path(__file__)),
        'author_model_source8_sha256': input_hashes[str(TAXONOMY.relative_to(ROOT))],
        'author_model_source8_lock_sha256': input_hashes[str(TAXONOMY_LOCK.relative_to(ROOT))],
        'qualified_domain_context_sha256': input_hashes[str(CONTEXT.relative_to(ROOT))],
        'qualified_domain_context_lock_sha256': input_hashes[str(CONTEXT_LOCK.relative_to(ROOT))],
        'W_cell_density_sha256': input_hashes[str(W_PATH.relative_to(ROOT))],
        'Val_cell2location_keymap_sha256': input_hashes[str(KEYMAP_PATH.relative_to(ROOT))],
    }


def _safe_metadata(metadata: dict) -> dict:
    output = {}
    for key, value in metadata.items():
        if isinstance(value, np.generic):
            value = value.item()
        output[key] = value
    return output


def summarize_population(metadata: dict, mask: np.ndarray, resource: str,
                         groups: list[str], group_values: np.ndarray | None,
                         model_valid: np.ndarray | None) -> dict:
    n_selected = int(mask.sum())
    row = _safe_metadata(metadata)
    row['nselected'] = n_selected
    row['model_availability'] = 'available' if resource == RESOURCE_WITH_MODEL else 'unavailable'
    row['model_available'] = resource == RESOURCE_WITH_MODEL
    group_sums = {f'sum_abundance_{name}': np.nan for name in groups}
    shares = {f'share_{name}': np.nan for name in groups}
    row['nmodel'] = np.nan
    row['nmissing'] = np.nan
    row['model_coverage_fraction'] = np.nan
    row['sum_modelled_abundance'] = np.nan
    if resource == RESOURCE_WITH_MODEL:
        assert group_values is not None and model_valid is not None
        assert len(mask) == len(model_valid) == len(group_values)
        selected_model = mask & model_valid
        n_model = int(selected_model.sum())
        row['nmodel'] = n_model
        row['nmissing'] = n_selected - n_model
        row['model_coverage_fraction'] = (n_model / n_selected) if n_selected > 0 else np.nan
        if n_selected > 0 and n_model > 0:
            totals = group_values[selected_model].sum(axis=0, dtype=np.float64)
            total = float(totals.sum())
            if math.isfinite(total) and total > 0:
                proportions = totals / total
                assert np.isfinite(proportions).all() and (proportions >= 0).all()
                assert abs(float(proportions.sum()) - 1.0) <= SHARE_TOLERANCE
                group_sums = {f'sum_abundance_{name}': float(totals[i])
                              for i, name in enumerate(groups)}
                shares = {f'share_{name}': float(proportions[i])
                          for i, name in enumerate(groups)}
                row['sum_modelled_abundance'] = total
                row['dominant_share'] = float(np.max(proportions))
                positive = proportions > 0
                row['shannon_entropy_natlog'] = float(-np.sum(proportions[positive] * np.log(proportions[positive])))
            else:
                row['dominant_share'] = np.nan
                row['shannon_entropy_natlog'] = np.nan
        else:
            row['dominant_share'] = np.nan
            row['shannon_entropy_natlog'] = np.nan
    else:
        row['dominant_share'] = np.nan
        row['shannon_entropy_natlog'] = np.nan
    row.update(group_sums)
    row.update(shares)
    row['eligibility_basis'] = 'saved_selection_coverage_marker_only; not selected by composition'
    return row


def process_section(section: dict, coverage: pd.DataFrame, identity: dict,
                    output: Path, groups: list[str], group_frame: pd.DataFrame,
                    valid_series: pd.Series, keymap_by_sample: dict[str, dict[str, str]]) -> dict:
    sid = section['sample_id']
    folder = output / 'sections' / section['resource'] / sid
    done_path = folder / 'DONE.json'
    if done_path.exists():
        done = json.loads(done_path.read_text())
        assert done['identity'] == identity
        for name, digest in done['outputs'].items():
            assert file_sha(folder / name) == digest
        return done

    folder.mkdir(parents=True, exist_ok=True)
    spots, local, masks, section_inputs = saved_inputs(section, coverage)
    populations, _blocks, _stroma, geometry_inputs = fixed_populations(section, spots, local, masks)
    section_inputs.update(geometry_inputs)
    assert len(populations) == 99
    if section['resource'] == RESOURCE_WITH_MODEL:
        assert sid in keymap_by_sample
        local_map = keymap_by_sample[sid]
        barcodes = spots.barcode.astype(str).tolist()
        assert len(barcodes) == len(set(barcodes)) and set(local_map) <= set(barcodes)
        spot_ids = [local_map.get(barcode) for barcode in barcodes]
        mapped = np.array([spot_id is not None for spot_id in spot_ids], dtype=bool)
        section_mapped = mapped.copy()
        matched_ids = [spot_id for spot_id in spot_ids if spot_id is not None]
        assert all(spot_id in group_frame.index for spot_id in matched_ids)
        section_groups = np.full((len(spots), len(groups)), np.nan, dtype=np.float64)
        section_valid = np.zeros(len(spots), dtype=bool)
        if mapped.any():
            section_groups[mapped] = group_frame.loc[matched_ids].to_numpy(dtype=np.float64)
            section_valid[mapped] = valid_series.loc[matched_ids].to_numpy(dtype=bool)
        assert section_groups.shape == (len(spots), len(groups))
    else:
        assert section['resource'] == 'GSE294385'
        section_groups = None
        section_valid = None
        section_mapped = None

    rows = []
    expected_mask_sizes = []
    for metadata, mask in populations:
        assert len(mask) == len(spots)
        row = summarize_population(metadata, mask, section['resource'], groups,
                                   section_groups, section_valid)
        rows.append({**row, 'resource': section['resource'], 'patient_id': section['patient_id'],
                     'sample_id': sid})
        expected_mask_sizes.append(int(mask.sum()))
    frame = pd.DataFrame(rows)
    assert len(frame) == 99
    assert frame.nselected.astype(int).tolist() == expected_mask_sizes
    if section['resource'] == RESOURCE_WITH_MODEL:
        nsel = frame.nselected.to_numpy(dtype=int)
        nmod = frame.nmodel.to_numpy(dtype=int)
        nmis = frame.nmissing.to_numpy(dtype=int)
        assert np.all(nmod <= nsel) and np.array_equal(nsel, nmod + nmis)
        assert frame.loc[nsel > 0, 'model_coverage_fraction'].notna().all()
    else:
        assert frame[['nmodel', 'nmissing', 'model_coverage_fraction',
                      'sum_modelled_abundance', 'dominant_share', 'shannon_entropy_natlog']].isna().all().all()
        comp_cols = [f'sum_abundance_{g}' for g in groups] + [f'share_{g}' for g in groups]
        assert frame[comp_cols].isna().all().all()

    comp_cols = ([f'sum_abundance_{g}' for g in groups] +
                 [f'share_{g}' for g in groups] +
                 ['sum_modelled_abundance', 'dominant_share', 'shannon_entropy_natlog'])
    no_comp = frame.nselected.eq(0) | frame.nmodel.fillna(0).eq(0) | frame.sum_modelled_abundance.fillna(0).eq(0)
    assert frame.loc[no_comp, comp_cols].isna().all().all()
    valid_comp = frame.sum_modelled_abundance.notna()
    if valid_comp.any():
        sum_shares = frame.loc[valid_comp, [f'share_{g}' for g in groups]].sum(axis=1)
        assert np.max(np.abs(sum_shares.to_numpy(dtype=float) - 1.0)) <= SHARE_TOLERANCE

    output_file = folder / 'population_composition.tsv'
    save(frame, output_file)
    done = {
        'identity': identity, 'resource': section['resource'], 'patient_id': section['patient_id'],
        'sample_id': sid, 'completed_utc': stamp(), 'population_count': len(frame),
        'registered_spots': len(spots), 'model_availability': 'available' if section_groups is not None else 'unavailable',
        'model_keymap_mapped_spots': int(section_mapped.sum()) if section_mapped is not None else None,
        'complete_model_spots': int(section_valid.sum()) if section_valid is not None else None,
        'spots_without_complete_model': int(len(spots) - section_valid.sum()) if section_valid is not None else None,
        'source_inputs': section_inputs, 'raw_count_matrices_opened': 0,
        'output': output_file.name, 'outputs': {output_file.name: file_sha(output_file)},
    }
    write_json(done_path, done)
    print('COMPOSITION_SECTION_DONE', sid, 'populations', len(frame),
          'model', done['model_availability'], flush=True)
    return done


def hierarchical_summaries(frame: pd.DataFrame, groups: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    group_keys = ['resource', 'patient_id', 'setting_id', 'selection_rule', 'population']
    metric_columns = (['nselected', 'nmodel', 'nmissing', 'model_coverage_fraction',
                       'sum_modelled_abundance'] +
                      [f'sum_abundance_{g}' for g in groups] +
                      [f'share_{g}' for g in groups] +
                      ['dominant_share', 'shannon_entropy_natlog'])
    patient_rows = []
    for key, subset in frame.groupby(group_keys, dropna=False, sort=True):
        row = dict(zip(group_keys, key))
        row['n_sections_total'] = int(subset.sample_id.nunique())
        row['n_sections_model_available'] = int(subset.loc[subset.model_available, 'sample_id'].nunique())
        row['n_sections_with_composition'] = int(subset.loc[subset.dominant_share.notna(), 'sample_id'].nunique())
        row['n_sections_source_coverage_qualified'] = int(subset.loc[subset.region_coverage_qualified.eq(True), 'sample_id'].nunique())
        for metric in metric_columns:
            values = pd.to_numeric(subset[metric], errors='coerce').dropna()
            row[f'median_{metric}'] = float(values.median()) if len(values) else np.nan
            row[f'n_valid_sections_{metric}'] = int(len(values))
        patient_rows.append(row)
    patient = pd.DataFrame(patient_rows)

    resource_keys = ['resource', 'setting_id', 'selection_rule', 'population']
    resource_rows = []
    for key, subset in patient.groupby(resource_keys, dropna=False, sort=True):
        row = dict(zip(resource_keys, key))
        raw = frame[(frame.resource == row['resource']) &
                    (frame.setting_id == row['setting_id']) &
                    (frame.selection_rule == row['selection_rule']) &
                    (frame.population == row['population'])]
        row['n_patients_total'] = int(raw.patient_id.nunique())
        row['n_sections_total'] = int(raw.sample_id.nunique())
        row['n_patients_with_model'] = int(raw.loc[raw.model_available, 'patient_id'].nunique())
        row['n_sections_with_model'] = int(raw.loc[raw.model_available, 'sample_id'].nunique())
        row['n_patients_with_composition'] = int(raw.loc[raw.dominant_share.notna(), 'patient_id'].nunique())
        row['n_sections_with_composition'] = int(raw.loc[raw.dominant_share.notna(), 'sample_id'].nunique())
        row['counting_unit'] = 'unique patients; section medians first, then median across patient medians'
        for metric in metric_columns:
            column = f'median_{metric}'
            values = pd.to_numeric(subset[column], errors='coerce').dropna()
            row[f'median_of_patient_medians_{metric}'] = float(values.median()) if len(values) else np.nan
            row[f'n_valid_patients_{metric}'] = int(len(values))
        resource_rows.append(row)
    return patient, pd.DataFrame(resource_rows), {'metric_columns': metric_columns,
        'patient_aggregation': 'median across sections within resource/patient/setting/rule/population',
        'resource_aggregation': 'median across patient-level medians; valid patient N recorded for every metric',
        'independent_unit': 'patient; settings and sections are not counted as independent patients'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--coverage', type=Path, required=True)
    args = parser.parse_args()
    marker.COVERAGE=args.coverage.resolve()
    if threadpool_info():
        assert max(info.get('num_threads', 1) for info in threadpool_info()) <= 1
    started = stamp()
    tic = time.monotonic()

    # load_plan validates frozen section/coverage identities but never opens raw counts.
    _amendment, _panel, sections, coverage, helper_identity = load_plan()
    taxonomy, contract, groups, type_to_group = load_author_model()
    (group_names, group_frame, valid_series, keymap_by_sample, keymap,
     source_hashes, source_verification, group_check) = read_model_inputs(type_to_group, groups)
    assert group_names == list(groups)
    assert len(sections) == 22
    assert sum(s['resource'] == RESOURCE_WITH_MODEL for s in sections) == 14
    assert sum(s['resource'] == 'GSE294385' for s in sections) == 8
    assert {s['sample_id'] for s in sections if s['resource'] == RESOURCE_WITH_MODEL} == set(keymap_by_sample)
    assert set(keymap.sample_id) == set(keymap_by_sample)
    identity = _base_identity(helper_identity, source_hashes)

    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'STARTED.json', {'started_utc': started, 'identity': identity,
               'status': 'source8_descriptive_composition_running', 'raw_count_matrices_opened': 0,
               'thread_limit': 1})
    save(group_check, output / 'SOURCE_GROUP_AGGREGATION_CHECK.tsv')
    section_records = [process_section(s, coverage, identity, output, group_names,
                                       group_frame, valid_series, keymap_by_sample)
                      for s in sections]
    section_frames = [read(output / 'sections' / s['resource'] / s['sample_id'] /
                           'population_composition.tsv') for s in sections]
    frame = pd.concat(section_frames, ignore_index=True)
    assert len(frame) == 22 * 99
    assert frame.groupby('sample_id').size().eq(99).all()
    selected_setting_counts = frame.loc[frame.selection_rule.ne('anatomical')].groupby(
        ['sample_id', 'setting_id', 'selection_rule']).size()
    assert selected_setting_counts.eq(4).all()
    assert frame.loc[frame.resource.eq('GSE294385'), 'model_availability'].eq('unavailable').all()
    assert frame.loc[frame.resource.eq(RESOURCE_WITH_MODEL), 'model_availability'].eq('available').all()
    assert frame.loc[frame.resource.eq('GSE294385'),
                     ['nmodel', 'nmissing', 'model_coverage_fraction', 'sum_modelled_abundance',
                      'dominant_share', 'shannon_entropy_natlog'] +
                     [f'sum_abundance_{g}' for g in group_names] +
                     [f'share_{g}' for g in group_names]].isna().all().all()
    save(frame, output / 'POPULATION_COMPOSITION.tsv')
    patient, resource, aggregation = hierarchical_summaries(frame, group_names)
    save(patient, output / 'PATIENT_MEDIANS.tsv')
    save(resource, output / 'RESOURCE_SUMMARIES.tsv')

    checks = {
        'status': 'mechanical_checks_passed', 'created_utc': stamp(),
        'source8_group_order': group_names, 'source8_source_types': type_to_group,
        'n_sections': len(sections), 'n_population_masks_per_section': 99,
        'n_population_records': len(frame), 'n_valdeolivas_sections_model_available': 14,
        'n_gse294385_sections_model_unavailable': 8,
        'n_GSE_model_derived_values_imputed': 0,
        'GSE_model_derived_fields_all_NA': True,
        'empty_zero_model_zero_total_composition_fields_NA': True,
        'composition_share_sum_max_abs_error': float(np.max(np.abs(
            frame.loc[frame.sum_modelled_abundance.notna(),
                      [f'share_{g}' for g in group_names]].sum(axis=1).to_numpy(dtype=float) - 1.0)))
            if frame.sum_modelled_abundance.notna().any() else 0.0,
        'composition_share_tolerance': SHARE_TOLERANCE,
        'source_group_aggregation': source_verification,
        'eligibility_used_as_filter_or_outcome': False,
        'eligibility_field': 'saved_selection_coverage_marker_only; not selected by composition',
        'aggregation': aggregation,
        'raw_count_matrices_opened': 0,
        'thread_limit': 1, 'threadpool_info': threadpool_info(),
    }
    assert checks['composition_share_sum_max_abs_error'] <= SHARE_TOLERANCE
    save_file_names = ('POPULATION_COMPOSITION.tsv', 'PATIENT_MEDIANS.tsv',
                       'RESOURCE_SUMMARIES.tsv', 'SOURCE_GROUP_AGGREGATION_CHECK.tsv')
    output_hashes = {name: file_sha(output / name) for name in save_file_names}
    check_path = output / 'SELF_CHECK.json'
    write_json(check_path, checks)
    manifest = {
        'started_utc': started, 'completed_utc': stamp(),
        'elapsed_seconds': time.monotonic() - tic,
        'status': 'descriptive_source8_domain_composition_complete',
        'identity': identity, 'source_input_sha256': source_hashes,
        'source_verification': source_verification,
        'taxonomy_groups': groups, 'taxonomy_amendment_sha256': file_sha(TAXONOMY),
        'taxonomy_lock_sha256': file_sha(TAXONOMY_LOCK),
        'context_contract_sha256': file_sha(CONTEXT),
        'context_contract_lock_sha256': file_sha(CONTEXT_LOCK),
        'scope': 'same 99 frozen populations in 22 sections; source8 author-model grouping for Valdeolivas; GSE model unavailable',
        'source_value': 'W_cell_density.csv mean_spot_factors* columns; no q05 file, raw counts or model fitting used',
        'aggregation_definition': 'For each population, sum modeled posterior mean abundance over its selected spots within each of 8 source groups, then divide each group sum by the total across all 8 groups.',
        'dominant_share_definition': 'maximum of the 8 group shares',
        'shannon_entropy_definition': 'natural-log Shannon entropy over positive group shares; descriptive only, no biological coherence threshold',
        'missingness_definition': 'Val nmissing=nselected-nmodel for rows lacking a complete finite nonnegative 36-subtype model vector; GSE model-unavailable nmodel/nmissing/coverage/shares/sums/dominant/entropy are NA.',
        'coverage_eligibility': 'saved region_coverage_qualified is retained as a source coverage marker only; it does not filter/select populations and no composition outcome determines eligibility.',
        'resource_aggregation': aggregation,
        'n_sections': 22, 'n_valdeolivas_model_sections': 14, 'n_gse_model_unavailable_sections': 8,
        'population_masks_per_section': 99, 'n_population_records': len(frame),
        'n_patient_median_records': len(patient), 'n_resource_summary_records': len(resource),
        'n_settings_per_section': int(coverage.setting_id.nunique()),
        'n_section_setting_rule_rows': len(coverage), 'raw_count_matrices_opened': 0,
        'inference_performed': False, 'model_fitting_performed': False,
        'older_seven_group_outputs_modified': False,
        'max_threads': 1, 'python': platform.python_version(),
        'numpy': np.__version__, 'pandas': pd.__version__,
        'sections': section_records, 'outputs': output_hashes,
        'self_check_sha256': file_sha(check_path),
    }
    write_json(output / 'RUN_MANIFEST.json', manifest)
    print(json.dumps({'status': manifest['status'], 'sections': len(sections),
                      'population_records': len(frame), 'patient_median_records': len(patient),
                      'resource_summary_records': len(resource), 'raw_counts_opened': 0,
                      'max_threads': 1}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
