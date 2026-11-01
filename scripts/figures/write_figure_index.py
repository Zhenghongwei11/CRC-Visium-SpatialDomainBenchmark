#!/usr/bin/env python3
"""Index generated figures, mapped source data and producing scripts."""
import argparse
import hashlib
import json
from pathlib import Path

from figure_numbering import FIGURE_OLD_TO_NEW, figure_name


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    figures = root / 'figures'
    baseline_sources = root / 'baseline/tables/figure_source_data'

    old_producers = {
        1: 'make_jbcb_corrected_support_figure.py',
        2: 'make_jbcb_corrected_main_figures.py',
        3: 'make_jbcb_corrected_sensitivity_figures.py',
        4: 'make_jbcb_corrected_sensitivity_figures.py',
        5: 'make_jbcb_corrected_tissue_figures.py',
        6: 'make_jbcb_corrected_graph_figure.py',
        7: 'make_jbcb_source_coverage_figure.py',
        8: 'make_jbcb_readout_withholding_figure.py',
    }
    producers = {
        'figure1': 'make_jbcb_corrected_tissue_figures.py',
        'figure2': 'make_jbcb_corrected_main_figures.py',
        'figure3': 'make_jbcb_corrected_main_figures.py',
        **{figure_name(f'figureS{old}'): producer
           for old, producer in old_producers.items()},
        figure_name('figureS9'): 'make_regional_profile_figures.py',
        figure_name('figureS10'): 'make_regional_profile_figures.py',
        figure_name('figureS11'): 'make_jbcb_corrected_main_figures.py',
    }
    regional = json.loads((figures / 'FIGURE_SOURCES.json').read_text())
    names = ['figure1', 'figure2', 'figure3',
             *[f'figureS{i}' for i in range(1, len(FIGURE_OLD_TO_NEW) + 1)]]
    records = {}
    for name in names:
        exports = {f'{name}.{ext}': sha(figures / f'{name}.{ext}')
                   for ext in ('png', 'pdf', 'tiff')}
        if name in regional:
            if exports != regional[name]['exports']:
                raise ValueError('Figure exports differ from their source receipt: ' + name)
            source_data = {f'table_sources/{path}': digest
                           for path, digest in regional[name]['sources'].items()}
            for path, digest in source_data.items():
                if sha(root / path) != digest:
                    raise ValueError('Figure source changed: ' + path)
            producer = 'make_regional_profile_figures.py'
        else:
            # Supplementary Figure S10 reuses the source rows for main Figure 3.
            original = 'figure3' if name == figure_name('figureS11') else name
            files = sorted(p for p in baseline_sources.glob(original + '*') if p.is_file())
            if not files:
                raise FileNotFoundError('No figure source data for ' + name)
            source_data = {str(path.relative_to(root)): sha(path) for path in files}
            producer = producers[name]
        records[name] = {
            'producer': 'scripts/figures/' + producer,
            'source_data': source_data,
            'exports': exports,
        }
    (figures / 'FIGURE_INDEX.json').write_text(json.dumps(records, indent=2) + '\n')
    print('Indexed', len(records), 'figures')


if __name__ == '__main__':
    main()
