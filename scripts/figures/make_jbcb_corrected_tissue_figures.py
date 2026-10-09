#!/usr/bin/env python3
"""Plot registered tissue using corrected bands and nominal array scale bars."""
from pathlib import Path
import argparse
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from PIL import Image
from jbcb_presentation_paths import read, unique, sha256, default_package, default_source_root

ROOT = Path(__file__).resolve().parents[1]
GRAPH = 'historical_raw_6nn'
NEAR, FAR, TUMOR = '#D47A25', '#2878A8', '#746A83'
CASES = [('GSE294385', 'M-ST-15'), ('GSE294385', 'M-ST-31'), ('GSE294385', 'M-ST-32'),
         ('Valdeolivas', 'SN048_A416371_Rep1'), ('Valdeolivas', 'SN048_A416371_Rep2')]


def save(fig, pack, stem):
    directory = pack / 'figures'; directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / (stem + '.pdf'), metadata={'Creator': 'Scientific figure'})
    fig.savefig(directory / (stem + '.png'), dpi=300)
    fig.savefig(directory / (stem + '.tiff'), dpi=600, pil_kwargs={'compression': 'tiff_lzw'})
    plt.close(fig)


def scalebar(ax, width, height, pixels, nominal):
    x, y = width * .07, height * .9
    ax.plot([x, x + pixels], [y, y], color='white', lw=1.8,
            path_effects=[pe.Stroke(linewidth=3.4, foreground='#27242B'), pe.Normal()])
    ax.text(x + pixels / 2, y - height * .025, str(nominal) + ' µm', ha='center', fontsize=8,
            bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .86, 'pad': 1})


def context(resource, sample, protocol, source_root, results):
    section = next(s for s in protocol['source_sections'] if s['resource'] == resource and s['sample_id'] == sample)
    spots = unique(read(source_root / section['source_root'] / 'spot_inputs.tsv.gz'), 'spots')
    base = results / 'sections' / resource / sample
    g = np.load(base / 'geometry_arrays.npz', allow_pickle=False)
    spots['morphology_near'] = g[GRAPH + '|baseline|near']; spots['morphology_far'] = g[GRAPH + '|baseline|far']
    geometry = read(source_root / 'data/reference_inputs/nominal_geometry/section_geometry.tsv')
    scale = geometry.loc[geometry.sample_id.eq(sample), 'nominal_lattice_equivalent_um_per_horizontal_image_px'].iloc[0]
    return spots, base, scale


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pack-dir', type=Path, default=default_package(__file__))
    parser.add_argument('--source-root', type=Path, help='Frozen workspace with registered images, inputs and geometry')
    args = parser.parse_args(); pack = args.pack_dir
    source_root = args.source_root or default_source_root(pack, __file__)
    results = source_root / 'analysis/regional'
    assets = source_root / 'data/tissue_images/cases'
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.size': 9})
    protocol = json.loads((source_root / 'config/analysis_pipeline/regional.json').read_text())
    sources = pack / 'tables/figure_source_data'; sources.mkdir(parents=True, exist_ok=True)
    resource, sample = 'Valdeolivas', 'SN124_A938797_Rep2'
    spots, base, scale = context(resource, sample, protocol, source_root, results)
    pos = read(source_root / 'data/tissue_images/main/registered_spots.tsv')
    pos = unique(pos[pos.sample_id.eq(sample)], 'image registration')
    pos = pos.loc[spots.index]
    if not np.array_equal(pos[['x_fullres', 'y_fullres']].to_numpy(), spots[['x_fullres', 'y_fullres']].to_numpy()):
        raise ValueError('Figure 1 coordinate mismatch')
    display_scale = np.median(pos.x_display.to_numpy() / spots.x_fullres.to_numpy())
    yscale = np.median(pos.y_display.to_numpy() / spots.y_fullres.to_numpy())
    if not np.isfinite(display_scale) or display_scale <= 0 or abs(display_scale / yscale - 1) > .001:
        raise ValueError('Image display scaling is not an isotropic registration')
    members = np.load(base / 'selected_memberships.npz', allow_pickle=False)
    uid = GRAPH + '|baseline|anchor_retention|M0_expr_kmeans|6|seed_11'
    spots['retained_near'] = np.unpackbits(members[uid + '|near'])[:len(spots)].astype(bool)
    spots['retained_far'] = np.unpackbits(members[uid + '|far'])[:len(spots)].astype(bool)
    spots['x_display'] = pos.x_display; spots['y_display'] = pos.y_display
    image_path = source_root / 'data/tissue_images/main/tissue_lowres_image.png'
    image = np.asarray(Image.open(image_path).convert('RGB')); height, width = image.shape[:2]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 4.2))
    fig.subplots_adjust(left=.025, right=.975, top=.84, bottom=.16, wspace=.035)
    for i, ax in enumerate(axes):
        ax.imshow(image); ax.axis('off'); ax.set(xlim=(0, width), ylim=(height, 0))
        tumor = spots.coarse_label.eq('tumor')
        ax.scatter(spots.loc[tumor, 'x_display'], spots.loc[tumor, 'y_display'], s=9, c=TUMOR, edgecolors='white', linewidths=.25)
        for band, color in [('near', NEAR), ('far', FAR)]:
            full = spots['morphology_' + band]; kept = spots['retained_' + band]
            selected = full if i == 0 else kept
            if i:
                missing = full & ~kept
                ax.scatter(spots.loc[missing, 'x_display'], spots.loc[missing, 'y_display'], s=12.5, facecolors='none', edgecolors=color, linewidths=.7)
            ax.scatter(spots.loc[selected, 'x_display'], spots.loc[selected, 'y_display'], s=11, c=color, edgecolors='white', linewidths=.25)
        prefix = 'morphology_' if i == 0 else 'retained_'
        ax.set_title(('A  Fixed bands' if i == 0 else 'B  Retained') + f' ({int(spots[prefix + "near"].sum())} near / {int(spots[prefix + "far"].sum())} far)', loc='left', weight='bold', fontsize=9.2)
        scalebar(ax, width, height, 200 * display_scale / scale, 200)
    handles = [Line2D([], [], marker='o', ls='', color=c, label=l) for c, l in [(TUMOR, 'Pure tumor'), (NEAR, 'Near band'), (FAR, 'Far band')]]
    handles.append(Line2D([], [], marker='o', ls='', markerfacecolor='none', markeredgecolor='#555555', label='Unselected band spots'))
    fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False, fontsize=8.2)
    save(fig, pack, 'figure1')
    spots.reset_index().to_csv(sources / 'figure1_spots.tsv', sep='\t', index=False)
    figure1_scale = {'nominal_scale_bar_um': 200, 'display_scale': display_scale, 'nominal_um_per_horizontal_fullres_px': scale,
                     'scale_bar_image_pixels': 200 * display_scale / scale, 'image_sha256': sha256(image_path)}
    (sources / 'figure1_scale.json').write_text(json.dumps(figure1_scale, indent=2) + '\n')
    records = json.loads((assets / 'manifest.json').read_text())['sections']
    records = {(r['resource'], r['sample_id']): r for r in records}
    fig, axes = plt.subplots(2, 3, figsize=(8.1, 6))
    fig.subplots_adjust(left=.035, right=.98, top=.94, bottom=.025, wspace=.12, hspace=.30)
    tables, sections = [], []
    for i, (resource, sample) in enumerate(CASES):
        spots, base, scale = context(resource, sample, protocol, source_root, results); r = records[(resource, sample)]
        image_path = assets / resource / sample / r['display_image']
        expected = next(a['sha256'] for a in r['assets'].values() if a['filename'] == image_path.name)
        if sha256(image_path) != expected:
            raise ValueError('Case image hash mismatch')
        spots = spots.reset_index(); spots['image_x'] = spots.x_fullres * r['image_scale']; spots['image_y'] = spots.y_fullres * r['image_scale']
        image = np.asarray(Image.open(image_path).convert('RGB')); height, width = image.shape[:2]
        spots['within_image'] = spots.image_x.ge(0) & spots.image_x.lt(width) & spots.image_y.ge(0) & spots.image_y.lt(height)
        ax = axes.flat[i]; ax.imshow(image); ax.axis('off'); ax.set(xlim=(0, width), ylim=(height, 0))
        for band, color in [('near', NEAR), ('far', FAR)]:
            m = spots['morphology_' + band]
            if not spots.loc[m, 'within_image'].all():
                raise ValueError('A band spot lies outside source image')
            ax.scatter(spots.loc[m, 'image_x'], spots.loc[m, 'image_y'], s=5, c=color, lw=0, alpha=.85, rasterized=True)
        n, f = int(spots.morphology_near.sum()), int(spots.morphology_far.sum())
        ax.set_title(chr(65 + i) + '  ' + r['patient_id'].replace('Patient', 'Patient ') + '\n' + sample.replace('SN048_', '') + f' (near {n}; far {f})', loc='left', fontsize=9)
        barpx = 500 * r['image_scale'] / scale; scalebar(ax, width, height, barpx, 500)
        sections.append({'resource': resource, 'sample_id': sample, 'patient_id': r['patient_id'], 'n_near': n, 'n_far': f,
                         'outside_image_spots': int((~spots.within_image).sum()), 'scale_bar_um': 500, 'scale_bar_image_pixels': barpx,
                         'nominal_um_per_horizontal_fullres_px': scale, 'image_sha256': sha256(image_path)})
        spots['sample_id'] = sample; spots['resource'] = resource; tables.append(spots)
    axes.flat[-1].axis('off'); axes.flat[-1].legend(handles=handles[1:3], loc='upper left', frameon=False)
    save(fig, pack, 'figureS5')
    pd.DataFrame(sections).to_csv(sources / 'figureS5_sections.tsv', sep='\t', index=False)
    pd.concat(tables).to_csv(sources / 'figureS5_spots.tsv.gz', sep='\t', index=False, compression='gzip')
    (sources / 'corrected_tissue_figure_manifest.json').write_text(json.dumps({'generator_sha256': sha256(__file__), 'graph': GRAPH,
        'scale': 'Nominal 100-um array pitch; inverse affine horizontal image scale; not independent microscope calibration.',
        'outputs': {p.name: sha256(p) for stem in ['figure1', 'figureS5'] for p in (pack / 'figures').glob(stem + '.*')}}, indent=2) + '\n')
    print('Rendered corrected tissue figures 1 and S5')


if __name__ == '__main__':
    main()
