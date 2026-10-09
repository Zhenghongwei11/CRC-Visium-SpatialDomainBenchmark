#!/usr/bin/env python3
"""Draw continuous corrected contrasts and the inherited support classifications."""
import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from jbcb_presentation_paths import read, default_package


def short_section_labels(frame):
    sections = frame[["resource", "sample_id", "patient_id"]].drop_duplicates().copy()
    resource_codes = {"GSE294385": "", "Valdeolivas": ""}
    labels = []
    for row in sections.itertuples(index=False):
        sample = str(row.sample_id)
        sample = re.sub(r"^SN\d+_", "", sample)
        sample = re.sub(r"_Rep(\d+)$", r"-R\1", sample)
        sample = sample.replace("_", "-")
        code = resource_codes.get(str(row.resource), str(row.resource)[:1].upper())
        labels.append(f"{code}-{sample}" if code else sample)
    sections["section_label"] = labels
    if sections.section_label.duplicated().any():
        for index in sections.index[sections.section_label.duplicated(keep=False)]:
            patient = re.search(r"(\d+)$", str(sections.at[index, "patient_id"]))
            suffix = patient.group(1) if patient else str(sections.at[index, "patient_id"])[-4:]
            sections.at[index, "section_label"] += f"-P{suffix}"
    if sections.section_label.duplicated().any():
        dup = sections.section_label.duplicated(keep=False)
        for rank, index in enumerate(sections.index[dup], start=1):
            sections.at[index, "section_label"] += f"-{rank:02d}"
    if sections.section_label.duplicated().any():
        raise ValueError("Short section labels are not unique")
    return {(r.resource, r.sample_id): r.section_label for r in sections.itertuples(index=False)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pack-dir', type=Path, default=default_package(__file__))
    args = parser.parse_args(); si = args.pack_dir / 'tables'
    (args.pack_dir / 'figures').mkdir(parents=True, exist_ok=True)
    source = si / 'figure_source_data'; source.mkdir(parents=True, exist_ok=True)
    anchors = read(si / 'anchor_section_summary.tsv')
    maps = read(si / 'current_per_map_results.tsv')
    maps = maps[maps.estimator_id.eq('anchor_retention')].copy()
    label_by_section = short_section_labels(anchors)
    anchors['section_label'] = [label_by_section[(r, s)] for r, s in zip(anchors.resource, anchors.sample_id, strict=True)]
    maps['section_label'] = [label_by_section[(r, s)] for r, s in zip(maps.resource, maps.sample_id, strict=True)]
    thresholds = read(si / 'real_threshold_sensitivity.tsv')
    thresholds = thresholds[thresholds.graph_id.eq('historical_raw_6nn') & thresholds.setting_id.eq('baseline') &
                            thresholds.estimator_id.eq('anchor_retention')]
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.size': 8.5,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig = plt.figure(figsize=(8, 6.8))
    grid = fig.add_gridspec(2, 2, width_ratios=(1.35, 1), left=.20, right=.94, top=.95, bottom=.20, wspace=.40, hspace=.43)
    aa = fig.add_subplot(grid[:, 0]); ab = fig.add_subplot(grid[0, 1]); ac = fig.add_subplot(grid[1, 1])
    anchors['panel_A_plotted'] = anchors.anchor_status.eq('evaluable') & anchors.primary_scale_mad.gt(0)
    a = anchors[anchors.panel_A_plotted].sort_values(['resource', 'morphology_anchor_delta'])
    for i, r in enumerate(a.itertuples()):
        color = '#D55E00' if r.anchor_supported else '#65717B'
        aa.plot(np.array([r.morphology_anchor_low, r.morphology_anchor_high]) / r.primary_scale_mad, [i, i], c=color, lw=1)
        aa.scatter(r.morphology_anchor_delta / r.primary_scale_mad, i, s=25, c=color)
    aa.set(yticks=range(len(a)), yticklabels=a.section_label, xlabel='Near minus far score / section MAD')
    aa.invert_yaxis(); aa.axvline(0, c='#B7C0C5', lw=.7); aa.set_title('A  Full-band contrasts', loc='left', weight='bold')
    supported = anchors[anchors.anchor_supported].sort_values(['resource', 'sample_id'])
    maps['panel_B_plotted'] = maps.status.eq('success') & maps.anchor_supported & maps.primary_scale_mad.gt(0)
    for i, r in enumerate(supported.itertuples()):
        ab.scatter(i - .22, r.morphology_anchor_delta / r.primary_scale_mad, marker='D', c='#009E73', s=30)
        g = maps[maps.sample_id.eq(r.sample_id) & maps.panel_B_plotted]
        for arm, offset, color in [(True, 0, '#24536B'), (False, .22, '#D55E00')]:
            v = (g.loc[g.is_reference.eq(arm), 'computational_delta'] / g.loc[g.is_reference.eq(arm), 'primary_scale_mad']).to_numpy()
            if len(v):
                ab.scatter(np.repeat(i + offset, len(v)), v, s=10, alpha=.4, c=color)
                ab.scatter(i + offset, np.median(v), marker='_', s=100, c=color)
    labels = [r.section_label for r in supported.itertuples()]
    ab.set(xticks=range(len(labels)), xticklabels=labels, ylabel='Near minus far score / section MAD')
    ab.tick_params(axis='x', labelsize=9); ab.axhline(0, c='#B7C0C5', lw=.7)
    for tick_label in ab.get_xticklabels():
        tick_label.set_rotation(35)
        tick_label.set_ha('right')
        tick_label.set_rotation_mode('anchor')
    ab.set_title('B  Full and selected bands', loc='left', weight='bold')
    curve = thresholds.groupby(['map_arm', 'margin_mad'], as_index=False)[['anchor_pass', 'map_pass']].sum()
    curve['map_pass_fraction'] = curve.map_pass / curve.anchor_pass.where(curve.anchor_pass.gt(0))
    for arm, color, label in [('reference', '#24536B', 'Default map setting'),
                              ('alternative', '#D55E00', 'Other map settings')]:
        g = curve[curve.map_arm.eq(arm)].sort_values('margin_mad')
        ac.plot(g.margin_mad, g.map_pass_fraction, marker='o', ms=4, lw=1.4, c=color, label=label)
    ac.set(xlabel='Support margin (section MAD)', ylabel='Fraction retaining support', ylim=(0, 1), xticks=[0, .25, .5, .75])
    ac.set_title('C  Support-margin sensitivity', loc='left', weight='bold'); ac.legend(frameon=False, fontsize=8)
    fig.legend(handles=[Line2D([], [], marker=m, ls='', c=c, label=l) for m, c, l in
                        [('D', '#009E73', 'Full bands'), ('o', '#24536B', 'Default map setting'), ('o', '#D55E00', 'Other map settings')]],
               loc='lower center', ncol=3, frameon=False)
    for ext in ['pdf', 'png', 'tiff']:
        options = {'dpi': 600 if ext == 'tiff' else 300}
        if ext == 'tiff': options['pil_kwargs'] = {'compression': 'tiff_lzw'}
        fig.savefig(args.pack_dir / 'figures' / ('figureS1.' + ext), **options)
    plt.close(fig)
    anchors.to_csv(source / 'figureS1_panel_A.tsv', sep='\t', index=False)
    maps.to_csv(source / 'figureS1_panel_B.tsv', sep='\t', index=False)
    curve.to_csv(source / 'figureS1_panel_C.tsv', sep='\t', index=False)
    print('Rendered corrected support Figure S1')


if __name__ == '__main__':
    main()
