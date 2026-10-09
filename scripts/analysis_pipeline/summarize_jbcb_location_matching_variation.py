"""Summarize configuration variation and patient omission in location matching.

Reads the complete published sensitivity table. Reconstructs partition ->
method/K -> section -> patient medians and checks against Table S2 before
writing descriptive reporting tables. No maps, contrasts or controls are fitted.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

C = "count_only_excess_abs_deviation_mad"
B = "block_matched_excess_abs_deviation_mad"


def summarize(tables: Path, output: Path):
    data = pd.read_csv(tables / "regional_map_sensitivity.tsv", sep="\t")
    maps = data.loc[
        data.setting_id.eq("baseline")
        & data.graph_id.eq("historical_raw_6nn")
        & data.estimator_id.eq("anchor_retention")
        & data.readout_id.eq("primary_barrier")
        & data.independent_status.eq("success")
    ].copy()
    if maps.empty or not np.isfinite(maps[[C, B]].to_numpy()).all():
        raise ValueError("Primary baseline comparisons must be nonempty and finite.")
    settings = maps.groupby(
        ["resource", "patient_id", "sample_id", "method_id", "K"]
    )[[C, B]].median()
    sections = settings.groupby(["resource", "patient_id", "sample_id"])[[C, B]].median()
    patients = sections.groupby(["resource", "patient_id"])[[C, B]].median()
    published = pd.read_csv(tables / "primary_patient_summary.tsv", sep="\t")
    published = published.loc[
        published.estimator_id.eq("anchor_retention") & published.n_evaluable_maps.gt(0)
    ].set_index(["resource", "patient_id"])
    if set(patients.index) != set(published.index):
        raise ValueError("Reconstructed and published patient sets differ.")
    if not np.allclose(patients, published.loc[patients.index, [C, B]], rtol=0, atol=1e-12):
        raise ValueError("Reconstructed patient summaries differ from Table S2.")

    counts = []
    groups = [("Both resources", maps), *maps.groupby("resource")]
    for resource, group in groups:
        difference = group[C] - group[B]
        decreased = int(difference.gt(1e-12).sum())
        increased = int(difference.lt(-1e-12).sum())
        tied = int(difference.abs().le(1e-12).sum())
        n = len(group)
        if decreased + increased + tied != n:
            raise ValueError("Configuration categories do not cover every row.")
        counts.append(dict(resource=resource, n_configurations=n,
                           n_decreased=decreased, n_increased=increased, n_tied=tied,
                           fraction_decreased=decreased/n, fraction_increased=increased/n,
                           fraction_tied=tied/n, tie_tolerance_mad=1e-12))

    omitted = []
    for resource, group in patients.reset_index().groupby("resource"):
        for index, row in group.iterrows():
            remaining = group.drop(index)
            count_median, block_median = remaining[[C, B]].median()
            omitted.append(dict(resource=resource, omitted_patient_id=row.patient_id,
                                n_remaining_patients=len(remaining),
                                count_matched_patient_median_mad=count_median,
                                location_matched_patient_median_mad=block_median,
                                difference_of_patient_medians_mad=count_median-block_median))
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in [
        ("configuration_location_matching_summary.tsv", pd.DataFrame(counts)),
        ("patient_leave_one_out.tsv", pd.DataFrame(omitted)),
    ]:
        frame.to_csv(output / name, sep="\t", index=False, lineterminator="\n", float_format="%.17g")
    print(f"Summarized {len(maps)} configurations and {len(patients)} patients.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables", type=Path, required=True, help="Supporting Information directory")
    parser.add_argument("--output", type=Path, required=True, help="Directory for the two reporting tables")
    args = parser.parse_args()
    summarize(args.tables, args.output)
