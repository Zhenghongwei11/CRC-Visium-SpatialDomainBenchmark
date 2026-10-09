"""Portable path configuration for fixed-map sensitivity replay."""
from pathlib import Path
import argparse


def parse_configuration(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--workspace', type=Path, required=True,
                        help='Root of the supplied corrected_nominal_statistics_v1/workspace tree')
    parser.add_argument('--results', type=Path, required=True,
                        help='New output directory, absolute or relative to workspace; existing outputs are preserved')
    parser.add_argument('--protocol', type=Path,
                        default=Path('config/analysis_pipeline/regional.json'))
    parser.add_argument('--record', '--plan', dest='plan', type=Path,
                        default=Path(__file__).resolve().parents[3]/'config/analysis_pipeline/spatial.json',
                        help='Retrospective method account; --plan is a legacy alias')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--main-frame-only', action='store_true')
    args = parser.parse_args()
    args.workspace = args.workspace.resolve()
    args.results = (args.workspace/args.results).resolve() if not args.results.is_absolute() else args.results.resolve()
    args.protocol = (args.workspace/args.protocol).resolve() if not args.protocol.is_absolute() else args.protocol.resolve()
    args.plan = args.plan.resolve()
    return args
