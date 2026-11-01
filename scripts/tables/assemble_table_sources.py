#!/usr/bin/env python3
"""Assemble workbook inputs from regenerated analysis result directories."""
import argparse,csv,hashlib,json,shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
SIMULATION_PROVENANCE={
    'simulation_summary.tsv': ('manifest_sha256','freeze_sha256'),
    'simulation_cell_metrics.tsv': ('freeze_sha256',),
    'simulation_threshold_sensitivity.tsv': ('manifest_sha256','freeze_sha256'),
}

def simulation_source(src,dst,name):
    """Attach the original study-design provenance, not a new run freeze."""
    reference=ROOT/'data/table_inputs'/name
    with reference.open(newline='',encoding='utf-8') as f:
        reader=csv.DictReader(f,delimiter='\t')
        reference_header=reader.fieldnames
        values={key:set() for key in SIMULATION_PROVENANCE[name]}
        for row in reader:
            for key in values: values[key].add(row[key])
    if any(len(v)!=1 for v in values.values()):
        raise ValueError('Historical simulation provenance is not constant: '+name)
    values={key:next(iter(v)) for key,v in values.items()}
    manifest=hashlib.sha256((ROOT/'config/analysis_pipeline/simulation.json').read_bytes()).hexdigest()
    if 'manifest_sha256' in values and values['manifest_sha256']!=manifest:
        raise ValueError('Historical simulation manifest does not match the current design')
    with src.open(newline='',encoding='utf-8') as inp, dst.open('w',newline='',encoding='utf-8') as out:
        reader=csv.DictReader(inp,delimiter='\t')
        if 'manifest_sha256' in reader.fieldnames and 'freeze_sha256' in values:
            expected_manifest=manifest
        else: expected_manifest=None
        if any(key in reader.fieldnames for key in values):
            raise ValueError('Unexpected simulation provenance already present: '+name)
        if set(reader.fieldnames).union(values)!=set(reference_header):
            raise ValueError('Simulation schema does not match the historical source: '+name)
        writer=csv.DictWriter(out,fieldnames=reference_header,delimiter='\t',lineterminator='\n')
        writer.writeheader()
        for row in reader:
            if expected_manifest and row['manifest_sha256']!=expected_manifest:
                raise ValueError('Recomputed simulation used a different manifest')
            row.update(values);writer.writerow(row)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True,help='Tables produced by the baseline analysis')
    p.add_argument('--regional',type=Path,required=True)
    p.add_argument('--domains',type=Path,required=True)
    p.add_argument('--purity',type=Path,required=True,help='Output of summarize_purity.py')
    p.add_argument('--composition',type=Path,required=True)
    p.add_argument('--balance',type=Path,required=True)
    p.add_argument('--marker-context',type=Path,required=True,
                   help='Completed marker-context outputs, including native RCTD membership')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();config=json.loads((ROOT/'config/table_workbooks.json').read_text())
    needed={x['source'] for sheets in config.values() for x in sheets if x.get('kind')!='metadata'}
    needed.add('composition_sensitivity/rctd_mesenchymal_marker_fit_membership.tsv')
    missing=[];pairs=[]
    for name in sorted(needed):
        path=Path(name)
        if path.is_absolute() or '..' in path.parts:raise ValueError('Unsafe source path: '+name)
        if path.parts[0]=='regional_profiles':src=a.regional/Path(*path.parts[1:])
        elif path.parts[0]=='pure_domain_selection':src=a.purity if path.name=='original_purity90_denominators.tsv' else a.domains/Path(*path.parts[1:])
        elif path.parts[0]=='inferred_composition' and path.name.startswith('dominant_balance_'):src=a.balance/path.name.removeprefix('dominant_balance_')
        elif path.parts[0]=='inferred_composition':src=a.composition/Path(*path.parts[1:])
        elif path.parts[0]=='composition_sensitivity':src=a.marker_context/Path(*path.parts[1:])
        else:src=a.baseline/path
        if not src.is_file():missing.append(str(src))
        else:pairs.append((src,a.output/path))
    if missing:raise FileNotFoundError('Required regenerated sources are missing:\n'+'\n'.join(missing))
    for src,dst in pairs:
        dst.parent.mkdir(parents=True,exist_ok=True)
        if dst.exists():raise FileExistsError('Choose an empty output directory: '+str(dst))
        if dst.name in SIMULATION_PROVENANCE and dst.parent==a.output:
            simulation_source(src,dst,dst.name)
        else:
            shutil.copyfile(src,dst)
    print('Assembled',len(pairs),'scientific source tables')
if __name__=='__main__':main()
