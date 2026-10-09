#!/usr/bin/env python3
"""Recalculate the study and generate Figures 1–3, S1–S8 and Tables S1–S16."""
from pathlib import Path
import argparse,hashlib,importlib.metadata,json,shutil,subprocess,sys

REPOSITORY=Path(__file__).resolve().parents[1]
ANALYSIS=REPOSITORY/'scripts/analysis_pipeline'
FIGURES=REPOSITORY/'scripts/figures'
SOURCE=REPOSITORY/'scripts/source_pipeline'

def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()

def identities(folder, suffixes=None):
    return {p.relative_to(folder).as_posix():file_hash(p) for p in sorted(folder.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and (suffixes is None or p.suffix in suffixes)}

def establish_run(output,workspace,counts):
    identity={'inputs':identities(workspace/'data'),'configuration':identities(workspace/'config'),
              'scripts':identities(REPOSITORY/'scripts',{'.py','.R'}),'count_root':str(counts),
              'python':sys.version,'software':{p:importlib.metadata.version(p) for p in
                  ['numpy','pandas','scipy','scikit-learn','h5py','igraph','leidenalg','matplotlib','Pillow','threadpoolctl']}}
    path=output/'run_inputs.json'
    if path.exists():
        if json.loads(path.read_text())!=identity:
            raise ValueError('Inputs or analysis scripts changed; select a new output directory')
    else:path.write_text(json.dumps(identity,indent=2)+'\n')

def check_counts(workspace,counts):
    manifest=json.loads((workspace/'config/source_pipeline/source_counts_manifest.json').read_text())
    for section in manifest['sections']:
        for expected in section['source_files']:
            path=counts/section['resource']/section['sample_id']/'members'/expected['name']
            if path.stat().st_size!=expected['size_bytes'] or file_hash(path)!=expected['sha256']:
                raise ValueError('Original count identity changed: '+str(path))

def completed_stage(output,target,resume,action):
    record=output/('completed_'+target.name+'.json')
    if resume and record.exists():
        if json.loads(record.read_text())!=identities(target):
            raise ValueError('Completed output files changed: '+str(target))
        return
    action()
    record.write_text(json.dumps(identities(target),indent=2)+'\n')

def run(script,*args,python=None):
    subprocess.run([str(python or sys.executable),str(script),*map(str,args)],check=True)

def figures(output,workspace):
    commands=[('make_jbcb_corrected_tissue_figures.py',['--pack-dir',output,'--source-root',workspace]),
              ('make_jbcb_corrected_main_figures.py',['--pack-dir',output,'--source-root',workspace]),
              ('make_jbcb_corrected_support_figure.py',['--pack-dir',output]),
              ('make_jbcb_corrected_sensitivity_figures.py',['--pack-dir',output,'--analysis-dir',workspace/'analysis/regional']),
              ('make_jbcb_corrected_graph_figure.py',['--pack-dir',output]),
              ('make_jbcb_source_coverage_figure.py',['--pack-dir',output]),
              ('make_jbcb_readout_withholding_figure.py',['--package',output,'--results',workspace/'analysis/withholding'])]
    for name,args in commands:run(FIGURES/name,*args)
    sys.path.insert(0,str(ANALYSIS))
    from report_tables import published_names
    published_names(output/'tables',REPOSITORY/'config/analysis_pipeline/table_names.tsv')

def tables(output,workspace,count_root):
    config=workspace/'config/analysis_pipeline';analysis=workspace/'analysis';dest=output/'tables';dest.mkdir(exist_ok=True)
    for name in ['simulation_summary.tsv','simulation_cell_metrics.tsv','simulation_calibration_by_map.tsv','simulation_threshold_sensitivity.tsv']:
        shutil.copy2(analysis/'simulation'/name,dest/name)
    for name in ['component_source_recovery.tsv','source_label_coverage.tsv']:
        shutil.copy2(analysis/'source'/name,dest/name)
    for folder in ['figure_source_data','additional_source_data']:
        shutil.copytree(analysis/'source'/folder,dest/folder,dirs_exist_ok=True)
    run(ANALYSIS/'build_jbcb_corrected_evidence_tables.py','--root',workspace,'--results',analysis/'regional','--protocol',config/'regional.json','--output-dir',output,'--auxiliary',dest)
    run(ANALYSIS/'report_tables.py','--spatial',analysis/'spatial','--tables',dest,'--config',config,'--precision',analysis/'precision','--markers',analysis/'markers','--withholding',analysis/'withholding')
    run(ANALYSIS/'summarize_jbcb_location_matching_variation.py','--tables',dest,'--output',dest)
    sys.path.insert(0,str(ANALYSIS))
    from report_tables import published_names
    published_names(dest,config/'table_names.tsv')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=REPOSITORY/'results')
    p.add_argument('--counts',type=Path,help='Verified original members organized as RESOURCE/SAMPLE/members')
    p.add_argument('--cache',type=Path,default=REPOSITORY/'raw_cache')
    p.add_argument('--offline',action='store_true');p.add_argument('--resume',action='store_true')
    p.add_argument('--stage',choices=['all','analysis','tables','figures'],default='all')
    p.add_argument('--workers',type=int,default=2);p.add_argument('--simulation-workers',type=int,default=1)
    p.add_argument('--source-fits',type=Path,help='Complete new original-count fits, made with fit_source_maps.py')
    a=p.parse_args();output=a.output.resolve();workspace=output/'workspace'
    if a.workers<1 or a.simulation_workers<1:raise ValueError('Workers must be positive')
    if a.stage in ['all','analysis']:
        if output.exists() and not a.resume:raise FileExistsError('Use a new output directory or --resume')
        output.mkdir(parents=True,exist_ok=True)
        if not workspace.exists():
            if a.source_fits:run(SOURCE/'bridge_inputs.py','--fits',a.source_fits.resolve(),'--workspace',workspace)
            else:
                shutil.copytree(REPOSITORY/'data',workspace/'data');shutil.copytree(REPOSITORY/'config',workspace/'config')
        counts=a.counts.resolve() if a.counts else output/'original_inputs/counts'
        establish_run(output,workspace,counts)
        if not a.counts:
            args=['--cache',a.cache.resolve(),'--output',output/'original_inputs','--workers',a.workers]
            if a.offline:args.append('--offline')
            if a.resume and (output/'original_inputs').exists():args.append('--resume')
            run(SOURCE/'source_inputs.py',*args)
        check_counts(workspace,counts)
        analysis=workspace/'analysis';config=workspace/'config/analysis_pipeline'
        run(ANALYSIS/'run_jbcb_corrected_statistics.py','--root',workspace,'--protocol',config/'regional.json','--workers',a.workers)
        # Regional section checkpoints retain their input and output identities.
        sim=analysis/'simulation'
        def simulate():
            if sim.is_dir() and not any(sim.iterdir()):sim.rmdir()
            args=['--manifest',config/'simulation.json','--output',sim,'--workers',a.simulation_workers]
            if a.resume:args.append('--resume')
            run(ANALYSIS/'run_simulation.py',*args)
        completed_stage(output,sim,a.resume,simulate)
        spatial=['--workspace',workspace,'--results',analysis/'spatial','--workers',a.workers]
        run(ANALYSIS/'spatial/run_spatial_sensitivity.py',*spatial)
        for name in ['extract_main_frame.py','summarize_sensitivity.py','summarize_component_availability.py']:
            run(ANALYSIS/'spatial'/name,*spatial)
        for name,folder,extra in [('summarize_jbcb_contractile_markers.py','markers',[]),('run_jbcb_readout_withholding.py','withholding',['--plan',config/'withholding.json'])]:
            target=analysis/folder
            completed_stage(output,target,a.resume,lambda:run(ANALYSIS/name,'--workspace',workspace,'--count-root',counts,'--source-manifest',workspace/'config/source_pipeline/source_counts_manifest.json','--output',target,'--workers',a.workers,*extra))
        precision_args=['--workspace',workspace,'--output',analysis/'precision']
        if a.resume:precision_args.append('--resume')
        completed_stage(output,analysis/'precision',a.resume,lambda:run(ANALYSIS/'estimate_jbcb_control_reference_precision.py',*precision_args))
        run(ANALYSIS/'source_summaries.py','--workspace',workspace,'--count-root',counts,'--source-manifest',workspace/'config/source_pipeline/source_counts_manifest.json','--output',analysis/'source')
        run(ANALYSIS/'check_withholding.py','--workspace',workspace,'--count-root',counts,'--source-manifest',workspace/'config/source_pipeline/source_counts_manifest.json','--plan',config/'withholding.json','--output',analysis/'withholding','--reference-anchors',analysis/'regional/anchor_section_summary.tsv','--workers',a.workers)
        (output/'completed_withholding.json').write_text(json.dumps(identities(analysis/'withholding'),indent=2)+'\n')
        (output/'analysis_inputs.json').write_text(json.dumps({'count_root':str(counts),'run_type':'new source fits' if a.source_fits else 'saved publication fits'},indent=2)+'\n')
    if a.stage in ['all','tables']:
        record=json.loads((output/'analysis_inputs.json').read_text());tables(output,workspace,Path(record['count_root']))
    if a.stage in ['all','figures']:figures(output,workspace)
    print('Scientific outputs:',output)

if __name__=='__main__':main()
