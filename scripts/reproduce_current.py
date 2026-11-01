#!/usr/bin/env python3
"""Regenerate study outputs from source counts and archived publication maps.

The archived maps are required scientific inputs for the reported comparisons.
Independent refitting of all original maps is a separate workflow.
"""
import argparse,importlib.metadata,shutil,subprocess,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def run(script,*args):
    cmd=[sys.executable,str(ROOT/'scripts'/script),*map(str,args)]
    print('Running:',script,flush=True)
    subprocess.run(cmd,cwd=ROOT,check=True)

def check_python_environment():
    # Fixed package versions are required for repeatable stochastic graph fits.
    if sys.version_info[:2] != (3,13):raise RuntimeError('Python 3.13 is required; see environments/analysis-python313.txt')
    requirements=(ROOT/'environments/analysis-python313.txt').read_text().splitlines()+['openpyxl==3.1.5']
    errors=[]
    for requirement in requirements:
        if not requirement.strip() or requirement.lstrip().startswith('#'):continue
        package,expected=requirement.strip().split('==',1)
        try:actual=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:actual='not installed'
        if actual!=expected:errors.append(f'{package}: required {expected}, found {actual}')
    if errors:raise RuntimeError('Python environment differs from the recorded versions:\n'+'\n'.join(errors))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--counts',type=Path,required=True,help='Verified source counts, organized RESOURCE/SAMPLE/members')
    p.add_argument('--output',type=Path,required=True,help='New output directory')
    p.add_argument('--workers',type=int,choices=(1,2),default=2)
    p.add_argument('--rscript',default='Rscript')
    a=p.parse_args();counts=a.counts.resolve();out=a.output.resolve()
    if not counts.is_dir():raise FileNotFoundError(counts)
    if out.exists() and any(out.iterdir()):raise FileExistsError('Choose an empty output directory')
    check_python_environment()
    # Check the long-running RCTD prerequisite before beginning earlier stages.
    r_check="if (!requireNamespace('spacexr', quietly=TRUE) || !requireNamespace('Matrix', quietly=TRUE) || !requireNamespace('jsonlite', quietly=TRUE)) stop('Install spacexr, Matrix, and jsonlite'); if (as.character(packageVersion('spacexr')) != '2.2.1') stop('spacexr 2.2.1 is required'); q <- c(paste0('Q_mat_', 1:5, '.rds'), 'X_vals.rds'); if (!all(vapply(q, function(n) nzchar(system.file('extdata','Qmat',n,package='spacexr')), logical(1)))) stop('spacexr Q-matrix data are required')"
    subprocess.run([a.rscript,'--vanilla','-e',r_check],cwd=ROOT,check=True)
    out.mkdir(parents=True,exist_ok=True)
    base=out/'baseline';regional=out/'regional_profiles';domains=out/'pure_domains';coverage=out/'coverage_qualified_domains'
    composition=out/'inferred_composition';balance=out/'dominant_balance';author_support=out/'author_calibration'
    residual=out/'residual_composition';rctd=out/'rctd';support=out/'rctd_calibration';weighted=out/'rctd_profiles'
    run('reproduce.py','--counts',counts,'--output',base,'--workers',a.workers,'--simulation-workers',a.workers)
    run('regional_profiles.py','--counts',counts,'--output',regional)
    run('pure_domains.py','--counts',counts,'--output',domains)
    run('coverage_qualified_domains.py','--counts',counts,'--previous',domains,'--output',coverage)
    run('inferred_composition.py','--counts',counts,'--profiles',regional,'--domains',domains,'--output',composition)
    run('dominant_balance.py','--counts',counts,'--profiles',regional,'--output',balance)
    run('author_calibration.py','--counts',counts,'--profiles',regional,'--output',author_support)
    run('author_profiles.py','--counts',counts,'--profiles',regional,'--support',author_support,'--output',out/'author_profiles')
    maps=base/'tables/current_per_map_results.tsv'
    run('residual_composition.py','--counts',counts,'--maps-results',maps,'--output',residual)
    run('conditional_chain.py','--residual',residual,'--domains',domains,'--maps-results',maps,'--output',out/'conditional_chain')
    run('run_rctd_batch.py','--counts',counts,'--output',rctd,'--rscript',a.rscript,'--remove-generated-counts')
    run('rctd_calibration.py','--model',rctd,'--profiles',regional,'--output',support)
    run('rctd_profiles.py','--counts',counts,'--profiles',regional,'--support',support,'--output',weighted)
    run('outside_direct_fit.py','--native',rctd,'--profiles',regional,'--support',support,'--primary',weighted,'--output',out/'outside_direct_fit')
    run('marker_context.py','--counts',counts,'--coverage',coverage,'--native',rctd,'--output',out/'marker_context')
    run('qualified_composition.py','--coverage',coverage,'--output',out/'qualified_composition')
    purity=out/'original_purity90_denominators.tsv'
    run('tables/summarize_purity.py','--input',maps,'--output',purity)
    sources=out/'table_sources'
    run('tables/assemble_table_sources.py','--baseline',base/'tables','--regional',regional,'--domains',domains,'--purity',purity,'--composition',composition,'--balance',balance,'--marker-context',out/'marker_context','--output',sources)
    tables=out/'tables';tables.mkdir()
    run('tables/build_supplementary_workbooks.py','--input',sources,'--output',tables)
    shutil.copy2(sources/'current_table1.tsv',tables/'Main_Table1.tsv')
    run('tables/build_main_table2.py','--input',weighted/'same_common_patient_resource_summary.tsv','--output',tables/'Main_Table2.tsv')
    shutil.copy2(ROOT/'data/table_inputs/main_table3.tsv',tables/'Main_Table3.tsv')
    figures=out/'figures';figures.mkdir()
    for name in ['figure1','figure2',*[f'figureS{i}' for i in range(1,8)],'figureS11','figure3']:
        dest='figureS10' if name=='figure3' else name
        for ext in ('png','pdf','tiff'):
            source=base/'figures'/(name+'.'+ext)
            if not source.is_file():raise FileNotFoundError(source)
            shutil.copy2(source,figures/(dest+'.'+ext))
    run('figures/make_regional_profile_figures.py','--input',sources,'--output',figures)
    run('figures/write_figure_index.py','--output',out)
    print('Completed analysis outputs in',out,flush=True)

if __name__=='__main__':main()
