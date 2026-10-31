#!/usr/bin/env python3
"""Index all generated figures, their source data and producing scripts."""
import argparse,hashlib,json
from pathlib import Path

def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=a.output.resolve();fig=root/'figures';source=root/'baseline/tables/figure_source_data'
    mapping={'figure1':'make_jbcb_corrected_tissue_figures.py','figure2':'make_jbcb_corrected_main_figures.py','figureS1':'make_jbcb_corrected_support_figure.py','figureS2':'make_jbcb_corrected_main_figures.py','figureS3':'make_jbcb_corrected_sensitivity_figures.py','figureS4':'make_jbcb_corrected_sensitivity_figures.py','figureS5':'make_jbcb_corrected_tissue_figures.py','figureS6':'make_jbcb_corrected_graph_figure.py','figureS7':'make_jbcb_source_coverage_figure.py','figureS8':'make_jbcb_readout_withholding_figure.py','figureS11':'make_jbcb_corrected_main_figures.py'}
    regional=json.loads((fig/'FIGURE_SOURCES.json').read_text());records={}
    for name in ['figure1','figure2','figure3',*[f'figureS{i}' for i in range(1,12)]]:
        exports={f'{name}.{ext}':sha(fig/f'{name}.{ext}') for ext in ('png','pdf','tiff')}
        if name in regional:
            if exports != regional[name]['exports']:raise ValueError('Figure exports differ from their source receipt: '+name)
            src={f'table_sources/{path}':digest for path,digest in regional[name]['sources'].items()}
            for path,digest in src.items():
                if sha(root/path)!=digest:raise ValueError('Figure source changed: '+path)
            producer='make_regional_profile_figures.py'
        else:
            original='figure3' if name=='figureS11' else name
            files=sorted(p for p in source.glob(original+'*') if p.is_file())
            if not files:raise FileNotFoundError('No figure source data for '+name)
            src={str(p.relative_to(root)):sha(p) for p in files};producer=mapping[name]
        records[name]={'producer':'scripts/figures/'+producer,'source_data':src,'exports':exports}
    (fig/'FIGURE_INDEX.json').write_text(json.dumps(records,indent=2)+'\n')
    print('Indexed',len(records),'figures')
if __name__=='__main__':main()
