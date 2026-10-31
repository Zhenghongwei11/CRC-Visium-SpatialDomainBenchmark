#!/usr/bin/env python3
"""Compare all 14 PNG exports with the archived reference renders."""
import argparse,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--figures',type=Path,required=True);a=p.parse_args();expected=ROOT/'expected/figures'
    names=['figure1','figure2','figure3',*[f'figureS{i}' for i in range(1,12)]];different=[]
    for name in names:
        fname=name+'.png';ref=expected/fname;observed=a.figures/fname
        if not ref.is_file() or not observed.is_file():different.append(fname+': missing file')
        elif sha(ref)!=sha(observed):different.append(fname+': PNG bytes differ')
    if different:
        print('\n'.join(different));print('Pixel rendering can vary across software and platforms.');raise SystemExit(1)
    print('All 14 PNG exports match the archived reference bytes.')
if __name__=='__main__':main()
