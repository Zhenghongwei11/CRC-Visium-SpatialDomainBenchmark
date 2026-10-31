#!/usr/bin/env python3
"""Recreate the published real-cell reference from the public GSE132465 raw UMI table."""
import argparse,gzip,hashlib,re,shutil,tempfile
from pathlib import Path
import numpy as np,pandas as pd

ROOT=Path(__file__).resolve().parents[1]
RAW_SHA256='deff856c7fe1d2c845af5b31e1c65a1dd95ea29541772a66c89bda3aaa8c0666'
def digest(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--raw-umi',type=Path,required=True,help='GSE132465 GEO raw UMI count matrix .txt.gz')
    a.add_argument('--output',type=Path,required=True)
    a.add_argument('--selection',type=Path,default=ROOT/'data/reference_inputs/single_cell/selected_reference_cells.tsv',help='Cell selection generated from source annotation')
    args=a.parse_args()
    if digest(args.raw_umi)!=RAW_SHA256:raise ValueError('The source raw UMI file does not match the published input')
    expected_selection=ROOT/'data/reference_inputs/single_cell/selected_reference_cells.tsv'
    if digest(args.selection)!=digest(expected_selection):raise ValueError('Reference-cell selection differs from the published selection')
    selected=pd.read_csv(args.selection,sep='\t',dtype={'Index':str})
    if len(selected)!=1400 or selected.Index.duplicated().any():raise ValueError('Invalid reference-cell selection')
    if args.output.exists() and any(args.output.iterdir()):raise FileExistsError('Choose an empty output directory')
    args.output.mkdir(parents=True,exist_ok=True)
    genes=[];totals=np.zeros(len(selected),dtype=np.int64);detected=totals.copy();nnz=0
    with tempfile.TemporaryDirectory(prefix='reference_matrix_',dir=args.output) as tmp:
      body=Path(tmp)/'matrix-body.gz'
      with gzip.open(args.raw_umi,'rt') as f, gzip.open(body,'wt',compresslevel=3) as z:
        header=f.readline().rstrip('\r\n').split('\t')
        if header[0]!='Index' or len(set(header[1:]))!=len(header)-1:raise ValueError('Unexpected source cell identifiers')
        selected_idx=pd.Index(header[1:]).get_indexer(selected.Index)
        if (selected_idx<0).any():raise ValueError('Selected cells missing from source data')
        for i,line in enumerate(f,1):
            gene,values=line.rstrip('\r\n').split('\t',1)
            x=np.fromstring(values,sep='\t',dtype=np.int64)
            if not re.fullmatch(r'[0-9\t]+',values) or len(x)!=len(header)-1 or (x<0).any():raise ValueError('Invalid UMI matrix row '+str(i))
            y=x[selected_idx];totals+=y;detected+=y>0;idx=np.flatnonzero(y)
            z.writelines(f'{i} {j+1} {y[j]}\n' for j in idx);nnz+=len(idx);genes.append(gene)
            if i%2000==0:print('Genes processed:',i,flush=True)
      with gzip.open(args.output/'GSE132465_real_cell_subset.mtx.gz','wt',compresslevel=6) as o, gzip.open(body,'rt') as b:
        o.write(f'%%MatrixMarket matrix coordinate integer general\n% Genuine source UMI columns selected according to the reference cell list.\n{len(genes)} {len(selected)} {nnz}\n');shutil.copyfileobj(b,o)
    selected['total_UMI']=totals;selected['detected_genes']=detected
    selected.to_csv(args.output/'GSE132465_real_cell_subset_labels.tsv.gz',sep='\t',index=False)
    with gzip.open(args.output/'GSE132465_real_cell_subset_genes.tsv.gz','wt') as f:f.writelines(x+'\n' for x in genes)
    with gzip.open(args.output/'GSE132465_real_cell_subset_barcodes.tsv.gz','wt') as f:f.writelines(x+'\n' for x in selected.Index)
    print('Selected cells:',len(selected),'genes:',len(genes),'nonzero entries:',nnz,flush=True)
if __name__=='__main__':main()
