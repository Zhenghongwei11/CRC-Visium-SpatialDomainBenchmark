#!/usr/bin/env python3
"""Retrieve the public GSE132465 processed raw-UMI matrix with identity verification."""
import argparse,hashlib,os,tempfile,urllib.request
from pathlib import Path
URL='https://ftp.ncbi.nlm.nih.gov/geo/series/GSE132nnn/GSE132465/suppl/GSE132465_GEO_processed_CRC_10X_raw_UMI_count_matrix.txt.gz'
SHA='deff856c7fe1d2c845af5b31e1c65a1dd95ea29541772a66c89bda3aaa8c0666'
SIZE=128941455
ANNOTATION_URL='https://ftp.ncbi.nlm.nih.gov/geo/series/GSE132nnn/GSE132465/suppl/GSE132465_GEO_processed_CRC_10X_cell_annotation.txt.gz'
ANNOTATION_SHA='4bf2ebdadc88da53b0c46e654a6576e841925addc25c981d7cdbc0d988b4677c'
NAME='GSE132465_GEO_processed_CRC_10X_raw_UMI_count_matrix.txt.gz'
def identify(p):
 with p.open('rb') as f:return p.stat().st_size,hashlib.file_digest(f,'sha256').hexdigest()
def main():
 a=argparse.ArgumentParser(description=__doc__);a.add_argument('--cache',type=Path,required=True);a.add_argument('--annotation',action='store_true',help='Fetch the smaller cell-annotation file instead of the UMI matrix');p=a.parse_args();p.cache.mkdir(parents=True,exist_ok=True)
 url=ANNOTATION_URL if p.annotation else URL;expected_sha=ANNOTATION_SHA if p.annotation else SHA;expected_size=None if p.annotation else SIZE;dest=p.cache/url.rsplit('/',1)[-1]
 def valid(path):
  size,sha=identify(path);return sha==expected_sha and (expected_size is None or size==expected_size)
 if dest.exists():
  if not valid(dest):raise ValueError('Existing reference file has unexpected identity')
  print('Verified:',dest);return
 fd,tmp=tempfile.mkstemp(prefix='gse132465-',suffix='.part',dir=p.cache);path=Path(tmp)
 try:
  with os.fdopen(fd,'wb') as out,urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'CRC-spatial-reproduction'}),timeout=90) as src:
   while chunk:=src.read(1024*1024):out.write(chunk)
  if not valid(path):raise ValueError('Downloaded reference file has unexpected identity')
  path.replace(dest);print('Verified:',dest)
 finally:
  if path.exists():path.unlink()
if __name__=='__main__':main()
