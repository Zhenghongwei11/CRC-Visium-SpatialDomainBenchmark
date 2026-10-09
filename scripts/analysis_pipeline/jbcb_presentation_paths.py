"""Locate scientific outputs and read their published field names."""
from pathlib import Path
import hashlib,csv
import pandas as pd

def default_package(script):
    return Path(script).resolve().parents[2]/'results'

def default_source_root(package,script):
    return Path(package)/'workspace'

def portable_label(path,package=None,source_root=None):
    for root,prefix in [(package,''),(source_root,'workspace/')]:
        if root is not None:
            try:return prefix+Path(path).resolve().relative_to(Path(root).resolve()).as_posix()
            except ValueError:pass
    return Path(path).name

def sha256(path):
    with Path(path).open('rb') as h:return hashlib.file_digest(h,'sha256').hexdigest()

def read(path,**kwargs):
    try:frame=pd.read_csv(path,**{'sep':'\t','float_precision':'round_trip',**kwargs})
    except pd.errors.EmptyDataError:return pd.DataFrame()
    for parent in Path(path).resolve().parents:
        m=parent/'TABLE_FIELD_NAME_MAP.tsv'
        if m.is_file():
            with m.open(newline='') as h:rows=list(csv.DictReader(h,delimiter='\t'))
            columns={r['submission_name']:r['original_name'] for r in rows if r['name_type']=='field'}
            values={r['submission_name']:r['original_name'] for r in rows if r['name_type']=='category'}
            return frame.rename(columns=columns).replace(values)
    return frame

def unique(frame,label):
    if frame.barcode.astype(str).duplicated().any():raise ValueError('Duplicate barcode: '+label)
    return frame.assign(barcode=frame.barcode.astype(str)).set_index('barcode')
