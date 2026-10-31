#!/usr/bin/env python3
"""Generate supplementary workbooks from the reported scientific source tables."""
import argparse,csv,gzip,json,re
from decimal import Decimal,InvalidOperation
from functools import lru_cache
from pathlib import Path
from openpyxl import Workbook
from openpyxl.compat.strings import safe_string
from openpyxl.cell import WriteOnlyCell
from openpyxl.styles import Font,PatternFill,Alignment

ROOT=Path(__file__).resolve().parents[2]
NUMBER=re.compile(r'^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$')
IDENTIFIER=re.compile(r'(?:^|_)(?:id|barcode|identifier|sha256|hash|key|label|symbol|gene|name|path|file|status|frame|rule|graph|setting|method|resource|patient|section|sample)(?:$|_)',re.I)
@lru_cache(maxsize=50000)
def convert(value,identifier=False):
    if identifier or not NUMBER.fullmatch(value):return value,'s'
    try:d=Decimal(value)
    except InvalidOperation:return value,'s'
    if not d.is_finite() or (d.is_zero() and d.is_signed()) or re.fullmatch(r'[+-]?0\d+',value):return value,'s'
    if len(d.normalize().as_tuple().digits)>15:return value,'s'
    n=float(d)
    if Decimal(safe_string(n))!=d:return value,'s'
    return (int(d),'n') if d==d.to_integral_value() and abs(d)<10**15 else (n,'n')
def make_cell(ws,value,kind='s',header=False):
    c=WriteOnlyCell(ws,value=value)
    if value is not None:c.data_type=kind
    if header:
        c.font=Font(name='Arial',size=10,bold=True,color='FFFFFF');c.fill=PatternFill('solid',fgColor='234963');c.alignment=Alignment(wrap_text=True,vertical='top')
    return c

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,default=ROOT/'data/table_inputs');p.add_argument('--output',type=Path,required=True);p.add_argument('--tables',type=int,nargs='*',default=list(range(1,20)));a=p.parse_args()
    spec=json.loads((ROOT/'config/table_workbooks.json').read_text());a.output.mkdir(parents=True,exist_ok=True)
    for number in a.tables:
        book=Workbook(write_only=True)
        for item in spec[str(number)]:
            ws=book.create_sheet(item['name']);ws.freeze_panes='A7'
            if item['kind']=='metadata':
                for row in item['rows']:ws.append(row)
                continue
            path=a.input/item['source'];opener=gzip.open if path.suffix=='.gz' else open
            for row in item['preamble']:ws.append(row)
            with opener(path,'rt',encoding='utf-8',newline='') as f:
                reader=csv.reader(f,delimiter='\t');header=next(reader)
                wanted=[x for x in item['header'] if x is not None]
                if header!=wanted:raise ValueError(f'Header mismatch in {path}')
                ws.append([make_cell(ws,x,header=True) for x in header]);flags=[bool(IDENTIFIER.search(x)) for x in header]
                for row in reader:
                    if len(row)!=len(header):raise ValueError(f'Wrong column count in {path}')
                    ws.append([make_cell(ws,*convert(v,i)) for v,i in zip(row,flags)])
        dest=a.output/f'Table_S{number}.xlsx';book.save(dest);print('Created',dest,flush=True)
if __name__=='__main__':main()
