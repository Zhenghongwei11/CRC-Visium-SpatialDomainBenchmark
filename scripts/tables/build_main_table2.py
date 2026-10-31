#!/usr/bin/env python3
"""Produce the same-patient complete versus composition-calibrated regional summary."""
import argparse,csv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
TIERS=[('composition_means','Composition means'),('composition_and_qc_means','Composition + QC means'),('composition_second_moments_and_qc','Composition means/second moments + QC')]
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,default=ROOT/'data/table_inputs/composition_sensitivity/rctd_continuous_profiles/same_common_patient_resource_summary.tsv');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 with a.input.open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f,delimiter='\t'))
 out=[]
 for resource in ['Valdeolivas','GSE294385']:
  for tier,label in TIERS:
   common=[r for r in rows if r['resource']==resource and r['tier']==tier and r['metric']=='scaled_abs_difference']
   by_control={r['control']:r for r in common}
   if not {'unweighted_full_regions','frozen_mean_calibrated_weights'}<=by_control.keys():raise ValueError(f'Missing control in {resource}, {tier}')
   u=by_control['unweighted_full_regions'];w=by_control['frozen_mean_calibrated_weights']
   if u['same_common_patient_ids']!=w['same_common_patient_ids'] or u['n_same_common_patients']!=w['n_same_common_patients']:raise ValueError('Patient sets differ')
   out.append([resource,label,int(w['n_same_common_patients']),f"{float(u['median_patient_far_minus_near']):+.3f}",f"{float(w['median_patient_far_minus_near']):+.3f}"])
 a.output.parent.mkdir(parents=True,exist_ok=True)
 with a.output.open('w',encoding='utf-8',newline='') as f:
  w=csv.writer(f,delimiter='\t',lineterminator='\n');w.writerow(['Resource','Calibration variables','Patients','Complete bands','Weighted bands']);w.writerows(out)
 print('Created',a.output)
if __name__=='__main__':main()
