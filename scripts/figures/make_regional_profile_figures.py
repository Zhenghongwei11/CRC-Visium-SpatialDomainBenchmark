#!/usr/bin/env python3
"""Draw patient-level figures from regional result tables."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'data/table_inputs'
OUT=ROOT/'figure_replay_output'
COLORS={'Valdeolivas':'#0072B2','GSE294385':'#D55E00'}
plt.rcParams.update({'font.family':'Arial','font.size':9,'axes.labelsize':9,'axes.titlesize':10,
 'xtick.labelsize':8,'ytick.labelsize':8,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,
 'axes.spines.right':False,'savefig.facecolor':'white'})

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def label(ax,letter,title):
    ax.set_title(title,loc='left',pad=12)
    ax.text(-.16,1.09,letter,transform=ax.transAxes,fontsize=12,fontweight='bold',va='top')

def save(fig,name,sources):
    fig.savefig(OUT/(name+'.pdf'))
    fig.savefig(OUT/(name+'.png'),dpi=300)
    fig.savefig(OUT/(name+'.tiff'),dpi=600,pil_kwargs={'compression':'tiff_lzw'})
    plt.close(fig)
    return {'sources':{p.relative_to(BASE).as_posix():sha(p) for p in sources},'exports':{name+ext:sha(OUT/(name+ext)) for ext in ['.png','.pdf','.tiff']}}

def main():
    global BASE, OUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=BASE,help='Directory containing regional_profiles, pure_domain_selection and inferred_composition tables')
    parser.add_argument('--output',type=Path,default=OUT,help='Directory for generated figures')
    args=parser.parse_args(); BASE=args.input.resolve(); OUT=args.output.resolve()
    OUT.mkdir(parents=True,exist_ok=True);receipts={}
    p=BASE/'regional_profiles'
    within=pd.read_csv(p/'within_patient_summaries.tsv',sep='\t')
    matched=pd.read_csv(p/'matched_sampling_patient_summaries.tsv',sep='\t')
    primary=within[(within.scope=='all_eligible')&(within.geometry=='exacthex')&(within.compartment=='stroma')&(within.panel=='broad')&(within.metric=='scaled_abs_difference')]
    fig,axes=plt.subplots(2,2,figsize=(7.2,6.2));fig.subplots_adjust(left=.13,right=.98,bottom=.10,top=.89,wspace=.43,hspace=.62)
    controls=[('original','#0072B2','o'),('spot_matched','#666666','^'),('spot_and_depth_matched','#D55E00','D')]
    for ax,resource,letter in zip(axes[0],['Valdeolivas','GSE294385'],'AB'):
        group=primary[primary.resource==resource].sort_values('patient_id')
        for y,(_,r) in enumerate(group.iterrows()):
            for control,color,marker in controls:
                if control=='original':value=r.far_minus_near
                else:
                    m=matched[(matched.resource==resource)&(matched.compartment=='stroma')&(matched.patient_id==r.patient_id)&(matched.panel=='broad')&(matched.metric=='scaled_abs_difference')&(matched.control==control)]
                    assert len(m)==1;value=m.iloc[0].far_minus_near
                ax.scatter(value,y,s=33,c=color,marker=marker,zorder=4)
        ax.axvline(0,color='#bbbbbb',lw=.8);ax.set_yticks(range(len(group)),group.patient_id)
        ax.set_ylim(-.6,len(group)-.4);ax.set_xlim(-.15,1.12)
        ax.set_xlabel('Far − near profile difference');label(ax,letter,resource+' stroma')
    handles=[Line2D([0],[0],marker=m,color='none',markerfacecolor=c,markeredgecolor=c,label=n) for (control,c,m),n in zip(controls,['Original bands','Equal spots','Equal spots and UMIs'])]
    fig.legend(handles=handles,ncol=3,loc='upper center',bbox_to_anchor=(.54,1.0),frameon=False,fontsize=8.5)
    joint=pd.read_csv(p/'cross_compartment_patient_comparison.tsv',sep='\t');joint=joint[(joint.panel=='broad')&(joint.metric=='scaled_abs_difference')]
    ax=axes[1,0]
    for resource,g in joint.groupby('resource'):
        ax.scatter(g.stroma,g.tumor,s=38,c=COLORS[resource],marker='o' if resource=='Valdeolivas' else 's',label=resource)
    ax.axhline(0,color='#bbbbbb',lw=.8);ax.axvline(0,color='#bbbbbb',lw=.8)
    ax.set_xlabel('Stroma far − near difference');ax.set_ylabel('Tumor far − near difference')
    ax.legend(frameon=False,fontsize=7.5,loc='best');label(ax,'C','Same-patient compartment comparison')
    between=pd.read_csv(p/'between_patient_resource_summary.tsv',sep='\t');lopo=pd.read_csv(p/'between_patient_leave_one_out.tsv',sep='\t')
    between=between[(between.scope=='all_eligible')&(between.geometry=='exacthex')&(between.panel=='broad')&(between.metric=='scaled_abs_difference')]
    ax=axes[1,1];labels=[]
    for y,(resource,compartment) in enumerate([(r,c) for r in ['Valdeolivas','GSE294385'] for c in ['stroma','tumor']]):
        row=between[(between.resource==resource)&(between.compartment==compartment)].iloc[0]
        leave=lopo[(lopo.scope=='all_eligible')&(lopo.geometry=='exacthex')&(lopo.panel=='broad')&(lopo.metric=='scaled_abs_difference')&(lopo.resource==resource)&(lopo.compartment==compartment)].difference_of_pair_medians
        ax.plot([leave.min(),leave.max()],[y,y],color=COLORS[resource],lw=1.3)
        ax.scatter(row.difference_of_pair_medians,y,c=COLORS[resource],marker='o' if compartment=='stroma' else 's',s=35,zorder=3)
        labels.append(('Val' if resource=='Valdeolivas' else 'GSE')+' '+compartment)
    ax.axvline(0,color='#bbbbbb',lw=.8);ax.set_yticks(range(4),labels);ax.set_ylim(-.6,3.6)
    ax.set_xlabel('Far − near between-patient distance');label(ax,'D','Between-patient variation')
    receipts['figure3']=save(fig,'figure3',[p/n for n in ['within_patient_summaries.tsv','matched_sampling_patient_summaries.tsv','cross_compartment_patient_comparison.tsv','between_patient_resource_summary.tsv','between_patient_leave_one_out.tsv']])
    dp=BASE/'pure_domain_selection';coverage=pd.read_csv(dp/'coverage.tsv',sep='\t');summary=pd.read_csv(dp/'resource_coverage.tsv',sep='\t')
    fig=plt.figure(figsize=(7.2,6.1));grid=fig.add_gridspec(3,2,height_ratios=[1,.34,1],hspace=.35,wspace=.36)
    axes=np.array([[fig.add_subplot(grid[0,col]) for col in range(2)],[fig.add_subplot(grid[2,col]) for col in range(2)]])
    legend_axis=fig.add_subplot(grid[1,:]);legend_axis.axis('off')
    fig.subplots_adjust(left=.10,right=.94,bottom=.10,top=.93)
    for col,resource in enumerate(['Valdeolivas','GSE294385']):
        ax=axes[0,col]
        for method,color,marker in [('expression_kmeans','#0072B2','o'),('exacthex_spatial_ward','#D55E00','s')]:
            for q,style in [(.25,'--'),(.5,'-')]:
                g=summary[(summary.resource==resource)&(summary.method==method)&(summary.merge_quantile==q)].sort_values('K')
                ax.plot(g.K,g.eligible_patients,style,c=color,marker=marker,markersize=4)
        ax.set_xticks([12,20,30]);ax.set_ylim(-.2,7.5 if resource=='Valdeolivas' else 5.5);ax.set_yticks(range(8 if resource=='Valdeolivas' else 6));ax.set_xlabel('Nominal initial K');ax.set_ylabel('Patients with a regional comparison')
        label(ax,'AB'[col],resource+' ≥90% stromal domains')
        ax.texts[-1].set_position((-.22,1.09))
        ax=axes[1,col];g=coverage[(coverage.resource==resource)&(coverage.selected_region_size>0)].copy()
        g['near_fraction']=g.n_selected_near/g.n_full_near.replace(0,np.nan);g['far_fraction']=g.n_selected_far/g.n_full_far.replace(0,np.nan)
        for method,marker in [('expression_kmeans','o'),('exacthex_spatial_ward','s')]:
            for ok in [False,True]:
                z=g[(g.method==method)&(g.primary_evaluable==ok)]
                ax.scatter(z.near_fraction,z.far_fraction,color='#0072B2' if method=='expression_kmeans' else '#D55E00',marker=marker,s=26 if ok else 17,alpha=.9 if ok else .28,edgecolors='black' if ok else 'none',linewidths=.4)
        ax.set(xlim=(-.03,1.03),ylim=(-.03,1.03),xlabel='Near-band spots retained',ylabel='Far-band spots retained')
        label(ax,'CD'[col],resource+' regional coverage')
        ax.texts[-1].set_position((-.22,1.09))
    legend_axis.legend(handles=[Line2D([0],[0],color='#0072B2',marker='o',label='k-means'),Line2D([0],[0],color='#D55E00',marker='s',label='Spatial Ward'),Line2D([0],[0],color='#555555',ls='--',label='25th percentile'),Line2D([0],[0],color='#555555',ls='-',label='50th percentile')],loc='center',ncol=4,frameon=False,fontsize=7.5)
    receipts['figureS8']=save(fig,'figureS8',[dp/'coverage.tsv',dp/'resource_coverage.tsv'])
    comp=BASE/'inferred_composition/regional_group_composition.tsv'
    if not comp.exists():raise FileNotFoundError(comp)
    if comp.exists():
        d=pd.read_csv(comp,sep='\t');d=d[(d.arm=='mean')&(d.compartment=='stroma')&d.evaluable]
        avg=d.groupby(['patient_id','band','group']).relative_model_share.mean()
        patients=sorted(d.patient_id.unique());groups=['fibroblast_like','smooth_muscle_like','pericyte_like','endothelial','immune','epithelial_or_proliferating','other_or_unknown']
        palette=['#009E73','#E69F00','#CC79A7','#56B4E9','#0072B2','#D55E00','#999999']
        fig=plt.figure(figsize=(7.2,7.0));grid=fig.add_gridspec(3,1,height_ratios=[1.35,.45,1],hspace=.45)
        axes=[fig.add_subplot(grid[0]),fig.add_subplot(grid[2])];legend_axis=fig.add_subplot(grid[1]);legend_axis.axis('off')
        fig.subplots_adjust(left=.16,right=.95,bottom=.10,top=.94)
        ax=axes[0];bottom=np.zeros(len(patients)*2);x=np.arange(len(bottom));names=[]
        for patient in patients:names.extend([patient+' N',patient+' F'])
        for group,color in zip(groups,palette):
            vals=np.array([avg.loc[(patient,band,group)] for patient in patients for band in ['near','far']]);ax.bar(x,vals,bottom=bottom,color=color,width=.78,label=group.replace('_',' '));bottom+=vals
        assert np.allclose(bottom,1);ax.set(ylim=(0,1),ylabel='Relative modeled abundance');ax.set_xticks(x,names,rotation=35,ha='right');label(ax,'A','Model-inferred composition of annotated stroma')
        handles,names_for_legend=ax.get_legend_handles_labels();legend_axis.legend(handles,names_for_legend,ncol=3,loc='center',frameon=False,fontsize=7.3)
        pr=pd.read_csv(comp.parent/'within_patient_composition_and_rna.tsv',sep='\t');pr=pr[(pr.arm=='mean')&(pr.compartment=='stroma')]
        ax=axes[1]
        for i,(_,r) in enumerate(pr.iterrows()):
            ax.scatter(r.composition_L1_far_minus_near,r.rna_matched_spots_far_minus_near,color=palette[i],s=40)
            right=r.composition_L1_far_minus_near>.08
            ax.annotate(r.patient_id,(r.composition_L1_far_minus_near,r.rna_matched_spots_far_minus_near),xytext=(-5 if right else 5,3),ha='right' if right else 'left',textcoords='offset points',fontsize=8)
        ax.axhline(0,color='#bbbbbb',lw=.8);ax.axvline(0,color='#bbbbbb',lw=.8)
        ax.set_xlabel('Far − near composition-profile difference (L1)');ax.set_ylabel('Far − near RNA-profile difference');label(ax,'B','Paired-section composition and expression')
        receipts['figureS9']=save(fig,'figureS9',[comp,comp.parent/'within_patient_composition_and_rna.tsv'])
    (OUT/'FIGURE_SOURCES.json').write_text(json.dumps(receipts,indent=2)+'\n')
    print(json.dumps({k:list(v['exports']) for k,v in receipts.items()},indent=2))

if __name__=='__main__':main()
