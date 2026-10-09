#!/usr/bin/env python3
"""Draw original Patient6 annotation coverage and its measured stromal bands."""
from pathlib import Path
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from jbcb_presentation_paths import read
from jbcb_presentation_paths import default_package

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pack-dir',type=Path,default=default_package(__file__))
    args=parser.parse_args(); package=args.pack_dir
    data=read(package/'tables/figure_source_data/figureS7_spots.tsv.gz',sep='\t')
    if len(data)!=9530 or data.barcode.duplicated().any(): raise ValueError('Patient6 spot registry differs')
    known=data.public_raw_present.astype(bool).to_numpy()
    if known.sum()!=7858 or not np.array_equal(known,data.public_qc_present.astype(bool)): raise ValueError('Original annotation coverage differs')
    x=(data.array_col-data.array_col.min())*.05
    y=(data.array_row-data.array_row.min())*np.sqrt(3)*.05
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
                         'font.size':9,'axes.labelsize':9,'axes.titlesize':10,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(7.15,5.0))
    fig.subplots_adjust(left=.07,right=.985,top=.85,bottom=.29,wspace=.16)
    colors={'tumor':'#D55E00','stroma':'#009E73','normal epithelium':'#0072B2',
            'unresolved':'#B7B7B7','mixed':'#CC79A7','exclude':'#E69F00'}
    handles=[]
    for label in ['tumor','stroma','normal epithelium','unresolved','mixed','exclude']:
        mask=known & data.coarse_label.eq(label).to_numpy()
        if not mask.any(): continue
        axes[0].scatter(x[mask],y[mask],s=1.8,c=colors[label],linewidths=0,rasterized=True)
        handles.append(Line2D([],[],marker='o',ls='',color=colors[label],markersize=4,
                              label=f'{label.capitalize()} ({mask.sum():,})'))
    axes[0].scatter(x[~known],y[~known],s=4,c='#332755',marker='x',linewidths=.35,rasterized=True)
    handles.append(Line2D([],[],marker='x',ls='',color='#332755',markersize=4,label='No source label (1,672)'))
    axes[0].set_title('Deposited annotation classes',pad=10)
    axes[0].legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,-.20),ncol=2,
                   fontsize=8,frameon=False,handletextpad=.35,columnspacing=.8)
    axes[1].scatter(x,y,s=1.8,c='#DADADA',linewidths=0,rasterized=True)
    axes[1].scatter(x[~known],y[~known],s=4,c='#332755',marker='x',linewidths=.35,rasterized=True)
    near=data.primary_anchor_near.astype(bool).to_numpy();far=data.primary_anchor_far.astype(bool).to_numpy()
    if (near.sum(),far.sum())!=(291,84): raise ValueError('Current stromal-band memberships differ')
    for mask,color in [(near,'#D55E00'),(far,'#0072B2')]:
        axes[1].scatter(x[mask],y[mask],s=5,c=color,linewidths=0,rasterized=True)
    axes[1].set_title('Stromal bands used for measurement',pad=10)
    axes[1].legend(handles=[Line2D([],[],marker='o',ls='',color='#D55E00',markersize=4,label='Near stroma (291)'),
                           Line2D([],[],marker='o',ls='',color='#0072B2',markersize=4,label='Far stroma (84)'),
                           Line2D([],[],marker='x',ls='',color='#332755',markersize=4,label='No source label (1,672)')],
                   loc='upper center',bbox_to_anchor=(.5,-.20),fontsize=8,frameon=False)
    for index,ax in enumerate(axes):
        ax.set_aspect('equal');ax.invert_yaxis();ax.set_xlabel('Nominal array position (mm)')
        ax.set_xticks([0,4,8]);ax.set_yticks([0,4,8]);ax.set_ylim(y.max()+.1,-.1)
        ax.spines[['top','right']].set_visible(False)
        ax.text(-.06,1.10,'AB'[index],transform=ax.transAxes,fontweight='bold',fontsize=12)
    axes[0].set_ylabel('Nominal array position (mm)')
    fig.suptitle('Patient6 (M-ST-15): original annotation coverage',fontsize=11,y=.97)
    figures=package/'figures';figures.mkdir(exist_ok=True)
    for extension in ['png','pdf','tiff']:
        kwargs={'pil_kwargs':{'compression':'tiff_lzw'}} if extension=='tiff' else {}
        fig.savefig(figures/f'figureS7.{extension}',dpi=600,**kwargs)
    plt.close(fig)

if __name__=='__main__':main()
