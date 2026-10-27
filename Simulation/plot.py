"""Simulation figures from the replicate suite."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from utils.io import load_npz as arrays
from utils.metrics import SIM_METHODS as METHODS, TRUE_K
from utils.plots import COLORS, centered, grid, save

def label(ax, s):
    ax.text(-.14, 1.04, s, transform=ax.transAxes, fontweight="bold", fontsize=11)

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);a=p.parse_args();r=a.root;f=r/'Simulation/figures';f.mkdir(exist_ok=True)
 s=json.loads((r/'Simulation/analysis/summary.json').read_text());assert s['all250_complete'],'Do not render a partial suite as 50 complete replicates'
 df=pd.read_csv(r/'Simulation/analysis/case_metrics.csv');ranks=pd.read_csv(r/'Simulation/analysis/rank_selection.csv');assert len(ranks)==50
 fig,axes=plt.subplots(1,3,figsize=(8.1,2.65));fig.subplots_adjust(wspace=.55,bottom=.25)
 specs=[('composition_correlation','Composition pattern','Mean Pearson correlation',METHODS),('loading_cosine','Gene loadings','Mean gene-centered cosine',['MMF','STAMP','SpiceMix','NMF']),('activity_correlation','Program activity shape','Mean Pearson correlation',['MMF','STAMP','SpiceMix','NMF'])]
 for ax,(metric,title,ylabel,methods),letter in zip(axes,specs,'abc'):
  vals=[df[df.method==m].sort_values('seed')[metric].to_numpy() for m in methods];assert all(len(z)==50 for z in vals)
  bp=ax.boxplot(vals,positions=range(len(methods)),widths=.52,patch_artist=True,showfliers=False,medianprops={'color':'#333333','linewidth':1.1},whiskerprops={'color':'#777777'},capprops={'color':'#777777'})
  for i,(m,v,b) in enumerate(zip(methods,vals,bp['boxes'])):
   b.set_facecolor(COLORS[m]);b.set_alpha(.23);b.set_edgecolor('none');rng=np.random.default_rng(20261027+i)
   ax.scatter(i+rng.uniform(-.2,.2,50),v,color=COLORS[m],s=13,alpha=.67,edgecolors='none',zorder=3)
  ax.set_xticks(range(len(methods)),methods,rotation=32,ha='right');ax.set_title(title,pad=10);ax.set_ylabel(ylabel);ax.set_ylim(min(0.,np.floor(min(v.min() for v in vals)*10)/10-.03),1.04);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);label(ax,letter)
 save(fig,f/'sim_primary50_metrics')
 fig,axes=plt.subplots(1,3,figsize=(8.1,2.7),gridspec_kw={'width_ratios':[1.3,1,1.2]});fig.subplots_adjust(wspace=.55,bottom=.2)
 matrix=np.array([[(ranks[f'K{c+1}']==k).sum() for c in range(5)] for k in range(4)])
 im=axes[0].imshow(matrix,cmap='Blues',vmin=0,vmax=50,origin='lower',aspect='auto')
 for k in range(4):
  for c in range(5):axes[0].text(c,k,str(matrix[k,c]),ha='center',va='center',fontsize=8,color='white' if matrix[k,c]>30 else 'black');
 for c,k in enumerate(TRUE_K):axes[0].add_patch(plt.Rectangle((c-.49,k-.49),.98,.98,fill=False,edgecolor='#D5893B',linewidth=1.7))
 axes[0].set_xticks(range(5),[f'T{i+1}' for i in range(5)]);axes[0].set_yticks(range(4));axes[0].set_ylabel('Selected rank');axes[0].set_title('Counts across 50 replicates');label(axes[0],'a')
 counts=ranks.total_rank.value_counts().sort_index();axes[1].bar(counts.index,counts.values,color=COLORS['MMF'],width=.7);axes[1].axvline(5,color='#D5893B',ls='--',lw=1);axes[1].set_xlabel('Total selected rank');axes[1].set_ylabel('Replicates');axes[1].set_title('True total = 5');label(axes[1],'b')
 correct=ranks.sort_values('seed').exact_vector.to_numpy(dtype=bool).reshape(5,10);axes[2].imshow(correct,cmap=matplotlib.colors.ListedColormap(['#D2D2D2',COLORS['MMF']]),vmin=0,vmax=1,aspect='equal')
 for i in range(5):
  for j in range(10):axes[2].text(j,i,str(i*10+j+1),ha='center',va='center',fontsize=6,color='white' if correct[i,j] else '#333333')
 axes[2].set_xticks([]);axes[2].set_yticks([]);axes[2].set_title(f'Exact vector: {int(correct.sum())}/50');axes[2].set_xlabel(f'Null-type false selections: {int(ranks.null_type_false_positive.sum())}/50',fontsize=7);label(axes[2],'c')
 save(fig,f/'sim_rank_accuracy')
 pd.DataFrame(matrix,index=range(4),columns=[f'T{i+1}' for i in range(5)]).to_csv(r/'Simulation/analysis/rank_frequency_source.csv')
 seed=s['representative']['seed'];inp=r/'data/simu_data/simulation'/f'gradient_{seed}_depth4x';t=arrays(inp/'evaluation_only.npz');d=arrays(inp/'fit_input.npz');coords=d['coords']
 fits={};mapping=pd.read_csv(r/'Simulation/analysis/composition_matching.csv');wmapping=pd.read_csv(r/'Simulation/analysis/program_matching.csv');source={'coords':coords,'true_pi':t['pi_true'],'true_h':t['h_true']}
 for method in METHODS:
  path=r/'Simulation/results'/f'sim_{seed}_{method}'
  fits[method]={'pi':pd.read_csv(path/'proportions.csv',index_col=0)[[f'T{i+1}' for i in range(5)]].to_numpy()} if method=='RCTD' else arrays(path/('final.npz' if method=='MMF' else 'factors.npz'))
 for which in ['composition','activities']:
  methods=METHODS if which=='composition' else [x for x in METHODS if x!='RCTD'];names=['Truth']+methods
  fig,axs=plt.subplots(len(names),5,figsize=(7.2,1.37*len(names)),squeeze=False);fig.subplots_adjust(wspace=.18,hspace=.32,left=.13,right=.98,bottom=.08,top=.96)
  for row,method in enumerate(names):
   for col in range(5):
    ax=axs[row,col];score=None
    if method=='Truth':v=t['pi_true'][:,col] if which=='composition' else t['h_true'][col]
    elif which=='composition':
     m=mapping[(mapping.seed==seed)&(mapping.method==method)&(mapping.type==col+1)].iloc[0];raw=fits[method]['pi'] if method in ['MMF','RCTD'] else fits[method]['activity'];v=raw[:,int(m.factor)];score=m.correlation
    else:
     m=wmapping[(wmapping.seed==seed)&(wmapping.method==method)&(wmapping.true_program==col)].iloc[0];raw=fits[method]['h_mean'] if method=='MMF' else fits[method]['activity'].T;v=np.zeros(len(coords)) if pd.isna(m.factor) else raw[int(m.factor)];score=m.activity_correlation
    source[f'{which}_{method}_{col}']=v;im=ax.imshow(grid(coords,centered(v)),origin='lower',cmap='RdBu_r',vmin=-2.5,vmax=2.5,interpolation='nearest');ax.set_xticks([]);ax.set_yticks([])
    for sp in ax.spines.values():sp.set_visible(False)
    if row==0:ax.set_title(f'Type {col+1}' if which=='composition' else f'T{int(t["program_cell_type"][col])+1} P{int(t["program_index"][col])+1}',fontsize=9)
    if col==0:ax.set_ylabel(method,rotation=0,ha='right',va='center',labelpad=14,fontsize=9)
    if score is not None:ax.set_xlabel(f'r = {score:.3f}',fontsize=8,labelpad=2)
  cax=fig.add_axes([.32,.035,.45,.012]);cb=fig.colorbar(im,cax=cax,orientation='horizontal',extend='both');cb.set_label('Centered map / spatial RMS (display limits ±2.5)',fontsize=8)
  save(fig,f/f'sim_representative_{which}')
 np.savez_compressed(r/'Simulation/analysis/representative_plot_source.npz',**source)
 # Original-scale diagnostic is retained independently of standardized maps.
 fmmf=fits['MMF'];fig,axs=plt.subplots(1,5,figsize=(8.1,2));fig.subplots_adjust(wspace=.45,bottom=.25)
 for col,ax in enumerate(axs):
  m=wmapping[(wmapping.seed==seed)&(wmapping.method=='MMF')&(wmapping.true_program==col)].iloc[0]
  if not pd.isna(m.factor):
   q=int(m.factor);ax.scatter(t['h_true'][col],fmmf['h_mean'][q],s=3,alpha=.4,color=COLORS['MMF']);lim=[min(t['h_true'][col].min(),fmmf['h_mean'][q].min()),max(t['h_true'][col].max(),fmmf['h_mean'][q].max())];ax.plot(lim,lim,color='#999999',lw=.7)
  ax.set_xlabel('True activity');ax.set_title(f'T{int(t["program_cell_type"][col])+1} P{int(t["program_index"][col])+1}')
 axs[0].set_ylabel('Estimated activity');save(fig,f/'sim_representative_native_scale')
 (f/'simulation_figure_audit.json').write_text(json.dumps(dict(n_replicates=50,selected_seed=seed,selection=s['representative'],backend='Python matplotlib',source_metrics='analysis/case_metrics.csv',source_matching='analysis/program_matching.csv',shape_normalization='center then divide spatial RMS; correlations use original unclipped arrays',no_activity_matching_or_sign_flip=True,export_formats=['PDF','SVG','PNG'],native_amplitude_diagnostic='sim_representative_native_scale.pdf'),indent=2))
 print(json.dumps({'figures':5,'representative_seed':seed}))
if __name__=='__main__':main()
