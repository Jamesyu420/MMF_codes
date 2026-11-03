"""Redraw the submitted PDFs from explicit frozen source-data paths."""
import argparse
from pathlib import Path
import csv, json, hashlib
from itertools import permutations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import FuncFormatter, MaxNLocator

COLORS={'MMF':'#25627D','RCTD':'#82AE9E','STAMP':'#CC956B','SpiceMix':'#9B8BB9','NMF':'#A8AAAC'}
GREY='#d6dadd';INK='#22313b'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,
 'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,
 'pdf.fonttype':42,'svg.fonttype':'none','text.color':INK,'axes.labelcolor':INK})
RECORD={}

def save(fig,name):
    fig.canvas.draw()
    for ext in ['pdf','svg','png']:
        fig.savefig(OUTPUT/f'{name}.{ext}',dpi=240,bbox_inches='tight',pad_inches=.06,facecolor='white')
    plt.close(fig)
    print('Saved',name,flush=True)

def linear_sum_assignment(cost):
    nr,nc=cost.shape
    if nr>nc:
        cc,rr=linear_sum_assignment(cost.T);order=np.argsort(rr);return rr[order],cc[order]
    best=min(permutations(range(nc),nr),key=lambda p:sum(cost[i,j] for i,j in enumerate(p)))
    return np.arange(nr),np.asarray(best)

def tag(fig,x,y,s):fig.text(x,y,s,fontsize=15,fontweight='bold',va='top',color='black')

def ids(side,c):return np.flatnonzero(F[side]['program_cell_type']==c)

def lim(c):
    return max([float(np.abs(f['h_mean'][ids(s,c)][:,f['pi'][:,c]>.05]).max())
      for s,f in enumerate(F) if len(ids(s,c))]+[1.])

def bare_map(ax,side,values,*,c=None,limit=None,proportion=False,cmap=None):
    f=F[side];xy=f['coords'];mask=np.ones(len(xy),dtype=bool) if c is None else f['pi'][:,c]>.05
    ax.scatter(*xy.T,c=GREY,s=3.2,linewidths=0,rasterized=True)
    if proportion:mask[:]=True
    im=ax.scatter(*xy[mask].T,c=values[mask],cmap=cmap or ('viridis' if proportion else 'RdBu_r'),
      vmin=0 if proportion else -limit,vmax=1 if proportion else limit,s=3.2,linewidths=0,rasterized=True)
    ax.set_aspect('equal');ax.invert_yaxis();ax.axis('off');return im

def cbar(fig,im,ax,proportion=False,limit=None,label=None):
    cb=fig.colorbar(im,ax=ax,fraction=.035,pad=.025,shrink=.8,
       ticks=[0,.5,1] if proportion else [-limit,0,limit])
    cb.ax.set_title(label or (r'$\pi$' if proportion else r'$h$'),fontsize=10,pad=4)
    cb.ax.tick_params(labelsize=8,length=2,width=.5);cb.outline.set_linewidth(.4)
    if not proportion:cb.ax.set_yticklabels([f'{-limit:.3g}','0',f'{limit:.3g}'])
    return cb

def align(c):
    ii=[ids(s,c) for s in (0,1)]
    a,b=[F[s]['loadings'][ii[s]].astype(float) for s in (0,1)]
    a-=a.mean(1,keepdims=True);b-=b.mean(1,keepdims=True)
    corr=a@b.T/(np.linalg.norm(a,axis=1)[:,None]*np.linalg.norm(b,axis=1)[None,:])
    rr,cc=linear_sum_assignment(-corr);mapping={int(ii[0][i]):(int(ii[1][j]),float(corr[i,j])) for i,j in zip(rr,cc)}
    used={x[0] for x in mapping.values()}
    slots=[(int(q),*mapping.get(int(q),(None,None))) for q in ii[0]]
    slots += [(None,int(q),None) for q in ii[1] if q not in used]
    return slots

def fig4():
    names=['Inhibitory','Ext_L23','HippocampalExc','OtherExc','Neuroblast']
    fig=plt.figure(figsize=(12,8.5));cols=[(.17,.53),(.59,.95)];top=.93;bottom=.025;rh=(top-bottom)/len(names)
    for s,(a,b) in enumerate(cols):fig.text((a+b)/2,.98,f'Section {s+1}',ha='center',va='top',fontsize=14,weight='bold')
    records=[]
    for row,name in enumerate(names):
        c=T.index(name);slots=align(c);bound=lim(c);yb=top-(row+1)*rh
        fig.text(.005,yb+rh*.50,chr(97+row),fontsize=14,weight='bold',va='center')
        fig.text(.035,yb+rh*.50,name,fontsize=10.5,weight='bold',va='center')
        for side,(x0,x1) in enumerate(cols):
            slotw=(x1-x0)/len(slots)
            for j,slot in enumerate(slots):
                q=slot[side];mid=x0+(j+.5)*slotw
                if q is None:
                    fig.text(mid,yb+rh*.50,'Unmatched',ha='center',va='center',fontsize=8.5,color='#697782');continue
                width=min(slotw*.93,.13)
                ax=fig.add_axes([mid-width/2,yb+.012,width,rh-.035])
                im=bare_map(ax,side,F[side]['h_mean'][q],c=c,limit=bound)
                label=f'P{int(F[side]["program_index"][q])+1}'
                if slot[0] is None or slot[1] is None:label+=' · unmatched'
                elif side==1:label+=f'  |  r = {slot[2]:.3f}'
                ax.set_title(label,fontsize=8.5,pad=3)
        ca=fig.add_axes([.967,yb+.040,.007,rh-.075]);cb=fig.colorbar(matplotlib.cm.ScalarMappable(norm=Normalize(-bound,bound),cmap='RdBu_r'),cax=ca)
        cb.set_ticks([-bound,0,bound]);cb.ax.set_yticklabels([f'{-bound:.3g}','0',f'{bound:.3g}']);cb.ax.tick_params(labelsize=7,length=2);cb.outline.set_linewidth(.4);cb.ax.set_title('h',fontsize=9)
        if row<4:fig.add_artist(matplotlib.lines.Line2D([.005,.995],[yb,yb],transform=fig.transFigure,color='#dce2e6',lw=.5))
        records.append({'type':name,'slots':slots,'limit':bound})
    fig.add_artist(matplotlib.lines.Line2D([.56,.56],[bottom,top],transform=fig.transFigure,color='#dce2e6',lw=.5))
    RECORD['Fig4']=records;save(fig,'Fig4_paired_activity')

def fig6():
    original_height=7.3; extra_height=1.0; height=original_height+extra_height
    # Keep the map rows and inter-row gaps at their original physical sizes.
    row_height=(.95-.09)*original_height/(2.9*(1+2*.38/3))
    row_gap=.38*row_height*2.9/3
    row_heights=[row_height,row_height,.9*row_height+extra_height]
    fig=plt.figure(figsize=(11.5,height))
    grid=fig.add_gridspec(3,4,left=.07,right=.975,
        bottom=.09*original_height/height,top=(.95*original_height+extra_height)/height,
        hspace=row_gap/(sum(row_heights)/3),wspace=.29,height_ratios=row_heights)
    for row,name in enumerate(['Astrocyte','OPC']):
        c=T.index(name);bound=lim(c);tag(fig,.008,((.97-row*.29)*original_height+extra_height)/height,chr(97+row))
        for side in (0,1):
            ax=fig.add_subplot(grid[row,side*2]);im=bare_map(ax,side,F[side]['pi'][:,c],proportion=True)
            ax.set_title(f'{name} | S{side+1}: '+r'$\pi$',fontsize=10);cbar(fig,im,ax,proportion=True)
            ax=fig.add_subplot(grid[row,side*2+1]);qs=ids(side,c)
            if len(qs):
                im=bare_map(ax,side,F[side]['h_mean'][qs[0]],c=c,limit=bound);ax.set_title(f'S{side+1}: P1',fontsize=10);cbar(fig,im,ax,limit=bound)
            else:ax.axis('off');ax.text(.5,.5,'No selected program',ha='center',va='center',transform=ax.transAxes,fontsize=10,color='#697782')
    tag(fig,.008,(.35*original_height+extra_height)/height,'c')
    d=pd.read_csv(SOURCE/'elbo_by_iteration.csv')
    for side in (0,1):
        ax=fig.add_subplot(grid[2,side*2:side*2+2]);a=d[(d.section==side+1)&(d.role=='retained')];b=d[(d.section==side+1)&(d.role!='retained')]
        ax.plot(a.iteration,a.elbo/1e6,color=['#2b719b','#c07343'][side],lw=1.2,marker='.',ms=2.6)
        ax.set(title=f'Section {side+1}',xlabel='Iterations',ylabel=r'ELBO ($\times10^6$)',xlim=(0,12500))
        ax.set_xticks([0,2500,5000,7500,10000,12500]);ax.xaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x:,.0f}'))
        ax.yaxis.set_major_locator(MaxNLocator(4));ax.tick_params(labelsize=8);ax.grid(alpha=.15);ax.set_axisbelow(True)
    RECORD['Fig6']={'letters':{'a':'Astrocyte','b':'OPC','c':'ELBO'},'elbo_unit':'million','raw_elbo_preserved':True,'rejected_candidate_markers_shown':False,
        'figure_size_inches':[11.5,height],'panel_c_added_height_inches':extra_height,'upper_panel_physical_sizes_preserved':True};save(fig,'Fig6_additional_types_elbo')

def fig5():
    benchmark=BASELINES
    validation=json.loads((benchmark/'validation.json').read_text())
    assert validation['completed_cases']==6 and validation['all_outputs_finite']
    verified={row['case']:row for row in validation['cases']}
    metadata=[dict(np.load(benchmark/f'input_metadata/Visium_{side}_metadata.npz',allow_pickle=False)) for side in (1,2)]
    for side in (0,1):
        assert len(F[side]['genes'])==2359 and len(set(metadata[side]['genes']))==2359
        assert np.array_equal(metadata[side]['genes'],F[side]['genes'])
        np.testing.assert_allclose(metadata[side]['coords'],F[side]['coords'],rtol=0,atol=1e-7)
    c=T.index('Oligodendrocyte');qs=[int(ids(s,c)[0]) for s in (0,1)];bound=lim(c)
    load=[F[s]['loadings'][qs[s]].astype(float) for s in (0,1)]
    selected={};matchrows=[]
    for method in ['STAMP','SpiceMix','NMF']:
        for side in (0,1):
            case=f'Visium_{side+1}_{method}'
            p=benchmark/f'results/{case}/factors.npz';z=dict(np.load(p,allow_pickle=False))
            fit=json.loads((p.parent/'fit.json').read_text())
            mmf_manifest=json.loads((SOURCE/f'Visium_{side+1}_run_manifest.json').read_text())
            assert fit['input_sha256']==mmf_manifest['input_sha256']
            assert hashlib.sha256(p.read_bytes()).hexdigest()==verified[case]['factors_sha256']
            baseline_genes=metadata[side]['genes'];lookup={g:i for i,g in enumerate(baseline_genes)}
            common=[(i,lookup[g]) for i,g in enumerate(GENES) if g in lookup];ii,jj=np.asarray(common).T
            assert len(common)==2359
            a=load[side][ii];a=a-a.mean();b=z['loadings'][:,jj].astype(float);b-=b.mean(1,keepdims=True)
            den=np.linalg.norm(b,axis=1)*np.linalg.norm(a)
            cor=np.divide(b@a,den,out=np.full(len(b),-np.inf),where=den>0)
            q=int(np.argmax(cor));act=z['activity'][:,q].astype(float)
            assert np.isfinite(cor[q]) and len(act)==len(F[side]['coords']) and np.isfinite(act).all()
            selected[method,side]=(q,act,float(cor[q]))
            matchrows.append({'method':method,'section':side+1,'factor_index_zero_based':q,'factor_label':str(z['factor_ids'][q]),'loading_correlation':float(cor[q]),'shared_genes':len(common),'baseline_genes':len(baseline_genes),'n_spots':len(act),'activity_min':float(act.min()),'activity_max':float(act.max()),'activity_nonzero':int(np.count_nonzero(act)),'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'input_sha256':fit['input_sha256']})
    fig=plt.figure(figsize=(12,8.3));main=fig.add_gridspec(2,2,left=.025,right=.59,bottom=.39,top=.94,hspace=.25,wspace=.20)
    tag(fig,.012,.98,'a');tag(fig,.012,.68,'b');tag(fig,.012,.32,'c');tag(fig,.67,.98,'d')
    for side in (0,1):
        ax=fig.add_subplot(main[0,side]);im=bare_map(ax,side,F[side]['pi'][:,c],proportion=True);ax.set_title(f'Section {side+1}',fontsize=12);cbar(fig,im,ax,proportion=True)
        ax=fig.add_subplot(main[1,side]);im=bare_map(ax,side,F[side]['h_mean'][qs[side]],c=c,limit=bound);ax.set_title(f'Section {side+1}: P1',fontsize=11);cbar(fig,im,ax,limit=bound)
    bg=fig.add_gridspec(3,2,left=.70,right=.955,bottom=.38,top=.92,hspace=.30,wspace=.12)
    baseline_scales={'STAMP':(1.,[0,.5,1],['0','0.5','1']),
      'SpiceMix':(5.,[0,2.5,5],['0','2.5','5']),
      'NMF':(.10,[0,.05,.10],['0','0.05','0.10'])}
    saturation=[]
    for row,method in enumerate(['STAMP','SpiceMix','NMF']):
        vmax,ticks,ticklabels=baseline_scales[method]
        assert all(selected[method,s][1].min()>=0 for s in (0,1))
        if method!='SpiceMix':assert all(selected[method,s][1].max()<=vmax for s in (0,1))
        for side in (0,1):
            act=selected[method,side][1]
            saturation.append({'method':method,'section':side+1,'display_max':vmax,'above_display_max':int((act>vmax).sum()),'n_spots':len(act),'fraction_above_display_max':float((act>vmax).mean())})
        for side in (0,1):
            q,act,r=selected[method,side];xy=F[side]['coords'];ax=fig.add_subplot(bg[row,side]);
            im=ax.scatter(*xy.T,c=act,cmap='viridis',vmin=0,vmax=vmax,s=1.7,linewidths=0,rasterized=True)
            ax.set_aspect('equal');ax.invert_yaxis();ax.axis('off');ax.set_title(f'S{side+1}: F{q+1}',fontsize=8.5,pad=3)
            if side==0:ax.text(-.09,.5,method,ha='right',va='center',rotation=90,transform=ax.transAxes,fontsize=10)
            if np.ptp(act)==0:ax.text(.5,-.02,f'Constant {act[0]:g}',ha='center',transform=ax.transAxes,fontsize=7.5,color='#596974')
            else:ax.text(.5,-.02,f'r = {r:.2f}',ha='center',transform=ax.transAxes,fontsize=7.5,color='#596974')
        ca=fig.add_axes([.965,.38+(2-row)*.18+.015,.006,.12]);cb=fig.colorbar(matplotlib.cm.ScalarMappable(norm=Normalize(0,vmax),cmap='viridis'),cax=ca,extend='max' if method=='SpiceMix' else 'neither')
        cb.set_ticks(ticks);cb.ax.set_yticklabels(ticklabels);cb.ax.tick_params(labelsize=7,length=2);cb.outline.set_linewidth(.4)
    low=fig.add_gridspec(1,3,left=.075,right=.955,bottom=.07,top=.30,wspace=.43,width_ratios=[1,1,1])
    maximum=max(x.max() for x in load)*1.12
    for side in (0,1):
        ax=fig.add_subplot(low[side]);w=load[side];ii=np.argsort(-w,kind='stable')[:15]
        ax.barh(np.arange(15),w[ii],color='#72ac9d',height=.73);ax.set_yticks(np.arange(15),GENES[ii],fontsize=8);ax.invert_yaxis();ax.set_ylim(14.7,-.7)
        ax.set_xlim(0,maximum);ax.set_xticks([0,.01,.02,.03]);ax.tick_params(axis='y',length=0);ax.tick_params(axis='x',labelsize=8);ax.set_title(f'Section {side+1}',fontsize=11,pad=7);ax.set_xlabel('Gene loading',fontsize=9)
    ax=fig.add_subplot(low[2]);x,y=load;corr=float(np.corrcoef(x,y)[0,1]);ax.scatter(x,y,s=6,color='#72ac9d',alpha=.75,linewidths=0,rasterized=True);ax.plot([0,maximum],[0,maximum],color='#9eaeb9',lw=.7)
    ax.set(xlim=(0,maximum),ylim=(0,maximum),xlabel='Section 1 loading',ylabel='Section 2 loading');ax.set_aspect('equal');ax.set_xticks([0,.01,.02,.03]);ax.set_yticks([0,.01,.02,.03]);ax.tick_params(labelsize=8);ax.set_title(f'r = {corr:.3f}',fontsize=11,pad=7)
    for gene,offset in [('Ptprd',(-8,10)),('Malat1',(7,-3)),('Fgfr2',(-22,8)),('Plp1',(7,-8))]:
        j=int(np.flatnonzero(GENES==gene)[0]);ax.annotate(gene,(x[j],y[j]),xytext=offset,textcoords='offset points',fontsize=7,ha='right' if offset[0]<0 else 'left',arrowprops={'arrowstyle':'-','color':'#536373','lw':.4})
    with (OUTPUT/'benchmark_2359_matches.csv').open('w') as h:
        w=csv.DictWriter(h,fieldnames=list(matchrows[0]));w.writeheader();w.writerows(matchrows)
    RECORD['Fig5']={'baseline_panel':'completed matched-input 2359 genes','common_genes':2359,'matching':'maximum signed loading correlation with current MMF oligo P1, separately per method/section','matches':matchrows,'main_program_native_scale':True,'baseline_native_activity':True,'input_hashes_match_MMF':True,'gene_order_and_coordinates_verified':True,'main_map_left_shift_figure_fraction':.05,'main_map_additional_left_shift_figure_fraction':.02}
    RECORD['Fig5'].update(baseline_color_ranges={m:[0,config[0]] for m,config in baseline_scales.items()},shared_scale_between_sections=True,activity_clipped=True,clipping_applies_to='color scale only',raw_activity_modified=False,saturation=saturation,colorbar_upper_extension='SpiceMix')
    save(fig,'Fig5_oligodendrocyte')


def main():
    global SOURCE, OUTPUT, BASELINES, F, T, GENES
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True,help='Visium source NPZs and ELBO table')
    p.add_argument('--baselines',type=Path,required=True,help='Completed matched 2359-gene baseline figure sources')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();SOURCE=a.data;OUTPUT=a.output;BASELINES=a.baselines
    OUTPUT.mkdir(parents=True,exist_ok=True)
    F=[dict(np.load(SOURCE/f'source/Visium_{s}_figure_source.npz',allow_pickle=False)) for s in (1,2)]
    T=list(F[0]['cell_types']);GENES=F[0]['genes']
    fig4();fig5();fig6()
    (OUTPUT/'plot_audit.json').write_text(json.dumps(RECORD,indent=2)+'\n')

if __name__=='__main__':main()
