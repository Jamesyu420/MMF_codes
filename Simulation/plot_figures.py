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

def fig1():
    d=pd.read_csv(DATA/'case_metrics.csv');d=d[d.study=='primary50']
    methods=list(COLORS);global_methods=[m for m in methods if m!='RCTD']
    specs=[('composition_correlation','Composition pattern','Mean Pearson correlation',methods),
      ('loading_cosine','Gene loadings','Mean gene-centered cosine',global_methods),
      ('activity_correlation','Program activity','Mean Pearson correlation',global_methods),
      ('seconds','Runtime',r'$\log_{10}(\mathrm{minutes})$',global_methods)]
    fig,axs=plt.subplots(2,2,figsize=(7.2,5.4));fig.subplots_adjust(left=.095,right=.985,bottom=.10,top=.94,wspace=.33,hspace=.44)
    rng=np.random.default_rng(1)
    for ax,(metric,title,ylabel,ms),letter in zip(axs.ravel(),specs,'abcd'):
        vals=[d.loc[d.method==m,metric].to_numpy()/(60 if metric=='seconds' else 1) for m in ms]
        if metric=='seconds':vals=[np.log10(v) for v in vals]
        assert all(len(v)==50 and np.isfinite(v).all() for v in vals)
        bp=ax.boxplot(vals,widths=.5,patch_artist=True,showfliers=False,
          medianprops={'color':'#333333','lw':1},whiskerprops={'color':'#777777','lw':.7},capprops={'color':'#777777','lw':.7})
        for i,(v,m,box) in enumerate(zip(vals,ms,bp['boxes']),1):
            box.set(facecolor=COLORS[m],alpha=.25,edgecolor='none')
            ax.scatter(i+rng.uniform(-.16,.16,len(v)),v,s=8,color=COLORS[m],alpha=.65,linewidths=0,zorder=3)
        ax.set_xticks(np.arange(1,len(ms)+1),ms,rotation=22,ha='right',fontsize=8)
        ax.set(title=title,ylabel=ylabel);ax.title.set_fontsize(10);ax.yaxis.label.set_fontsize(9);ax.tick_params(axis='y',labelsize=8)
        ax.grid(axis='y',color='#e5e5e5',lw=.5);ax.set_axisbelow(True)
        ax.text(-.18,1.07,letter,transform=ax.transAxes,weight='bold',fontsize=13)
        if metric=='seconds':
            ax.set_ylim(-2,2);ax.set_yticks([-2,-1,0,1,2])
        else:ax.set_ylim(0,1.04)
    # Preserve the original runtime panel, including the original seeded jitter.
    supplemental=OUTPUT/'supplementary';supplemental.mkdir(parents=True,exist_ok=True)
    ax=axs[1,1]
    for artist in ax.texts:artist.set_visible(False)
    fig.canvas.draw()
    runtime_bbox=ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted()).padded(.06)
    for ext in ['pdf','svg','png']:
        fig.savefig(supplemental/f'FigS_runtime_boxplots.{ext}',dpi=240,bbox_inches=runtime_bbox,facecolor='white')
    # Replace only panel d with the chosen categorical-spacing scalability view.
    scaling_path=DATA/'scalability_runtime_source.csv'
    scaling=pd.read_csv(scaling_path)
    assert len(scaling)==16 and scaling.status.eq('completed').all()
    sizes=[400,2500,10000,40000];x=np.arange(4)
    curve_colors={**COLORS,'NMF':'#92969A'}
    markers={'MMF':'o','STAMP':'s','SpiceMix':'D','NMF':'^'}
    ax.clear();handles={}
    for method in ['NMF','STAMP','SpiceMix','MMF']:
        rows=scaling[scaling.method==method].sort_values('N');assert rows.N.tolist()==sizes
        y=np.log10(rows.minutes.to_numpy());assert np.isfinite(y).all() and y.min()>-2 and y.max()<2
        line,=ax.plot(x,y,color=curve_colors[method],lw=1.15,marker=markers[method],
            ms=4.5 if method=='MMF' else 3.8,
            markerfacecolor='none' if method=='MMF' else curve_colors[method],
            markeredgecolor=curve_colors[method] if method=='MMF' else 'white',
            markeredgewidth=.85 if method=='MMF' else .35,label=method,zorder=4 if method=='MMF' else 3)
        handles[method]=line
    ax.set(xlim=(-.18,3.18),ylim=(-2,2),xlabel=r'$N$',ylabel=r'$\log_{10}(\mathrm{minutes})$')
    ax.set_xticks(x,[f'{n:,}' for n in sizes]);ax.set_yticks([-2,-1,0,1,2])
    ax.tick_params(axis='x',labelsize=7.5);ax.tick_params(axis='y',labelsize=8)
    ax.xaxis.label.set_fontsize(9);ax.yaxis.label.set_fontsize(9)
    ax.grid(axis='y',color='#e5e5e5',lw=.5);ax.set_axisbelow(True)
    ax.legend(handles=[handles[m] for m in global_methods],loc='lower center',bbox_to_anchor=(.5,1.035),
        ncol=4,frameon=False,fontsize=6.8,handlelength=1.2,handletextpad=.35,columnspacing=.8,borderaxespad=0)
    ax.text(-.18,1.07,'d',transform=ax.transAxes,weight='bold',fontsize=13)
    d[d.method!='RCTD'][['seed','method','seconds','host','device']].to_csv(OUTPUT/'runtime_source.csv',index=False)
    RECORD['Fig1']={'panels_abc_n_per_method':50,'panels_abc_unchanged':True,
        'panel_d':'scalability','panel_d_sizes':sizes,'panel_d_replicates_per_size':1,
        'panel_d_methods':global_methods,'panel_d_unit':'log10(minutes)',
        'panel_d_x_spacing':'equally spaced categories','panel_d_xlabel':'N','panel_d_title':None,
        'panel_d_ylim':[-2,2],'panel_d_source':str(scaling_path),
        'MMF_N40000_rho':.7,'smaller_MMF_rho':.25,'hardware_homogeneous':False,
        'original_runtime_panel_saved':str(supplemental/'FigS_runtime_boxplots.pdf'),
        'original_runtime_n_per_method':50,'original_runtime_seeded_jitter_preserved':True,
        'runtime_median_minutes':d.groupby('method').seconds.median().div(60).to_dict()}
    save(fig,'Fig1_boxplots')

def load(path):
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}

def centered(v):
    z = v - np.mean(v)
    return z / max(np.sqrt(np.mean(z*z)), 1e-15)

def grid(coords, v):
    x, y = np.unique(coords[:, 0]), np.unique(coords[:, 1])
    out = np.full((len(y), len(x)), np.nan)
    out[np.searchsorted(y, coords[:, 1]), np.searchsorted(x, coords[:, 0])] = v
    return out

def letter(ax, value, x=-.15):
    ax.text(x, 1.06, value, transform=ax.transAxes, weight="bold", fontsize=10, va="bottom")

def maps():
    import string
    from matplotlib.font_manager import findfont
    try:
        findfont('Arial', fallback_to_default=False)
    except ValueError as error:
        raise RuntimeError('Submitted Figures 2–3 require Arial. Install a licensed Arial font and rebuild the Matplotlib font cache before redrawing.') from error
    data = DATA
    METHODS=list(COLORS); GLOBALS=['STAMP','SpiceMix','NMF']
    comp=pd.read_csv(data/'composition_matching.csv')
    prog=pd.read_csv(data/'program_matching.csv')
    validation=json.loads((data/'validation_summary.json').read_text())
    plt.rcdefaults()
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                    "font.size": 7.5, "axes.titlesize": 8.5, "axes.labelsize": 8,
                    "axes.spines.top": False, "axes.spines.right": False,
                    "axes.linewidth": .65, "pdf.fonttype": 42, "svg.fonttype": "none",
                    "legend.frameon": False, "savefig.facecolor": "white"})
    def save(fig,name,title,caption):
        for ext in ['png','pdf','svg']:
            fig.savefig(OUTPUT/f'{name}.{ext}',dpi=300,bbox_inches='tight',pad_inches=.05)
        plt.close(fig)
    rep = data/"representative"
    truth, training = load(rep/"evaluation_only.npz"), load(rep/"fit_input.npz")
    fits = {m: load(rep/f"{m}.npz") for m in METHODS}
    seed = validation["representative_seed"]
    cp = comp.query("study == 'primary50' and seed == @seed")
    pp = prog.query("study == 'primary50' and seed == @seed")
    coords = training["coords"]
    plot_source = dict(coords=coords, true_pi=truth["pi_true"], true_h=truth["h_true"],
                       program_cell_type=truth["program_cell_type"], program_index=truth["program_index"])
    for kind, number in [("pi", 2), ("h", 3)]:
        names = ["Truth"] + (METHODS if kind == "pi" else ["MMF"]+GLOBALS)
        fig, axs = plt.subplots(len(names), 5, figsize=(7.2, 1.27*len(names)+.2), squeeze=False)
        fig.subplots_adjust(wspace=.13, hspace=.36, left=.14, right=.99, top=.95, bottom=.09)
        for i, method in enumerate(names):
            for j, ax in enumerate(axs[i]):
                score = None
                if method == "Truth":
                    value = truth["pi_true"][:,j] if kind == "pi" else truth["h_true"][j]
                elif kind == "pi":
                    match = cp[(cp.method == method) & (cp.type == j+1)].iloc[0]
                    raw = fits[method]["pi"] if method in ["MMF", "RCTD"] else fits[method]["activity"]
                    value = raw[:,int(match.factor)]; score = match.correlation
                else:
                    match = pp[(pp.method == method) & (pp.true_program == j)].iloc[0]
                    raw = fits[method]["h_mean"] if method == "MMF" else fits[method]["activity"].T
                    value = np.zeros(len(coords)) if pd.isna(match.factor) else raw[int(match.factor)]
                    score = match.activity_correlation
                plot_source[f"{kind}_{method}_{j}"] = value
                im = ax.imshow(grid(coords, centered(value)), origin="lower", cmap="RdBu_r", vmin=-2.5, vmax=2.5, interpolation="nearest")
                ax.set_xticks([]); ax.set_yticks([])
                for spine in ax.spines.values(): spine.set_visible(False)
                if i == 0:
                    title = f"Cell type {j+1}" if kind == "pi" else f"T{truth['program_cell_type'][j]+1}  P{truth['program_index'][j]+1}"
                    ax.set_title(title, pad=6)
                if j == 0:
                    label = method + ("*" if method in GLOBALS else "")
                    ax.set_ylabel(label, rotation=0, ha="right", va="center", labelpad=13, fontsize=8)
                    letter(ax, string.ascii_lowercase[i], -.35)
                if score is not None:
                    ax.set_xlabel(f"r = {score:.3f}", fontsize=7, labelpad=2)
        cax = fig.add_axes([.31, .035, .49, .012])
        cb = fig.colorbar(im, cax=cax, orientation="horizontal", extend="both")
        cb.set_ticks([-2.5, 0, 2.5])
        cb.set_label("Centered map / spatial RMS (display limits ±2.5)", fontsize=7)
        title = "Spatial composition patterns in the first prespecified replicate" if kind == "pi" else "Distinct within-type program patterns in the same replicate"
        caption = ("Rows show truth and fitted patterns for the first prespecified primary seed (2026102901), chosen independently of outcomes. "
                   + ("Columns are the five cell types. MMF and RCTD estimate mixture proportions; starred rows are post-fit, correlation-matched global factors and are not proportions. " if kind == "pi" else "Columns are the five true programs, including two distinct programs in type 2; type 3 has no true program. Estimated programs are matched using gene loadings, within cell type for MMF and globally for starred factor methods. No activity-based matching or sign reversal is used. ")
                   + "Every map is centered and divided by its spatial root-mean-square deviation solely for display. All maps share limits of -2.5 to 2.5; values outside these limits saturate. Printed Pearson correlations use the complete, unclipped original arrays. These maps assess pattern shape, not native amplitude or interval calibration. Coordinates and pixels are unchanged.")
        save(fig, f"Fig{number}_{kind}_maps", title, caption)
    np.savez_compressed(OUTPUT/"map_plot_source.npz", **plot_source)


def main():
    global DATA, OUTPUT
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True,help='Simulation figure-source directory')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();DATA=a.data;OUTPUT=a.output;OUTPUT.mkdir(parents=True,exist_ok=True)
    fig1(); maps()
    (OUTPUT/'plot_audit.json').write_text(json.dumps(RECORD,indent=2)+'\n')

if __name__=='__main__':main()
