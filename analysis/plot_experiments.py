#!/usr/bin/env python3
"""Render manuscript figures from aggregate evidence only (Python/matplotlib)."""
from pathlib import Path
import csv
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'evidence/derived/experiments_20261101'
OUT = ROOT/'manuscript/images/experiments_20261101'
FONTS = Path.home()/'.local/share/fonts/las-paper'
MODELS = ('llama_instruct','qwen_instruct')
TITLES = ('Llama-3.2-3B / English', 'Qwen3-4B / Chinese')
KS = (16,32,128,256,0)
K_LABELS = ('16','32','128','256','Full')
MARKERS = ('o','s','^','D','P')
TAUS = (0,.125,.25,.5,1,2,4)
LAS = ('las_imec','adaptive_0125_imec','adaptive_0250_imec','adaptive_0500_imec',
       'adaptive_1000_imec','adaptive_2000_imec','adaptive_4000_imec')
LAS_BLUE, SYNC_BLACK = '#0B5CAD', '#111111'
# Chapter-wide grammar: method = color + line style; k = marker shape; filled =
# distribution-preserving, hollow = filtering; model = line style and gray tone.
STYLES = {
    'las_imec': (r'LAS($\tau$=0)', LAS_BLUE, '-', True),
    'adaptive_4000_imec': (r'LAS($\tau$=4)', '#7FB0E0', '-', True),
    'syncpool_imec': ('SyncPool', SYNC_BLACK, '--', True),
    'stepwise_imec': ('Stepwise', '#0F766E', (0,(5,2,1,2)), False),
    'mwis_imec': ('MWIS', '#B64040', '-.', False),
    'basic_imec': ('Basic', '#B7791F', ':', False),
}
MODEL_STYLE = {'llama_instruct': ('#222222', '-', 'Llama-3.2-3B (EN)'),
               'qwen_instruct': ('#8A8A8A', (0,(4,2)), 'Qwen3-4B (ZH)')}
COST_LABEL = 'Sender + receiver time (ms / body token)'
for font in font_manager.findSystemFonts([str(FONTS)]):
    font_manager.fontManager.addfont(font)
# Style follows the installed ieee-figure-table helper; text matches the Times body.
plt.rcParams.update({
    'font.family':'Times New Roman', 'mathtext.fontset':'stix', 'font.size':8,
    'axes.labelsize':8, 'axes.titlesize':8, 'xtick.labelsize':8, 'ytick.labelsize':8,
    'legend.fontsize':8, 'legend.frameon':False,
    'axes.linewidth':.7, 'axes.spines.top':False, 'axes.spines.right':False,
    'grid.color':'#D7DEE8','grid.linewidth':.45,
    'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
    'savefig.facecolor':'white','axes.axisbelow':True,
})
read = lambda name: list(csv.DictReader((DATA/name).open()))
points = {(r['model'],int(r['top_k']),r['variant']):r for r in read('MAIN_POINTS.csv')}
intervals = {(r['model'],int(r['top_k']),r['variant'],r['metric']):
             (float(r['ci95_low']),float(r['ci95_high'])) for r in read('MAIN_POINT_INTERVALS.csv')}
ambiguity = {(r['model'],int(r['top_k'])):r for r in read('AMBIGUITY_FREQUENCY.csv')}
mechanism = {(r['model'],int(r['top_k'])):r for r in read('MAIN_MECHANISM.csv')}
relative = {(r['model'],int(r['top_k']),r['variant']):r for r in read('THRESHOLD_RELATIVE.csv')}
coordinates = []


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    # Exact final canvas: 3.5 in for one column, 7.16 in for two; text stays 8 pt.
    for suffix in ('pdf','svg','png'):
        fig.savefig(OUT/f'{name}.{suffix}', dpi=450, metadata={'Creator':'plot_experiments.py'})
    plt.close(fig)


def record(figure, panel, model, k, v, x, y, mx, my):
    ix, iy = intervals[model,k,v,mx], intervals[model,k,v,my]
    coordinates.append(dict(figure=figure,panel=panel,model=model,top_k=k,
        variant=v,x=x,y=y,x_metric=mx,y_metric=my,
        x_ci95_low=ix[0],x_ci95_high=ix[1],y_ci95_low=iy[0],y_ci95_high=iy[1]))


# Fig. mechanism: ambiguity in ordinary sampling and the discarded share.
fig, (a1, a2) = plt.subplots(2, 1, figsize=(3.5, 3.1))
fig.subplots_adjust(left=.15, right=.97, top=.93, bottom=.13, hspace=.72)
for model in MODELS:
    color, line, label = MODEL_STYLE[model]
    y = np.array([100*float(ambiguity[model,k]['ambiguous_step_share']) for k in KS])
    lo = np.array([100*float(ambiguity[model,k]['ambiguous_ci95_low']) for k in KS])
    hi = np.array([100*float(ambiguity[model,k]['ambiguous_ci95_high']) for k in KS])
    a1.plot(range(len(KS)), y, color=color, ls=line, lw=1.0)
    a1.vlines(range(len(KS)), lo, hi, color=color, lw=.7)
    for i, marker in enumerate(MARKERS):
        a1.plot(i, y[i], marker=marker, ms=4.4, color=color, mec=color)
a1.set_xticks(range(len(KS)), K_LABELS)
a1.set_xlim(-.3, len(KS)-.7)
a1.set_xlabel('Top-$k$ setting', labelpad=1)
a1.set_ylim(0, 105); a1.set_yticks((0, 25, 50, 75, 100))
a1.set_ylabel('Ambiguous steps (%)')
a1.set_title('(a) Ambiguity frequency in ordinary sampling', loc='left')
a1.grid(True, axis='y')
a1.legend(handles=[Line2D([],[],color=MODEL_STYLE[m][0],ls=MODEL_STYLE[m][1],lw=1.0,
                          label=MODEL_STYLE[m][2]) for m in MODELS],
          loc='lower right', handlelength=2.4)
s = np.linspace(0, .94, 200)
a2.plot(100*s, 1/(1-s), color='#9AA6B4', lw=.9, ls=(0,(3,2)))
a2.annotate(r'$1/(1-s)$', (87, 1/(1-.87)), xytext=(-34, 4), textcoords='offset points',
            color='#5F6B78')
for model in MODELS:
    color = MODEL_STYLE[model][0]
    for k, marker in zip(KS, MARKERS):
        r = mechanism[model,k]
        x, y = 100*float(r['discarded_share']), float(r['bpt_gain'])
        a2.errorbar(x, y, xerr=[[x-100*float(r['discarded_share_ci95_low'])],
                                [100*float(r['discarded_share_ci95_high'])-x]],
                    yerr=[[y-float(r['bpt_gain_ci95_low'])],[float(r['bpt_gain_ci95_high'])-y]],
                    fmt=marker, ms=4.8, color=color, mec='white', mew=.4, elinewidth=.6, capsize=0)
# Qwen's four finite-k points and Llama k=16 coincide near s = 13%.
a2.annotate('Qwen, $k$=16–256; Llama, $k$=16', (15, 1.12), xytext=(27, .99),
            textcoords='data', va='bottom', color='#444444',
            arrowprops=dict(arrowstyle='-', color='#777777', lw=.5, shrinkA=1, shrinkB=2))
a2.set_yscale('log')
a2.set_yticks((1,2,5,10), ('1','2','5','10')); a2.set_yticks([], minor=True)
a2.set_ylim(.95, 15)
a2.set_xlim(0, 100); a2.set_xticks((0,25,50,75,100))
a2.set_xlabel('Entropy share synchronized by SyncPool, $s$ (%)', labelpad=1)
a2.set_ylabel(r'BPT, LAS($\tau$=0) / SyncPool')
a2.set_title('(b) Synchronized share and capacity gain', loc='left')
a2.grid(True, which='major')
a2.legend(handles=[Line2D([],[],color='#555555',marker=m,ls='',ms=4.4,label=l)
                   for m, l in zip(MARKERS, K_LABELS)],
          title='$k$', loc='upper left', handletextpad=.1, labelspacing=.2, borderaxespad=.2)
save(fig, 'ambiguity_mechanism')

# Fig. capacity-cost: all methods at every k.
fig, axes = plt.subplots(1,2,figsize=(7.16,2.4))
fig.subplots_adjust(left=.07,right=.985,bottom=.175,top=.75,wspace=.22)
cx, cy = 'sender_receiver_ms_per_body_token','native_body_bpt'
for col,(model,ax) in enumerate(zip(MODELS,axes)):
    for v,(label,color,line,filled) in STYLES.items():
        available = [k for k in KS if (model,k,v) in points]
        x = [float(points[model,k,v][cx]) for k in available]
        y = [float(points[model,k,v][cy]) for k in available]
        ax.plot(x,y,color=color,linestyle=line,lw=.95,alpha=.9)
        for k,xx,yy in zip(available,x,y):
            ix, iy = intervals[model,k,v,cx], intervals[model,k,v,cy]
            ax.hlines(yy,*ix,color=color,lw=.6,alpha=.5)
            ax.vlines(xx,*iy,color=color,lw=.6,alpha=.5)
            ax.plot(xx,yy,marker=MARKERS[KS.index(k)],ms=4.4,ls='',color=color,
                    markerfacecolor=color if filled else 'white',markeredgewidth=.85)
            record('capacity_cost',chr(97+col),model,k,v,xx,yy,cx,cy)
    ax.set_xscale('log')
    ax.set_xlim(60,13000)
    ax.set_xticks((100,1000,10000),('100','1,000','10,000'))
    ax.set_ylim(bottom=0)
    ax.set_xlabel(COST_LABEL, labelpad=1)
    ax.set_ylabel('Confirmed bits / body token')
    ax.set_title(f'({chr(97+col)}) {TITLES[col]}',loc='left',pad=5)
    ax.grid(True,axis='both')
methods = [Line2D([],[],color=c,ls=l,lw=1.4,label=n) for n,c,l,f in STYLES.values()]
kkeys = [Line2D([],[],color='#444444',ls='',marker=m,ms=4.3,markerfacecolor='white',
                label='Full' if k==0 else f'$k$={k}') for k,m in zip(KS,MARKERS)]
fig.legend(handles=methods,loc='upper center',bbox_to_anchor=(.53,1.0),ncol=6,
           columnspacing=1.3,handlelength=2.2,handletextpad=.45,borderpad=.15)
fig.legend(handles=kkeys,loc='upper center',bbox_to_anchor=(.53,.92),ncol=5,
           columnspacing=1.6,handletextpad=.3,borderpad=.15)
save(fig,'capacity_cost')

# Fig. threshold: LAS(tau) relative to LAS(0); k=256 and Full; SP endpoint.
fig, axes = plt.subplots(2, 1, figsize=(3.5, 2.9), sharex=True)
fig.subplots_adjust(left=.15, right=.97, top=.835, bottom=.13, hspace=.38)
xx = np.arange(len(TAUS)); SP_X = len(TAUS) + .55
LINES = [(m, k) for m in MODELS for k in (256, 0)]
DODGE = dict(zip(LINES, (-.12, -.04, .04, .12)))
for ax, metric, title in zip(axes, ('capacity', 'time'),
                             ('(a) Confirmed bits per body token', '(b) Sender + receiver time per body token')):
    for model, k in LINES:
        line = MODEL_STYLE[model][1]
        marker = MARKERS[KS.index(k)]
        rows = [relative[model,k,v] for v in LAS]
        y = np.array([float(r[f'{metric}_ratio']) for r in rows])
        lo = np.array([float(r[f'{metric}_ci95_low']) for r in rows])
        hi = np.array([float(r[f'{metric}_ci95_high']) for r in rows])
        x = xx + DODGE[model,k]
        ax.plot(x, y, color=LAS_BLUE, ls=line, lw=1.0)
        ax.vlines(x, lo, hi, color=LAS_BLUE, lw=.6, alpha=.75)
        ax.plot(x, y, marker=marker, ms=3.9, ls='', color=LAS_BLUE, mec='white', mew=.4)
        # SyncPool endpoints carry the model gray tone of the mechanism figure.
        sp, tone = relative[model,k,'syncpool_imec'], MODEL_STYLE[model][0]
        ax.vlines(SP_X + DODGE[model,k], float(sp[f'{metric}_ci95_low']),
                  float(sp[f'{metric}_ci95_high']), color=tone, lw=.6)
        ax.plot(SP_X + DODGE[model,k], float(sp[f'{metric}_ratio']), marker=marker, ms=3.9,
                ls='', color=tone, mec='white', mew=.4)
    ax.axvline(len(TAUS)-.2, color='#B9C3CF', lw=.6, ls=(0,(2,2)))
    ax.set_ylim(0, 1.32); ax.set_yticks((0,.25,.5,.75,1,1.25))
    ax.set_ylabel(r'Relative to LAS($\tau$=0)')
    ax.set_title(title, loc='left', pad=4)
    ax.grid(True, axis='y')
axes[1].set_xticks(list(xx)+[SP_X], ('0','1/8','1/4','1/2','1','2','4','SP'))
axes[1].set_xlim(-.45, SP_X+.45)
axes[1].set_xlabel(r'Threshold $\tau$ (bits)', labelpad=1)
handles = [Line2D([],[],color=LAS_BLUE,ls=MODEL_STYLE[m][1],lw=1.0,marker=MARKERS[KS.index(k)],
                  ms=3.9,label=f"{'Llama' if m==MODELS[0] else 'Qwen'}, {'Full' if k==0 else '$k$=256'}")
           for m, k in LINES]
labels = [h.get_label() for h in handles]
for m in MODELS:
    handles.append(tuple(Line2D([],[],color=MODEL_STYLE[m][0],ls='',marker=MARKERS[KS.index(k)],ms=3.9)
                         for k in (256, 0)))
    labels.append(f"SP, {'Llama' if m==MODELS[0] else 'Qwen'}")
fig.legend(handles=handles, labels=labels, loc='upper center', ncol=3, bbox_to_anchor=(.55, 1.0),
           handlelength=2.4, columnspacing=1.0, labelspacing=.2, borderpad=.1, handletextpad=.4,
           handler_map={tuple: HandlerTuple(ndivide=None, pad=.4)})
save(fig, 'threshold_scan')

with (DATA/'FIGURE_COORDINATES.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(coordinates[0]),lineterminator='\n');w.writeheader();w.writerows(coordinates)
receipt = dict(date='2026-11-02', backend='Python/matplotlib',
    matplotlib_version=matplotlib.__version__, font='Times New Roman 8 pt (body font)',
    widths_inches=dict(ambiguity_mechanism=3.5, capacity_cost=7.16, threshold_scan=3.5),
    uncertainty='10,000 prompt-cluster percentile bootstrap; ratios resample documents jointly; exploratory',
    capacity_cost_points=len(coordinates),
    stepwise_full='not evaluated; no coordinate or interpolated endpoint',
    comparison_contract='Filled: LAS/SyncPool common selected source; hollow: filtering methods',
    ambiguity_mechanism=dict(a='AMBIGUITY_FREQUENCY.csv (replay of the 320 main ordinary texts)',
                             b='MAIN_MECHANISM.csv; reference curve 1/(1-s) has no fitted parameter'),
    threshold_scan='THRESHOLD_RELATIVE.csv; k=256 and Full; SP = shared-planner SyncPool endpoint',
    errorbars='All panels: 95% intervals as whiskers',
    data_sha256={name: hashlib.sha256((DATA/name).read_bytes()).hexdigest() for name in (
        'MAIN_POINTS.csv','MAIN_POINT_INTERVALS.csv','AMBIGUITY_FREQUENCY.csv',
        'MAIN_MECHANISM.csv','THRESHOLD_RELATIVE.csv')},
    exports={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(OUT.iterdir())})
(ROOT/'verification/EXPERIMENT_FIGURES_20261101.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({k:v for k,v in receipt.items() if k not in ('exports','data_sha256')}))
