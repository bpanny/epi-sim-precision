"""Self-contained six-panel toy DKW figure.

Install: python -m pip install numpy pandas matplotlib
Run:     python scripts/simulate_dkw_cdf_error.py
Outputs: ./toy_DKW_CDF_reusability/ (relative to this script)
No inputs, prior outputs, or versioned filenames are required.
"""
#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter, NullFormatter, FuncFormatter, LogLocator

OUT = Path(__file__).resolve().parents[1] / 'toy_DKW_CDF_reusability'
OUT.mkdir(parents=True, exist_ok=True)
SEED=20261014; H=43; REPS=35; N_REF=150000; ALPHA=.05
NS=np.array([50,75,100,150,200,300,400,500,600,800,1000,1500,2000,3000,4000,6000,8000,10000])
n=NS.astype(float)
RATE=(160000*100000/341784857)/2
SUPPORT_MAX=2*RATE
plt.rcParams.update({'font.size':10,'axes.labelsize':11,'xtick.labelsize':10,'ytick.labelsize':10,
                     'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})

def trajectories(num,rng):
    r=rng.choice(4,size=num,p=[.83,.09,.06,.02]); t=np.arange(H)[None,:]
    c=rng.normal(20.,2.7,num)+np.select([r==1,r==2,r==3],[1,12,-10],default=0)
    w=np.maximum(2,rng.normal(6.,1.,num))*np.where(r==3,.6,1.)
    a=rng.lognormal(np.log(.46),.22,num)*np.select([r==1,r==2,r==3],[1.7,.85,1.3],default=1)
    p=rng.uniform(.015,.05,num)[:,None]+a[:,None]*np.exp(-.5*((t-c[:,None])/w[:,None])**2)
    p+=rng.normal(0,.014,(num,1))+rng.normal(0,.009,(num,H))
    return (np.clip(p,0,2)*RATE).astype(np.float32),r

def cdf_discrepancy_at_sample(vals,ref_sorted):
    s=np.sort(vals); N=len(s)
    right_ref=np.searchsorted(ref_sorted,s,'right')/len(ref_sorted)
    left_ref=np.searchsorted(ref_sorted,s,'left')/len(ref_sorted)
    lo=np.searchsorted(s,s,'left')/N
    hi=np.searchsorted(s,s,'right')/N
    return max(float(np.max(np.abs(hi-right_ref))),float(np.max(np.abs(lo-left_ref))))

def style_x(ax):
    ax.set_xscale('log'); ax.set_xlim(50,10000)
    ax.set_xticks([50,100,200,400,600,1000,2000,4000,10000])
    ax.xaxis.set_major_formatter(ScalarFormatter()); ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis='x',labelsize=9)

def decfmt(x,pos):
    if x<=0: return ''
    return f'{x:.4f}'.rstrip('0').rstrip('.') if x<1 else f'{x:g}'

def add_n600(ax):
    ax.axvline(600,color='.3',ls='--',lw=1)
    ax.text(625,.88,'n = 600',transform=matplotlib.transforms.blended_transform_factory(ax.transData,ax.transAxes),fontsize=9,color='.25')

def min_n_to_precision(eps, J=1):
    """Exact minimum integer n under the fixed-sample DKW bound."""
    return int(np.ceil(np.log(2 * J / ALPHA) / (2 * eps**2)))


def add_multiplier_lines_panelB(ax):
    target = 0.05
    baseline = min_n_to_precision(target, J=1)
    requirements = [min_n_to_precision(target, J=H),
                    min_n_to_precision(target, J=3*H)]
    ax.axhline(target, color='#d4772e', ls='-.', lw=1.3)
    ax.text(56, target*1.06, 'Reference max CDF error', color='#d4772e', fontsize=8.6,
            ha='left', va='bottom', bbox=dict(facecolor='white', edgecolor='none', alpha=.84, pad=.3))
    cols = ['#368a82', '#a37638']
    ys = [0.82, 0.72]
    for x, c, y in zip(requirements, cols, ys):
        ax.axvline(x, color=c, ls=':', lw=1.15, alpha=.95)
        ax.text(x, y, f'×{x/baseline:.2f}', color=c, fontsize=8.5, ha='center', va='bottom',
                transform=matplotlib.transforms.blended_transform_factory(ax.transData, ax.transAxes),
                bbox=dict(facecolor='white', edgecolor='none', alpha=.82, pad=.4))
    print(f'Integer trajectory requirements at epsilon={target}: '
          f'J=1: {baseline}; J=43: {requirements[0]}; J=129: {requirements[1]}', flush=True)

# Generate the entire simulation dataset from scratch (no external CSV inputs).
# The 150,000-trajectory reference is a high-sample approximation to the
# population distribution, not the exact population CDF.
rng = np.random.default_rng(SEED)
ref, _ = trajectories(N_REF, rng)
spag, spag_r = trajectories(180, np.random.default_rng(SEED + 1))
mag_ref = np.sort(ref.max(axis=1))
peak_ref = np.argmax(ref, axis=1)
peak_ref_cdf = np.bincount(peak_ref, minlength=H).cumsum() / N_REF
cum_ref = np.sort(ref.sum(axis=1))

# Independent simulation replicates for peak magnitude and timing.
# Keeping RNG continuation from the reference matches the original analysis.
peak_rows = []
for rep in range(REPS):
    sims, _ = trajectories(int(NS[-1]), rng)
    mags = sims.max(axis=1)
    peaks = np.argmax(sims, axis=1)
    for nn in NS:
        nn = int(nn)
        magerr = cdf_discrepancy_at_sample(mags[:nn], mag_ref)
        peakcdf = np.bincount(peaks[:nn], minlength=H).cumsum() / nn
        timerr = float(np.max(np.abs(peakcdf - peak_ref_cdf)))
        peak_rows.append((rep, nn, magerr, timerr))
    print(f'Peak CDF replicate {rep + 1}/{REPS}', flush=True)
peak_df = pd.DataFrame(peak_rows, columns=[
    'replicate', 'n', 'peak_magnitude_cdf_error', 'peak_timing_cdf_error'
])
peak_df.to_csv(OUT / 'peak_cdf_replicates.csv', index=False)

# Independent replicates for cumulative 43-week hospitalization burden.
cum_rows = []
rng_cum = np.random.default_rng(SEED + 222)
for rep in range(REPS):
    sims, _ = trajectories(int(NS[-1]), rng_cum)
    sums = sims.sum(axis=1)
    for nn in NS:
        nn = int(nn)
        err = cdf_discrepancy_at_sample(sums[:nn], cum_ref)
        cum_rows.append((rep, nn, err))
cum_df = pd.DataFrame(cum_rows, columns=[
    'replicate', 'n', 'cumulative_burden_cdf_error'
])
cum_df.to_csv(OUT / 'cumulative_burden_cdf_replicates.csv', index=False)

# For each replicate, take the largest CDF discrepancy across the 43
# weekly hospitalization marginals, then summarize across 35 replicates.
# This matches the J=43 simultaneous DKW bound in panel B.
ref_weekly_sorted = np.sort(ref, axis=0)
weekly_rows = []
rng_weekly = np.random.default_rng(SEED + 123)
for rep in range(REPS):
    sims, _ = trajectories(int(NS[-1]), rng_weekly)
    for nn in NS:
        nn = int(nn)
        max_error = max(
            cdf_discrepancy_at_sample(sims[:nn, h], ref_weekly_sorted[:, h])
            for h in range(H)
        )
        weekly_rows.append((rep, nn, max_error))
    print(f'Weekly max CDF replicate {rep + 1}/{REPS}', flush=True)
weekly_df = pd.DataFrame(weekly_rows, columns=[
    'replicate', 'n', 'max_cdf_error_across_43_weeks'
])
weekly_df.to_csv(OUT / 'weekly_max_cdf_replicates.csv', index=False)

# analytic curves
e_fixed=np.sqrt(np.log(2/ALPHA)/(2*n))
e_43=np.sqrt(np.log(2*H/ALPHA)/(2*n))
e_three=np.sqrt(np.log(2*3*H/ALPHA)/(2*n))

# summaries
peak_mag_grp=peak_df.groupby('n')['peak_magnitude_cdf_error']
peak_tim_grp=peak_df.groupby('n')['peak_timing_cdf_error']
peak_mag_med=peak_mag_grp.median().reindex(NS).to_numpy(); peak_mag_lo=peak_mag_grp.min().reindex(NS).to_numpy(); peak_mag_hi=peak_mag_grp.max().reindex(NS).to_numpy()
peak_tim_med=peak_tim_grp.median().reindex(NS).to_numpy(); peak_tim_lo=peak_tim_grp.min().reindex(NS).to_numpy(); peak_tim_hi=peak_tim_grp.max().reindex(NS).to_numpy()

cum_grp=cum_df.groupby('n')['cumulative_burden_cdf_error']
cum_med=cum_grp.median().reindex(NS).to_numpy(); cum_lo=cum_grp.min().reindex(NS).to_numpy(); cum_hi=cum_grp.max().reindex(NS).to_numpy()

weekly_grp=weekly_df.groupby('n')['max_cdf_error_across_43_weeks']
weekly_med=weekly_grp.median().reindex(NS).to_numpy()
weekly_lo=weekly_grp.min().reindex(NS).to_numpy()
weekly_hi=weekly_grp.max().reindex(NS).to_numpy()

fig,axs=plt.subplots(3,2,figsize=(12.8,13.8),gridspec_kw={'hspace':.64,'wspace':.29})
a,b,c,d,e,f=axs.flat

# A trajectories
regcolors={0:'#b9b9b9',1:'#c47a2c',2:'#2f7f7b',3:'#9a4f88'}
reglabels={0:'Ordinary regime',1:'High magnitude',2:'Delayed',3:'Rare early peak'}
for typ in range(4):
    for ii in np.where(spag_r==typ)[0]:
        a.plot(np.arange(1,H+1),spag[ii],color=regcolors[typ],alpha=.18 if typ==0 else .35,lw=.7)
handles=[Line2D([0],[0],color=regcolors[typ],lw=2,label=reglabels[typ]) for typ in range(4)]
for q,color,name in [(.5,'#2c6db2','Median'),(.9,'#d97c27','90th percentile'),(.99,'#a23e56','99th percentile')]:
    a.plot(np.arange(1,H+1),np.quantile(spag,q,axis=0),color=color,lw=2)
    handles.append(Line2D([0],[0],color=color,lw=2,label=name))
a.set(xlabel='Forecast horizon (week)',ylabel='Incident hospitalizations per 100,000')
a.text(.08,.95,'n = 180',transform=a.transAxes,va='top',fontsize=10)
a.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,1.02),ncol=3,frameon=False,fontsize=8.2,handlelength=1.6,columnspacing=.85)
a.grid(alpha=.13)

# B analytical DKW
for curve,label,color in [(e_fixed,'One outcome, one horizon','#355d87'),(e_43,'One outcome, 43 horizons','#368a82'),
                          (e_three,'Three outcomes, 43 horizons','#a37638')]:
    b.plot(n,curve,lw=1.9,color=color,label=label)
b.set(xlabel='Sampled trajectories, n',ylabel='Maximum CDF-error bound (probability)')
b.set_yscale('log'); b.set_ylim(0.012,.65)
b.yaxis.set_major_locator(LogLocator(base=10,subs=(1.,2.,5.)))
b.yaxis.set_major_formatter(FuncFormatter(decfmt)); b.yaxis.set_minor_formatter(NullFormatter())
b.grid(alpha=.15)
b.legend(loc='lower left',bbox_to_anchor=(0,1.02),ncol=2,frameon=False,fontsize=8.4,columnspacing=.85,handlelength=1.75)
style_x(b); add_n600(b); add_multiplier_lines_panelB(b)

# C,D,E scalar-outcome CDF errors
scalar_specs=[
    (c,'Peak magnitude','#a14b54',peak_mag_med,peak_mag_lo,peak_mag_hi),
    (d,'Peak timing','#355d87',peak_tim_med,peak_tim_lo,peak_tim_hi),
    (e,'Cumulative burden','#328479',cum_med,cum_lo,cum_hi),
]
for ax,title,color,med,lo,hi in scalar_specs:
    ax.plot(n,med,color=color,lw=2.3,label='Median maximum CDF error')
    ax.fill_between(n,lo,hi,color=color,alpha=.19,label='Range across 35 replicates')
    ax.plot(n,e_fixed,color='#8b4a78',ls='--',lw=1.65,label='One Outcome DKW Bound')
    ax.plot(n,e_three,color='#c67a41',ls=':',lw=1.65,label='Three Outcomes DKW Bound')
    ax.set(xlabel='Sampled trajectories, n',ylabel=f'{title} CDF error (probability)')
    ax.set_yscale('log'); ax.set_ylim(.002,.45)
    ax.yaxis.set_major_locator(LogLocator(base=10,subs=(1.,2.,5.)))
    ax.yaxis.set_major_formatter(FuncFormatter(decfmt)); ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(alpha=.14)
    ax.legend(loc='lower left',bbox_to_anchor=(0,1.02),ncol=2,frameon=False,fontsize=8.2,columnspacing=.85,handlelength=1.65)
    style_x(ax); add_n600(ax)

# F: simultaneous maximum empirical CDF error over 43 weekly marginals.
f.plot(n,weekly_med,color='#355d87',lw=2.3,label='Median maximum CDF error')
f.fill_between(n,weekly_lo,weekly_hi,color='#355d87',alpha=.19,label='Range across 35 replicates')
f.plot(n,e_43,color='#8b4a78',ls='--',lw=1.65,label='Simultaneous DKW bound (43 weeks)')
f.set(xlabel='Sampled trajectories, n',ylabel='Maximum CDF Error (probability)')
f.set_yscale('log'); f.set_ylim(.002,.45)
f.yaxis.set_major_locator(LogLocator(base=10,subs=(1.,2.,5.)))
f.yaxis.set_major_formatter(FuncFormatter(decfmt)); f.yaxis.set_minor_formatter(NullFormatter())
f.grid(alpha=.14)
f.legend(loc='lower left',bbox_to_anchor=(0,1.02),ncol=2,frameon=False,fontsize=8.0,columnspacing=.8,handlelength=1.6)
style_x(f); add_n600(f)

for ax,letter in zip([a,b,c,d,e,f],'ABCDEF'):
    ax.text(.012,.975,letter,transform=ax.transAxes,ha='left',va='top',fontsize=12,weight='bold',bbox=dict(facecolor='white',alpha=.9,edgecolor='none',pad=1.2))

png=OUT/'toy_DKW_figure.png'
pdf=OUT/'toy_DKW_figure.pdf'
fig.savefig(png,dpi=240,bbox_inches='tight')
fig.savefig(pdf,bbox_inches='tight')
plt.close(fig)
print(png)
print(pdf)
