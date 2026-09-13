"""Rebuild manuscript figures from keyed results and explicit mathematical examples."""
from __future__ import annotations
import argparse,json,sys,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'data/processed/mpl-cache'))
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap,TwoSlopeNorm
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection,Line3DCollection
from psl_flexibility.sheaf import AlphaSheaf
from summarize_study import read as verified_result,check_case_inputs,sha256_file,SOURCES

TEAL='#007f86';ORANGE='#db6a32';INK='#142e3b';GRAY='#6e7d86'
CMAP=LinearSegmentedColormap.from_list('contribution',[TEAL,'#f1f2ec',ORANGE])
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.titlesize':12,
    'axes.labelsize':12,'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':INK,
    'text.color':INK,'xtick.color':GRAY,'ytick.color':GRAY,'pdf.fonttype':42,'svg.fonttype':'none',
    'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white'})
OUT=ROOT/'paper/figures';OUT.mkdir(parents=True,exist_ok=True)
STUDY=ROOT/'results/study'
LABELS={'graph':'Graph geometry','native0':'Legacy graph spectra','graph+native0':'Graph + legacy spectra',
    'center0':'Center-zero spectra','graph+identity0':'Graph + identity sheaf',
    'graph+geometric0':'Graph + geometric sheaf','graph+center0':'Graph + center-zero sheaf'}
ORDER=list(LABELS)

def save(fig,name):
    for ext in ('pdf','svg','png'):
        fig.savefig(OUT/f'{name}.{ext}',dpi=400,bbox_inches='tight',pad_inches=.12)
    plt.close(fig)

def title(ax,letter,label):
    ax.set_title(f'{letter}   {label}',loc='left',fontweight='bold',pad=12,
                 fontsize=11 if ax.figure.get_figwidth()<8 else 12)

def cloud(ax,pts,sheaf=None,radius=None,center=True):
    if sheaf is not None:
        edges=sheaf.simplices(radius,1);triangles=sheaf.simplices(radius,2)
        if triangles:ax.add_collection3d(Poly3DCollection([pts[list(s)] for s in triangles],facecolor=TEAL,alpha=.09,edgecolor='none'))
        if edges:ax.add_collection3d(Line3DCollection([pts[list(e)] for e in edges],colors=GRAY,linewidths=.6,alpha=.55))
    ax.scatter(*pts.T,s=12,c=TEAL,depthshade=False,edgecolors='white',linewidths=.3)
    if center:ax.scatter(*pts[0],s=85,c=ORANGE,edgecolors='white',linewidths=1,depthshade=False)
    span=np.ptp(pts,axis=0).max()/2*1.1;mid=(pts.max(0)+pts.min(0))/2
    for fun,c in zip((ax.set_xlim,ax.set_ylim,ax.set_zlim),mid):fun(c-span,c+span)
    ax.set_box_aspect((1,1,1));ax.view_init(24,-62);ax.set_axis_off()

def load_case(pid='1ULR'):
    residues=pd.read_csv(STUDY/'dataset/residues.csv',keep_default_na=False)
    return residues[residues.protein_id==pid].sort_values('row_index').reset_index(drop=True)

def theory_figures():
    case=load_case();pts=case[['x','y','z']].to_numpy(float)
    idx=int(np.argmin(abs(case.residue_number.to_numpy()-43)))
    d=np.linalg.norm(pts-pts[idx],axis=1);ids=np.r_[idx,np.flatnonzero((d<=13)&(np.arange(len(pts))!=idx))]
    local=pts[ids];s=AlphaSheaf(local,kind='center_zero')
    fig=plt.figure(figsize=(10.8,6.2));grid=fig.add_gridspec(2,3,hspace=.45,wspace=.42)
    ax=fig.add_subplot(grid[0,0],projection='3d');ax.plot(*pts.T,color='#acb8bc',lw=1.6)
    cloud(ax,local);title(ax,'a','The 13 Å neighborhood')
    for j,a in enumerate((3.,4.),start=1):
        ax=fig.add_subplot(grid[0,j],projection='3d');cloud(ax,local,s,a)
        title(ax,chr(97+j),f'Alpha complex: {a:g} Å')
        ax.text2D(.03,-.04,f'{len(s.simplices(a,1))} edges · {len(s.simplices(a,2))} triangles',transform=ax.transAxes,color=GRAY,fontsize=12)
    ax=fig.add_subplot(grid[1,0]);ax.axis('off');title(ax,'d','Compatible restrictions')
    text=(r'$q_c=0,\qquad q_v=1\;(v\ne c)$'+'\n\n'+r'$\rho_{\sigma,\tau}=\frac{F(\sigma)}{F(\tau)}\prod_{v\in\tau\setminus\sigma}q_v$'+'\n\n'+r'$D^1D^0=0$'+'\n\n'+r'$F(v)=1,\quad F(ij)=d_{ij}$')
    ax.text(.02,.87,text,va='top',fontsize=13,color=INK)
    ax=fig.add_subplot(grid[1,1]);L=s.laplacian(0,4.);limit=float(abs(L).max())
    im=ax.imshow(L,cmap=CMAP,norm=TwoSlopeNorm(vcenter=0,vmin=-limit,vmax=limit),interpolation='nearest')
    title(ax,'e','Center-zero operator');ax.set_xlabel('Local vertex');ax.set_ylabel('Local vertex')
    fig.colorbar(im,cax=ax.inset_axes([.15,-.36,.70,.045]),orientation='horizontal',ticks=[-limit,0,limit],format='%.2g')
    ax=fig.add_subplot(grid[1,2])
    for a,c in ((3.,TEAL),(4.,ORANGE)):
        e=np.linalg.eigvalsh(s.laplacian(0,a));ax.plot(np.arange(1,len(e)+1),e,marker='o',ms=2.8,lw=1.3,color=c,label=f'a = {a:g} Å')
    title(ax,'f','Spectral descriptors');ax.set_xlabel('Ordered eigenvalue');ax.set_ylabel('Eigenvalue');ax.legend(frameon=False);ax.grid(axis='y',alpha=.13)
    save(fig,'fig2_pipeline')
    points=np.array([[0.,0.,0.],[2,0,0],[0,2,0],[-2,0,0],[0,-2,0]])
    alpha=AlphaSheaf(points,kind='center_zero');L=alpha.laplacian(0,3.)
    dist=np.linalg.norm(points[:,None,:]-points[None,:,:],axis=2);A=(dist<=3).astype(float)
    np.fill_diagonal(A,0);A[0]*=2;A[:,0]*=2;native=np.diag(A.sum(1))-A
    fig=plt.figure(figsize=(10.8,5.8));g=fig.add_gridspec(2,3,height_ratios=[1.2,.8],wspace=.4,hspace=.5)
    ax=fig.add_subplot(g[0,0])
    for i in range(5):
        for j in range(i+1,5):
            if dist[i,j]<=3:ax.plot(points[[i,j],0],points[[i,j],1],color=ORANGE if i==0 else TEAL,lw=2)
    ax.scatter(points[:,0],points[:,1],s=[150]+[95]*4,c=[ORANGE]+[TEAL]*4,edgecolors='white',zorder=3)
    for i,(x,y,_) in enumerate(points):ax.text(x+.15,y+.12,'c' if i==0 else str(i),fontweight='bold')
    ax.set_aspect('equal');ax.axis('off');title(ax,'a','Center and neighbors')
    for col,m,label in [(1,native,'Historical graph'),(2,L,'Center-zero sheaf')]:
        ax=fig.add_subplot(g[0,col]);limit=abs(m).max();ax.imshow(m,cmap=CMAP,vmin=-limit,vmax=limit)
        for (i,j),v in np.ndenumerate(m):
            label_value={-.125:r'$-\frac{1}{8}$',.25:r'$\frac{1}{4}$'}.get(round(float(v),10),f'{v:g}')
            ax.text(j,i,label_value,ha='center',va='center',fontsize=13,color=INK if abs(v)<.7*limit else 'white')
        ax.set_xticks(range(5),['c','1','2','3','4']);ax.set_yticks(range(5),['c','1','2','3','4']);title(ax,chr(97+col),label)
        if col==2:ax.axhline(.5,color=ORANGE,lw=1.8);ax.axvline(.5,color=ORANGE,lw=1.8)
    ax=fig.add_subplot(g[1,0]);title(ax,'d','Different spectra')
    ax.plot(range(5),np.linalg.eigvalsh(native),'o-',c=ORANGE,label='Historical graph')
    ax.plot(range(5),np.linalg.eigvalsh(L),'s-',c=TEAL,label='Geometric sheaf')
    ax.set_xlabel('Ordered eigenvalue');ax.set_ylabel('Eigenvalue');ax.legend(frameon=False,fontsize=11)
    ax=fig.add_subplot(g[1,1:]);ax.axis('off');title(ax,'e','Interpreting the blocks')
    ax.text(0,.82,r'$L_0=\left[\sum_{j\sim c}d_{cj}^{-2}\right]\;\oplus\;L_{\mathrm{neighbors}}$',fontsize=16,va='center')
    ax.text(0,.40,'Center: inverse-square packing',fontsize=12,va='top')
    ax.text(0,.18,'Neighbors: connectivity and weighted geometry',fontsize=12,va='top')
    ax.text(0,-.06,'Higher degrees use the augmented, weighted link.',fontsize=12,color=GRAY)
    save(fig,'fig3_decomposition')
    (STUDY/'theory_figure_inputs.json').write_text(json.dumps({'case':'1ULR','center_residue':int(case.iloc[idx].residue_number),'support_residue_keys':case.iloc[ids].residue_key.tolist(),'example_points':points.tolist(),'native_laplacian':native.tolist(),'sheaf_laplacian':L.tolist()},indent=2)+'\n',encoding='utf-8')

def paired(table,left,right,protocol='cluster',metric='pcc'):
    f=table[(table.protocol==protocol)&(table.metric==metric)];row=f[(f.left_mode==left)&(f.right_mode==right)]
    if len(row):
        r=row.iloc[0];return np.array([r.mean_difference,r.ci95_low,r.ci95_high])
    r=f[(f.left_mode==right)&(f.right_mode==left)].iloc[0]
    return np.array([-r.mean_difference,-r.ci95_high,-r.ci95_low])

def interval(ax,y,estimate,color=TEAL,marker='o',label=None):
    if not np.isfinite(estimate).all():raise ValueError('Cannot plot nonfinite estimates as completed comparisons')
    mean,lo,hi=estimate;ax.plot([lo,hi],[y,y],color=color,lw=1.7);ax.plot(mean,y,marker,color=color,ms=5,label=label)

def performance_figure():
    summary=verified_result('core','summary.csv');per=verified_result('core','per_protein.csv');pairs=verified_result('core','paired_differences.csv')
    fig,axs=plt.subplots(2,2,figsize=(10.8,7.6),gridspec_kw={'hspace':.55,'wspace':.95})
    ax=axs[0,0];title(ax,'a','Mean held-out correlation')
    for y,mode in enumerate(ORDER):
        for protocol,offset,color,marker in [('cluster',-.1,TEAL,'o'),('protein',.1,GRAY,'s')]:
            r=summary[(summary.protocol==protocol)&(summary['mode']==mode)].iloc[0]
            interval(ax,y+offset,[r.mean_pcc,r.pcc_ci95_low,r.pcc_ci95_high],color,marker)
    ax.set_yticks(range(len(ORDER)),[LABELS[m] for m in ORDER],fontsize=12);ax.invert_yaxis()
    ax.set_xlabel('Mean per-protein Pearson correlation')
    ax.legend(handles=[Line2D([],[],color=TEAL,marker='o',label='Sequence-controlled'),Line2D([],[],color=GRAY,marker='s',label='Protein-only')],frameon=False,fontsize=11,loc='lower left')
    ax.set_ylim(8.6,-.5)
    ax.grid(axis='x',alpha=.13)
    ax=axs[0,1];title(ax,'b','Change from graph geometry')
    for y,mode in enumerate(ORDER[1:]):interval(ax,y,paired(pairs,'graph',mode),ORANGE if mode=='graph+center0' else TEAL)
    ax.axvline(0,color=GRAY,lw=1,ls='--');ax.set_yticks(range(6),[LABELS[m] for m in ORDER[1:]],fontsize=12);ax.invert_yaxis()
    ax.set_xlabel('Paired change in Pearson correlation');ax.grid(axis='x',alpha=.13)
    p=per[per.protocol=='cluster'].pivot(index='protein_id',columns='mode',values='pcc')
    ax=axs[1,0];title(ax,'c','Each point is a held-out protein')
    ax.plot([-1,1],[-1,1],color=GRAY,lw=1,ls='--');ax.scatter(p['graph'],p['graph+center0'],s=13,c=TEAL,alpha=.45,edgecolors='none')
    for pid in ('1ULR','1X3O'):
        x,y=p.loc[pid,['graph','graph+center0']];ax.scatter(x,y,c=ORANGE,s=45,zorder=4,edgecolors='white')
        ax.annotate(pid,(x,y),xytext=(25,-24) if pid=='1ULR' else (-55,22),textcoords='offset points',fontsize=12,arrowprops={'arrowstyle':'-','color':ORANGE,'lw':.8},bbox={'facecolor':'white','edgecolor':'none','alpha':.8,'pad':1})
    minimum=float(p[['graph','graph+center0']].min().min());low=max(-1.,np.floor(minimum*10)/10-.05)
    ax.set_xlim(low,1);ax.set_ylim(low,1);ax.set_xlabel('Graph geometry: protein PCC');ax.set_ylabel('Graph + center-zero: protein PCC')
    ax=axs[1,1];title(ax,'d','Changes across proteins')
    delta=(p['graph+center0']-p['graph']).dropna();extent=max(.05,float(delta.abs().max())*1.08)
    ax.hist(delta,bins=np.linspace(-extent,extent,31),color=TEAL,alpha=.85,edgecolor='white',linewidth=.5)
    ax.axvline(0,color=GRAY,lw=1,ls='--');ax.axvline(delta.mean(),color=ORANGE,lw=1.7)
    ax.text(.97,.94,f'{sum(delta>0)} / {len(delta)} improve\nMean change {delta.mean():+.4f}',transform=ax.transAxes,ha='right',va='top',fontsize=12)
    ax.set_ylim(0,ax.get_ylim()[1]*1.3)
    ax.set_xlabel('Center-zero addition: change in PCC');ax.set_ylabel('Number of proteins')
    save(fig,'fig4_performance')

def persistence_figure():
    summary=verified_result('d1','summary.csv');pairs=verified_result('d1','paired_differences.csv')
    fig,axs=plt.subplots(2,2,figsize=(10.8,7.4),gridspec_kw={'wspace':.9,'hspace':.6})
    ax=axs[0,0];title(ax,'a','Degree one: matched additions')
    for y,kind in enumerate(('identity','geometric','center')):
        base=f'graph+{kind}0'
        for shift,suffix,c,label in [(-.22,'1',GRAY,'Ordinary lower: (3,3), (5,5)'),(0,'_upper1',ORANGE,'Ordinary upper: (4,4), (6,6)'),(.22,'_p1',TEAL,'Persistent: (3,4), (5,6)')]:
            interval(ax,y+shift,paired(pairs,base,base+'+'+kind+suffix),c,label=label if y==0 else None)
    ax.axvline(0,c=GRAY,ls='--',lw=1);ax.set_yticks(range(3),['Identity','Geometric','Center-zero']);ax.invert_yaxis()
    ax.set_xlabel('PCC change from graph + degree zero')
    ax.legend(frameon=True,facecolor='white',edgecolor='none',framealpha=1,fontsize=10.5,loc='lower right');ax.set_ylim(4.1,-.45)
    ax=axs[0,1];title(ax,'b','Persistence versus upper endpoint')
    for y,kind in enumerate(('identity','geometric','center')):
        base=f'graph+{kind}0';interval(ax,y,paired(pairs,base+'+'+kind+'_upper1',base+'+'+kind+'_p1'),TEAL)
    ax.axvline(0,c=GRAY,ls='--',lw=1);ax.set_yticks(range(3),['Identity','Geometric','Center-zero']);ax.invert_yaxis()
    ax.set_xlabel('Persistent minus ordinary-upper PCC');ax.grid(axis='x',alpha=.13)
    ax=axs[1,0];title(ax,'c','Scale and spectrum ablations');records=[]
    full=verified_result('core','summary.csv');r=full[(full.protocol=='cluster')&(full['mode']=='graph+center0')].iloc[0]
    records.append(('All scales, all summaries',r))
    for name,label in [('nullity','Nullities only'),('positive','Positive spectra only')]+[(f'a{a}',f'Single scale: {a} Å') for a in (3,4,5,6)]:
        records.append((label,verified_result(f'ablation_{name}','summary.csv').iloc[0]))
    for y,(label,r) in enumerate(records):interval(ax,y,[r.mean_pcc,r.pcc_ci95_low,r.pcc_ci95_high],ORANGE if y==0 else TEAL)
    ax.set_yticks(range(len(records)),[x[0] for x in records],fontsize=12);ax.invert_yaxis()
    ax.set_xlabel('Mean protein PCC, graph + center-zero');ax.grid(axis='x',alpha=.13)
    ax=axs[1,1];title(ax,'d','Measured feature cost')
    names=['Degree-zero pipeline','Degree-one addition','Upper-endpoint control']
    times=[float(pd.read_csv(STUDY/f).seconds.sum())/60 for f in ('d0_timing.csv','d1_timing.csv','upper_timing.csv')]
    ax.barh(range(3),times,color=[GRAY,TEAL,ORANGE],height=.48);ax.set_yticks(range(3),names,fontsize=12);ax.invert_yaxis()
    ax.set_xlabel('Summed worker minutes (four workers)')
    for y,x in enumerate(times):ax.text(x+max(times)*.02,y,f'{x:.1f}',va='center',fontsize=12)
    ax.set_xlim(0,max(times)*1.22);ax.text(0,-.39,f'Full cohort: {int(summary.n_proteins.iloc[0])} structures',transform=ax.transAxes,fontsize=12,color=GRAY)
    save(fig,'fig5_persistence')

def hero_figure(pid='1ULR',readme=False):
    data=pd.read_csv(STUDY/f'cases/{pid}_values.csv',keep_default_na=False);audit=check_case_inputs(pid,render=True)
    fig=plt.figure(figsize=(10.8,5.25) if readme else (7.0,5.6))
    top=.83 if readme else .90;bottom=.19 if readme else .43
    if readme:
        fig.text(.035,.955,'Local sheaf spectra, on a real protein',fontsize=21,fontweight='bold',color=INK)
        fig.text(.035,.89,f'{pid}  ·  {len(data)} residues  ·  Predictions held out by sequence cluster',fontsize=11,color=GRAY)
    for k,(name,label) in enumerate([('observed','Observed B-factor'),('predicted','Held-out prediction'),('shap','Sheaf contribution')]):
        ax=fig.add_axes([.01+k/3,bottom,.32,top-bottom]);ax.imshow(plt.imread(OUT/f'renders/{pid}_{name}.png'));ax.axis('off')
        ax.text(.5,1.01,('' if readme else chr(97+k)+'   ')+label,ha='center',va='bottom',transform=ax.transAxes,fontsize=11,fontweight='bold')
        if k==1:
            corr=np.corrcoef(data.z_b_factor,data.prediction_z)[0,1]
            ax.text(.5,-.025,f'Protein PCC = {corr:.3f}',ha='center',transform=ax.transAxes,fontsize=9,color=GRAY)
    for left,width,label,field in [(.15,.39,'Within-protein standard deviations','observed'),(.746,.17,'Signed SHAP, B-factor z units','shap')]:
        limit=audit['rendering'][field]['mapping']['color_limits'][1];cax=fig.add_axes([left,bottom-.075,width,.014])
        fig.colorbar(ScalarMappable(norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit),cmap=CMAP),cax=cax,orientation='horizontal',ticks=[-limit,0,limit],format='%.2g')
        cax.set_xlabel(label,fontsize=8,labelpad=2);cax.tick_params(labelsize=8,pad=1,length=2)
    if readme:
        fig.text(.035,.02,"Teal: lower values. Orange: higher values. SHAP colors show the sheaf group's model contribution.",fontsize=9,color=GRAY)
        save(fig,'readme_hero');return
    ax=fig.add_axes([.075,.06,.58,.17]);title(ax,'d','A residue-by-residue check')
    ax.plot(data.residue_number,data.z_b_factor,c=GRAY,lw=1.2,label='Observed')
    ax.plot(data.residue_number,data.prediction_z,c=TEAL,lw=1.5,label='Prediction')
    ax.set_xlabel('Residue number',fontsize=10.5);ax.set_ylabel('B-factor z score',fontsize=10.5)
    ax.tick_params(labelsize=10.5);ax.legend(frameon=False,fontsize=8,ncol=2,loc='upper center');ax.grid(axis='y',alpha=.13)
    case=load_case(pid);pts=case[['x','y','z']].to_numpy(float);center=int(np.argmin(abs(case.residue_number.to_numpy()-43)))
    dist=np.linalg.norm(pts-pts[center],axis=1);ids=np.r_[center,np.flatnonzero((dist<=13)&(np.arange(len(pts))!=center))]
    local=pts[ids];s=AlphaSheaf(local,kind='center_zero')
    ax=fig.add_axes([.72,.01,.24,.23],projection='3d');cloud(ax,local,s,4);title(ax,'e','A local alpha complex')
    ax.text2D(.02,-.06,f'Residue {int(case.iloc[center].residue_number)} · a = 4 Å · support = 13 Å',transform=ax.transAxes,fontsize=8,color=GRAY)
    save(fig,'fig1_protein' if pid=='1ULR' else 'supp_1X3O')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ('theory','results','cases','all'):p.add_argument('--'+flag,action='store_true')
    args=p.parse_args()
    if args.theory or args.all:theory_figures()
    if args.results or args.all:performance_figure();persistence_figure()
    if args.cases or args.all:hero_figure();hero_figure('1X3O');hero_figure(readme=True)
    manifest={'source_sha256':{str(p.relative_to(ROOT)).replace('\\','/'):sha256_file(p) for p in [Path(__file__),ROOT/'scripts/summarize_study.py',ROOT/'scripts/render_protein.py',ROOT/'src/psl_flexibility/sheaf.py']},
        'data_sha256':{**SOURCES,'results/study/dataset/residues.csv':sha256_file(STUDY/'dataset/residues.csv')},
        'exports_sha256':{str(p.relative_to(ROOT)).replace('\\','/'):sha256_file(p) for p in OUT.iterdir() if p.suffix in {'.pdf','.svg','.png'}}}
    (STUDY/'figure_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':main()
