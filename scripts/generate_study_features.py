"""Generate the prespecified coordinate-only feature families, with resumable caches."""
from __future__ import annotations

import os
for _key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[_key]='1'
import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits
from psl_flexibility.sheaf import AlphaSheaf, spectral_statistics

STATS=['nullity','min_positive','max','mean','median','std']
KINDS={'identity':'identity','geometric':'geometric','center':'center_zero'}
RADII=(3.,4.,5.,6.)
PAIRS=((3.,3.),(5.,5.),(3.,4.),(5.,6.))

def graph_features(points):
    d=cdist(points,points);np.fill_diagonal(d,np.inf)
    n=len(points);cols=[];names=[]
    for r in (6,8,10,12):
        c=(d<=r).sum(1)
        cols.extend([c,c/max(n-1,1)])
        names.extend([f'contact_count_r{r}',f'contact_fraction_r{r}'])
    for r in (8,12):
        adj=d<=r;clustering=[]
        for row in adj:
            ids=np.flatnonzero(row);k=len(ids)
            clustering.append(adj[np.ix_(ids,ids)].sum()/(k*(k-1)) if k>1 else 0.)
        cols.append(clustering);names.append(f'clustering_r{r}')
    ordered=np.sort(d,axis=1)
    for k in (4,8,12):
        cols.append(ordered[:,:min(k,n-1)].mean(1));names.append(f'mean_distance_nearest{k}')
    return np.column_stack(cols),names

def native_features(points):
    d=cdist(points,points);rows=[]
    for i in range(len(points)):
        row=[]
        for r in (6,9,12):
            ids=np.r_[i,np.flatnonzero((d[i]<=r+1e-12)&(np.arange(len(points))!=i))]
            sub=d[np.ix_(ids,ids)]
            adj=(sub<=r+1e-12).astype(float);np.fill_diagonal(adj,0)
            adj[0,:]*=2;adj[:,0]*=2
            row.extend(spectral_statistics(np.linalg.eigvalsh(np.diag(adj.sum(1))-adj)))
        rows.append(row)
    return np.asarray(rows),[f'native_r{r}_d0_{s}' for r in (6,9,12) for s in STATS]

def generate_one(task):
    protein_id,records,out_dir,degree1,signature=task
    started=time.perf_counter();out=Path(out_dir)/f'{protein_id}.npz'
    points=records[['x','y','z']].to_numpy(float)
    with threadpool_limits(limits=1):
        if out.exists():
            with np.load(out,allow_pickle=False) as z:
                payload={k:z[k] for k in z.files}
            if str(payload.get('source_signature',''))!=signature:
                raise ValueError(f'{protein_id}: stale cache signature; use a new output directory')
            if 'center0' in payload and (not degree1 or 'center_p1' in payload):
                return {'protein_id':protein_id,'n_residues':len(points),'seconds':0.,'cached':True}
        else:
            payload={'residue_keys':records.residue_key.to_numpy(str),'source_signature':np.asarray(signature)}
        if 'graph' not in payload:
            g,names=graph_features(points);payload.update(graph=g,graph_names=np.asarray(names))
            n,names=native_features(points);payload.update(native0=n,native0_names=np.asarray(names))
        collect={f'{k}0':[] for k in KINDS} if 'center0' not in payload else {}
        if degree1:collect.update({f'{k}{suffix}':[] for k in KINDS for suffix in ('1','_p1')})
        d=cdist(points,points);counts=[]
        for i in range(len(points)):
            ids=np.r_[i,np.flatnonzero((d[i]<=13.+1e-12)&(np.arange(len(points))!=i))]
            alpha=AlphaSheaf(points[ids],kind='identity')
            counts.append([len(ids),len(alpha.simplices(6.,1)),len(alpha.simplices(6.,2))])
            for name,kind in KINDS.items():
                sheaf=alpha if kind=='identity' else alpha.with_kind(kind)
                if f'{name}0' in collect:
                    collect[f'{name}0'].append(np.concatenate([spectral_statistics(np.linalg.eigvalsh(sheaf.laplacian(0,a))) for a in RADII]))
                if degree1:
                    vals=[spectral_statistics(np.linalg.eigvalsh(sheaf.laplacian(1,a,b))) for a,b in PAIRS]
                    collect[f'{name}1'].append(np.concatenate(vals[:2]));collect[f'{name}_p1'].append(np.concatenate(vals[2:]))
        for key,values in collect.items():
            payload[key]=np.asarray(values)
            if key.endswith('0'):
                payload[f'{key}_names']=np.asarray([f'{key}_a{a:g}_d0_{s}' for a in RADII for s in STATS])
            else:
                pairs=PAIRS[2:] if '_p1' in key else PAIRS[:2]
                payload[f'{key}_names']=np.asarray([f'{key}_a{a:g}_b{b:g}_d1_{s}' for a,b in pairs for s in STATS])
        payload['simplex_counts']=np.asarray(counts,dtype=int)
        payload['simplex_counts_names']=np.asarray(['support_vertices','alpha_edges_a6','alpha_triangles_a6'])
        for key in collect:
            if not np.isfinite(payload[key]).all():raise ValueError(f'{protein_id}: nonfinite {key}')
        temp=out.with_suffix('.tmp.npz');np.savez_compressed(temp,**payload);temp.replace(out)
    return {'protein_id':protein_id,'n_residues':len(points),'seconds':time.perf_counter()-started,'cached':False,
            'max_support':int(max(c[0] for c in counts)),'max_edges':int(max(c[1] for c in counts))}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest',type=Path,default=ROOT/'results/study/dataset')
    ap.add_argument('--out',type=Path,default=ROOT/'data/features')
    ap.add_argument('--degree1',action='store_true')
    ap.add_argument('--ids',help='Comma-separated protein identifiers')
    ap.add_argument('--cohort',choices=['all','nested60','nested20'],default='all')
    ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--timing-name',default='feature_timing.csv')
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    residues=pd.read_csv(args.manifest/'residues.csv',keep_default_na=False)
    proteins=pd.read_csv(args.manifest/'proteins.csv',keep_default_na=False)
    selected=set(proteins.protein_id)
    if args.cohort!='all':
        selected=set(proteins.loc[proteins[args.cohort].astype(str).str.lower().isin(['true','1']), 'protein_id'])
    if args.ids:selected &= set(args.ids.split(','))
    tasks=[]
    # Cache identity includes both the residue table and all feature-source modules.
    # Degree-one is an additive family under the same geometric definition.
    codehash=hashlib.sha256()
    for path in [Path(__file__),ROOT/'src/psl_flexibility/sheaf.py']:
        codehash.update(path.read_bytes())
    for pid,group in residues.groupby('protein_id',sort=True):
        if pid not in selected:continue
        group=group.sort_values('row_index')
        signature=hashlib.sha256((codehash.hexdigest()+group.to_csv(index=False)).encode()).hexdigest()
        tasks.append((pid,group,str(args.out),args.degree1,signature))
    if not tasks:raise ValueError('Empty selected cohort')
    config={'support_angstrom':13,'radii0':RADII,'pairs1':PAIRS,'kinds':KINDS,'stats':STATS,
            'degree1':args.degree1,'selected_proteins':[t[0] for t in tasks],'workers':args.workers,
            'code_sha256':codehash.hexdigest()}
    (args.out/f'feature_config_{"d1" if args.degree1 else "d0"}.json').write_text(json.dumps(config,indent=2)+'\n')
    timings=[];start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(generate_one,t) for t in tasks]
        for future in as_completed(futures):
            row=future.result();timings.append(row)
            print(f'{len(timings)}/{len(tasks)} {row["protein_id"]}: {row["seconds"]:.1f}s; wall {time.perf_counter()-start:.1f}s',flush=True)
            pd.DataFrame(timings).to_csv(args.manifest.parent/args.timing_name,index=False)
    print(json.dumps({'wall_seconds':time.perf_counter()-start,'proteins':len(tasks),'worker_seconds':sum(t['seconds'] for t in timings)}),flush=True)

if __name__=='__main__':main()
