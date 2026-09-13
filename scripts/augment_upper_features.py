"""Add degree-one upper-endpoint controls without changing existing feature blocks."""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
import argparse,hashlib,json,sys,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits
from psl_flexibility.sheaf import AlphaSheaf,spectral_statistics,STATISTIC_NAMES

def one(task):
    pid,frame,directory,codehash=task;out=Path(directory)/f'{pid}.npz';start=time.perf_counter()
    with np.load(out,allow_pickle=False) as z:payload={k:z[k] for k in z.files}
    if not np.array_equal(payload['residue_keys'],frame.residue_key.to_numpy(str)):raise ValueError(f'{pid}: residue mapping mismatch')
    if 'center_upper1' in payload:
        if str(payload['upper_source_sha256'])!=codehash:raise ValueError('Upper feature code changed')
        return {'protein_id':pid,'seconds':0.,'cached':True}
    values={f'{kind}_upper1':[] for kind in ('identity','geometric','center')}
    coords=frame[['x','y','z']].to_numpy(float);d=cdist(coords,coords)
    with threadpool_limits(limits=1):
        for i in range(len(coords)):
            ids=np.r_[i,np.flatnonzero((d[i]<=13.+1e-12)&(np.arange(len(coords))!=i))]
            alpha=AlphaSheaf(coords[ids],kind='identity')
            for name,kind in [('identity','identity'),('geometric','geometric'),('center','center_zero')]:
                sheaf=alpha if name=='identity' else alpha.with_kind(kind)
                values[f'{name}_upper1'].append(np.concatenate([spectral_statistics(sheaf.laplacian(1,a)) for a in (4.,6.)]))
    for key,rows in values.items():
        payload[key]=np.asarray(rows);payload[f'{key}_names']=np.asarray([f'{key}_a{a}_b{a}_d1_{s}' for a in (4,6) for s in STATISTIC_NAMES])
    payload['upper_source_sha256']=np.asarray(codehash)
    temp=out.with_suffix('.upper.tmp.npz');np.savez_compressed(temp,**payload);temp.replace(out)
    return {'protein_id':pid,'n_residues':len(frame),'seconds':time.perf_counter()-start,'cached':False}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,default=ROOT/'results/study/dataset');p.add_argument('--features',type=Path,default=ROOT/'data/features');p.add_argument('--workers',type=int,default=4);args=p.parse_args()
    table=pd.read_csv(args.manifest/'residues.csv',keep_default_na=False)
    codehash=hashlib.sha256(Path(__file__).read_bytes()+(ROOT/'src/psl_flexibility/sheaf.py').read_bytes()).hexdigest()
    tasks=[(pid,g.sort_values('row_index'),str(args.features),codehash) for pid,g in table.groupby('protein_id',sort=True)]
    config={'purpose':'Disentangle two-scale persistence from access to upper-radius ordinary degree-one information','pairs':[[4,4],[6,6]],'support':13,'source_sha256':codehash,'selection':'full audited cohort, all three sheaf constructions','decided':'before predictive evaluation'}
    (args.features/'upper_control_config.json').write_text(json.dumps(config,indent=2)+'\n')
    rows=[];start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for f in as_completed([pool.submit(one,t) for t in tasks]):
            row=f.result();rows.append(row);print(f'upper {len(rows)}/{len(tasks)} {row["protein_id"]} {row["seconds"]:.1f}s',flush=True)
            pd.DataFrame(rows).to_csv(args.manifest.parent/'upper_timing.csv',index=False)
    print(json.dumps({'wall_seconds':time.perf_counter()-start,'worker_seconds':sum(r['seconds'] for r in rows)}))

if __name__=='__main__':main()
