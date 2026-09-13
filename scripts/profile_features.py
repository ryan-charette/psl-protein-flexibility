"""Prespecified timing pilot; selects structures without inspecting model outcomes."""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
import argparse,hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from psl_flexibility.dataset import read_structure,stable_hash
from generate_study_features import generate_one

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw',type=Path,default=ROOT/'data/raw/MDG_bfactor-main')
    p.add_argument('--workers',type=int,default=4)
    args=p.parse_args();frames={};info=[];excluded=[]
    for line in (args.raw/'datasets/list-365.txt').read_text().splitlines():
        if not line.strip():continue
        path=args.raw/'datasets/365'/line.strip()
        if path.suffix!='.pdb':path=path.with_suffix('.pdb')
        try:frame,_=read_structure(path)
        except ValueError as exc:excluded.append({'file':path.name,'reason':str(exc)});continue
        pid=frame.protein_id.iloc[0];frames[pid]=frame
        counts=cKDTree(frame[['x','y','z']]).query_ball_point(frame[['x','y','z']],13,return_length=True)
        info.append({'protein_id':pid,'n_residues':len(frame),'max_support':int(max(counts)),'hash':stable_hash(pid)})
    table=pd.DataFrame(info).sort_values(['max_support','protein_id']).reset_index(drop=True)
    table['stratum']=np.minimum(np.arange(len(table))*3//len(table),2)
    selected=set(table.groupby('stratum',group_keys=False).apply(lambda g:g.sort_values('hash').head(2)).protein_id)
    selected.add('1ULR')
    out=ROOT/'data/pilot_features';out.mkdir(parents=True,exist_ok=True)
    result=ROOT/'results/study';result.mkdir(parents=True,exist_ok=True)
    plan={'selection':'Two seeded-hash records from each maximum-support-size tercile, plus preset hero 1ULR',
          'proteins':sorted(selected),'eligible_proteins':len(table),'eligible_residues':int(table.n_residues.sum()),
          'maximum_support':int(table.max_support.max()),'degree1_budget_seconds':21600,'projection_safety_factor':2.,
          'projected_workers':args.workers}
    (result/'pilot_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    table.to_csv(result/'pilot_sampling_frame.csv',index=False)
    timings=[];start=time.perf_counter()
    for pid in sorted(selected):
        row=generate_one((pid,frames[pid],str(out),True,'pilot-v1'))
        timings.append(row);print(json.dumps(row),flush=True)
        pd.DataFrame(timings).to_csv(result/'pilot_timing.csv',index=False)
    perres=[t['seconds']/t['n_residues'] for t in timings if not t['cached']]
    if perres:
        projection=2*max(perres)*int(table.n_residues.sum())/args.workers
        decision={**plan,'wall_seconds':time.perf_counter()-start,'max_seconds_per_residue':max(perres),
                  'conservative_full_suite_projection_seconds':projection,
                  'decision':'all' if projection<=21600 else 'nested60',
                  'note':'Pilot includes degree-zero, graph, native, degree-one operators, eigensolvers and serialization. Full-suite projection upper-bounds degree-one-only cost.'}
        (result/'pilot_decision.json').write_text(json.dumps(decision,indent=2)+'\n');print(json.dumps(decision),flush=True)

if __name__=='__main__':main()
