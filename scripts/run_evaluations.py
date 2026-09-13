"""Run the fixed primary, persistence-control, and degree-zero ablation comparisons."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from psl_flexibility.evaluation import run_evaluation,sha256_file

CORE=['graph+center0','graph','native0','graph+native0','center0','graph+identity0','graph+geometric0']
D1=[m for kind in ('identity','geometric','center') for m in (
    f'graph+{kind}0',f'graph+{kind}0+{kind}1',
    f'graph+{kind}0+{kind}_p1',f'graph+{kind}0+{kind}_upper1')]
ABLATIONS={
    'nullity':r'^graph:|_nullity$',
    'positive':r'^graph:|_(?:min_positive|max|mean|median|std)$',
    **{f'a{a}':rf'^graph:|center0_a{a}_' for a in (3,4,5,6)},
}

def intact(path):
    cfgpath=path/'evaluation_config.json'
    if not cfgpath.exists():return False
    cfg=json.loads(cfgpath.read_text())
    if not cfg.get('complete'):return False
    for name,digest in cfg['manifest_sha256'].items():
        if sha256_file(ROOT/'results/study/dataset'/name)!=digest:raise ValueError(f'Stale manifest in {path}')
    for info in cfg['feature_files'].values():
        if sha256_file(Path(info['path']))!=info['sha256']:raise ValueError(f'Stale features in {path}')
    for name,digest in cfg['outputs_sha256'].items():
        if sha256_file(path/name)!=digest:raise ValueError(f'Changed output in {path}')
    return True

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--parts',default='core,d1,ablations')
    ap.add_argument('--jobs',type=int,default=4)
    ap.add_argument('--resume',action='store_true',help='Reuse only completed evaluations whose recorded hashes still agree')
    args=ap.parse_args();parts=args.parts.split(',')
    if set(parts)-{'core','d1','ablations'}:raise ValueError('Unknown evaluation part')
    runs=[]
    if 'core' in parts:runs.append(('core',dict(modes=CORE,protocols=('cluster','protein'),phase='core')))
    if 'd1' in parts:runs.append(('d1',dict(modes=D1,protocols=('cluster',),phase='d1')))
    if 'ablations' in parts:
        runs.extend((f'ablation_{name}',dict(modes=['graph+center0'],protocols=('cluster',),phase='ablation',feature_columns=regex,save_model_modes=())) for name,regex in ABLATIONS.items())
    for name,kw in runs:
        out=ROOT/'results/study'/name
        if args.resume and intact(out):
            print(f'Reusing completed and verified {name}',flush=True);continue
        print(f'Starting {name}',flush=True)
        run_evaluation(ROOT/'results/study/dataset',ROOT/'data/features',out,n_jobs=args.jobs,**kw)

if __name__=='__main__':main()
