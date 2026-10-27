"""One fresh historical-protocol global or RCTD comparison; never opens truth.

The baseline-code directory contains unchanged fit_global.py,
export_rctd_inputs.py and run_rctd.R snapshots, with explicit hashes in receipt.
The RCTD R seed remains 240 for all datasets to preserve its original protocol.
"""
import argparse
import hashlib
import json
import os
import socket
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode=True


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--method',choices=['NMF','STAMP','SpiceMix','RCTD'],required=True)
    p.add_argument('--rank',type=int,required=True)
    p.add_argument('--baseline-code',type=Path,required=True)
    p.add_argument('--vendor',type=Path,required=True)
    p.add_argument('--reference',type=Path)
    p.add_argument('--rscript',type=Path)
    p.add_argument('--rlib',type=Path)
    p.add_argument('--device',default='cuda')
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    status_path=a.output/'execution_receipt.json'
    if status_path.exists() or (a.output/'factors.npz').exists():
        raise FileExistsError('Existing attempt retained; use a new output directory')
    names=['export_rctd_inputs.py','run_rctd.R'] if a.method=='RCTD' else ['fit_global.py']
    source={str(a.baseline_code/n):sha(a.baseline_code/n) for n in names}
    env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1',
         'MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1'}
    if a.device=='cpu':env['CUDA_VISIBLE_DEVICES']=''
    if a.method=='RCTD':
        if any(v is None for v in [a.reference,a.rscript,a.rlib]):
            raise ValueError('RCTD requires --reference --rscript --rlib')
        env['R_LIBS_USER']=str(a.rlib.resolve())
        commands=[[sys.executable,str(a.baseline_code/'export_rctd_inputs.py'),
            '--fit',str(a.input),'--reference',str(a.reference),'--output',str(a.output/'inputs')],
            [str(a.rscript),str(a.baseline_code/'run_rctd.R'),str(a.output/'inputs'),
             str(a.output),str(a.rlib.resolve())]]
    else:
        commands=[[sys.executable,str(a.baseline_code/'fit_global.py'),
            '--input',str(a.input),'--output',str(a.output),'--vendor',str(a.vendor),
            '--method',a.method,'--rank',str(a.rank),'--seed',str(a.seed),'--device',a.device]]
    began=time.time()
    row=dict(phase='running',seed=a.seed,method=a.method,pid=os.getpid(),host=socket.gethostname(),device=a.device,started_unix=began,
        input_sha256=sha(a.input),source_sha256=source,commands=commands,
        truth_or_heldout_read=False,reference_profiles_accessed=a.method=='RCTD',
        global_rank=None if a.method=='RCTD' else a.rank,
        RCTD_R_seed=240 if a.method=='RCTD' else None)
    write(status_path,row)
    try:
        with (a.output/'outer.log').open('x') as log:
            for cmd in commands:
                subprocess.run(cmd,check=True,stdout=log,stderr=subprocess.STDOUT,env=env,
                               timeout=max(1,7100-(time.time()-began)))
        import numpy as np
        if a.method=='RCTD':
            import pandas as pd
            frame=pd.read_csv(a.output/'proportions.csv',index_col=0)
            expected=[f'T{j+1}' for j in range(len(frame.columns))]
            if list(frame.columns)!=expected:
                raise ValueError(f'RCTD type order differs from frozen reference: {list(frame.columns)}')
            fractions=frame.to_numpy(dtype=float)
            assert np.isfinite(fractions).all() and (fractions>=0).all()
            np.testing.assert_allclose(fractions.sum(1),1,atol=1e-8)
            np.savez_compressed(a.output/'factors.npz',pi=fractions,cell_types=np.asarray(expected))
            assert (a.output/'fit.rds').exists()
        else:
            fit=json.loads((a.output/'fit.json').read_text())
            assert fit.get('status')=='completed'
            with np.load(a.output/'factors.npz',allow_pickle=False) as z:
                assert z['activity'].shape[1]==a.rank and z['loadings'].shape[0]==a.rank
                assert np.isfinite(z['activity']).all() and np.isfinite(z['loadings']).all()
        row.update(phase='finished',factors_sha256=sha(a.output/'factors.npz'),
                   source_unchanged=all(sha(path)==h for path,h in source.items()))
        assert row['source_unchanged']
    except Exception as error:
        row.update(phase='incomplete',error=repr(error))
        raise
    finally:
        row['elapsed_seconds']=time.time()-began;write(status_path,row)
    print(json.dumps(row),flush=True)


if __name__=='__main__':main()
