"""Print selected catalog commands by default; --execute runs serial fresh attempts.

GPU admission and scope gates belong to the parent execution controller. This
tool never self-detaches, starts no daemon, and never reuses a partial output.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--catalog',type=Path,required=True)
    p.add_argument('--method',action='append')
    p.add_argument('--experiment',action='append')
    p.add_argument('--seed',type=int,action='append')
    p.add_argument('--execute',action='store_true')
    p.add_argument('--receipt',type=Path)
    a=p.parse_args()
    rows=json.loads(a.catalog.read_text())['commands']
    rows=[r for r in rows if (not a.method or r['method'] in a.method) and
          (not a.experiment or r['experiment'] in a.experiment) and
          (not a.seed or r['seed'] in a.seed)]
    if not a.execute:
        print(json.dumps(dict(executed=False,count=len(rows),commands=rows),indent=2));return
    if a.receipt is None:
        raise ValueError('--execute requires --receipt in the new execution directory')
    if a.receipt.exists():raise FileExistsError('Existing queue receipt retained; use a new receipt')
    a.receipt.parent.mkdir(parents=True,exist_ok=True)
    suite=dict(phase='running',started_unix=time.time(),pid=os.getpid(),total=len(rows),completed=[])
    def save():
        temp=a.receipt.with_suffix('.tmp');temp.write_text(json.dumps(suite,indent=2)+'\n');temp.replace(a.receipt)
    save()
    for row in rows:
        out=Path(row['output']);out.mkdir(parents=True,exist_ok=True)
        if any((out/n).exists() for n in ['queue_outer.log','experiment_status.json','execution_receipt.json','final.npz','factors.npz']):
            suite.update(phase='stopped_existing_attempt',active=row['run_id']);save()
            raise FileExistsError(out)
        began=time.time()
        with (out/'queue_outer.log').open('x') as log:
            proc=subprocess.Popen(row['command'],stdout=log,stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,env={**os.environ,**row['environment']})
            suite['active']=dict(run_id=row['run_id'],pid=proc.pid,started_unix=began,output=str(out));save()
            code=proc.wait()
        receipt_file=out/('experiment_status.json' if row['method'].startswith('MMF') else 'execution_receipt.json')
        child=json.loads(receipt_file.read_text()) if receipt_file.exists() else {}
        complete=code==0 and child.get('phase')==row.get('expected_terminal_phase','finished')
        suite['completed'].append(dict(run_id=row['run_id'],returncode=code,complete=complete,
            elapsed_seconds=time.time()-began,child_receipt=str(receipt_file)))
        suite.pop('active',None);save()
        if not complete:
            suite['phase']='stopped_incomplete_attempt';save();raise RuntimeError(row['run_id'])
    suite.update(phase='finished',elapsed_seconds=time.time()-suite['started_unix']);save()


if __name__=='__main__':main()
