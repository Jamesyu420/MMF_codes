"""Export plotting arrays and full ELBO chronology from completed Visium fits."""
import argparse, csv, json, hashlib
from pathlib import Path
import numpy as np

def iteration_records(side):
    d=RAW['sections'][f'Visium_{side+1}'];phases={p['path']:p['record'] for p in d['phases']}
    sel=d['selection.json']['record'];run=d['run_manifest.json']['record'];scores=d['scores']
    assert run['status']=='completed' and run['source_unchanged']
    used=set();cursor=0;call=0;points=[];segments=[]
    def phase(path):
        assert path not in used;used.add(path);return phases[path]
    def strict(name):
        nonlocal cursor
        path=f'{name}/phase.json';p=phase(path);assert p['strict_exit_validation']['passed']
        begin=cursor
        for row in p['history']:
            points.append({'section':side+1,'iteration':cursor+row['step'],'elbo':row['elbo_per_entry']*run['N']*run['G'],
                           'phase':name,'role':'retained','source':path,'local_step':row['step'],'score_call':'','seed':51023,'particles':16})
        cursor+=p['steps'];segments.append({'phase':name,'start':begin,'end':cursor})
    def take(role,name,expected):
        nonlocal call
        rec=scores[call]['record'];assert rec['call']==call and rec['ranks']==expected
        point={'section':side+1,'iteration':cursor,'elbo':rec['mean'],'phase':name,'role':role,'source':scores[call]['path'],
               'local_step':'','score_call':call,'seed':rec['seed'],'particles':rec['particles']}
        points.append(point);call+=1;return rec['mean']
    strict('k0')
    for rnd in sel['rounds']:
        k=rnd['round'];name=f'round{k}';begin=cursor;ranks=list(rnd['parent_ranks'])
        for trial in rnd['trials']:
            baseline=take('retained',name,ranks)
            path=f'round{k:02d}/type{trial["cell_type"]}/forward/phase.json';cursor+=phase(path)['steps']
            candidate=ranks.copy();candidate[trial['cell_type']]+=1
            after=take('retained' if trial['accepted'] else 'rejected candidate',name,candidate)
            assert abs(after-baseline-trial['forward_delta_elbo'])<1e-6
            if trial['accepted']:ranks=candidate
        p=phase(f'round{k:02d}/backfit/phase.json');cursor+=p['steps'];ranks=p['ranks'].copy()
        for j,deletion in enumerate(rnd['deletions']):
            assert ranks==deletion['before_ranks'];baseline=take('retained',name,ranks)
            path=f'round{k:02d}/cleanup/delete_{j:02d}_T{deletion["cell_type"]}P{deletion["program"]}/phase.json'
            cursor+=phase(path)['steps']
            after=take('retained' if deletion['removed'] else 'rejected candidate',name,deletion['after_ranks'])
            assert abs(after-baseline-deletion['delta_elbo'])<1e-6
            if deletion['removed']:ranks=deletion['after_ranks']
        assert ranks==rnd['retained_ranks'];segments.append({'phase':name,'start':begin,'end':cursor})
    strict('polish');last=take('retained','final',sel['final_ranks'])
    assert abs(last-sel['final_training_elbo'])<1e-8
    assert used==set(phases) and cursor==sum(p['steps'] for p in phases.values()) and call==len(scores)
    assert np.all(np.diff([r['iteration'] for r in points])>=0)
    return points,{'section':side+1,'total_iterations':cursor,'phases_consumed':len(used),'score_calls_consumed':call,'segments':segments}

def main():
    global RAW
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--results',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();(a.output/'source').mkdir(parents=True,exist_ok=False)
    RAW={'sections':{}}
    def record(path,root):
        return {'path':str(path.relative_to(root)), 'record':json.loads(path.read_text()),
                'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    for s in (1,2):
        name=f'Visium_{s}';root=a.results/name
        (a.output/f'{name}_run_manifest.json').write_text((root/'run_manifest.json').read_text())
        f=dict(np.load(root/'final.npz',allow_pickle=False));d=dict(np.load(a.inputs/f'{name}.npz',allow_pickle=False))
        np.savez_compressed(a.output/f'source/{name}_figure_source.npz',coords=d['coords'],genes=d['genes'],
            cell_types=d['cell_types'],visible=f['pi']>.05,
            **{k:f[k] for k in ['pi','h_mean','loadings','program_cell_type','program_index']})
        RAW['sections'][name]={'scores':[record(p,root) for p in sorted((root/'scores').glob('*.json'))],
            'phases':[record(p,root) for p in sorted(root.rglob('phase.json'))],
            **{n:record(root/n,root) for n in ['selection.json','run_manifest.json']}}
    (a.output/'full_run_records.json').write_text(json.dumps(RAW,indent=2)+'\n')
    datasets=[iteration_records(s) for s in (0,1)]
    with (a.output/'elbo_by_iteration.csv').open('w') as h:
        w=csv.DictWriter(h,fieldnames=list(datasets[0][0][0]));w.writeheader()
        for points,_ in datasets:w.writerows(points)

if __name__=='__main__':main()
