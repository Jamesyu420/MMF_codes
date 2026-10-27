"""Generate separate calibration / positive-epsilon inputs without editing originals.

Calibration regenerates NEW seeds 2026102601..03 from the frozen original
generator. Extension reads archived main-50 primitives only at generation time.
All fitting inputs, reference cells, heldout counts and truth stay in separate
files. One epsilon draw is shared by all molecule splits of the same entry.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode=True


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def arrays(path):
    with np.load(path,allow_pickle=False) as z:return dict(z)


def new_calibration(module, seed):
    fit,truth=module.simulate(seed,'gradient',side=20,genes=250)
    cells=truth.pop('reference_cells')
    base=fit['library_size'].copy()
    previous=1
    for depth in [2,4]:
        for split,key in enumerate(['Y','Y_validation','Y_test']):
            rng=np.random.default_rng(np.random.SeedSequence([seed,20261024,0,depth,split]))
            fit[key]+=rng.poisson((depth-previous)*[.6,.2,.2][split]*truth['rate_true'])
        fit['library_size']=depth*base
        previous=depth
    truth['rate_true']*=4
    held={k:fit.pop(k) for k in ['Y_validation','Y_test']}
    reference=dict(X=cells.reshape(-1,250),cell_type=np.repeat(np.arange(5),200),
                   genes=np.asarray([f'g{g}' for g in range(250)]))
    return fit,held,truth,reference


def coupled_counts(old, old_mean, new_mean, rng):
    reduced=new_mean<old_mean
    out=np.empty_like(old)
    out[reduced]=rng.binomial(old[reduced],new_mean[reduced]/old_mean[reduced])
    out[~reduced]=old[~reduced]+rng.poisson(new_mean[~reduced]-old_mean[~reduced])
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind',choices=['calibration','extension'],required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source-data',type=Path)
    p.add_argument('--original-generator',type=Path)
    p.add_argument('--seeds',help='Comma-separated full integer seeds')
    p.add_argument('--sigmas',help='Comma-separated SDs; no -sigma^2/2 mean correction')
    a=p.parse_args()
    seeds=list(map(int,a.seeds.split(','))) if a.seeds else (
        [2026102601,2026102602,2026102603] if a.kind=='calibration' else list(range(202610201,202610251)))
    sigmas=list(map(float,a.sigmas.split(','))) if a.sigmas else ([0.,.3] if a.kind=='calibration' else [.1,.3,.5])
    if a.kind=='calibration':
        if any(seed in range(202610201,202610251) for seed in seeds):
            raise ValueError('Calibration must use seeds separate from the primary 50')
        if a.original_generator is None:raise ValueError('--original-generator required')
        spec=importlib.util.spec_from_file_location('frozen_mmf_generator',a.original_generator)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    elif a.source_data is None:raise ValueError('--source-data required')
    a.output.mkdir(parents=True,exist_ok=True)
    records=[]
    for seed in seeds:
        if a.kind=='calibration':
            fit,held,truth,ref=new_calibration(module,seed)
            source=dict(original_generator=str(a.original_generator),original_generator_sha256=sha(a.original_generator))
        else:
            inp=a.source_data/f'gradient_{seed}_depth4x'
            fit,held,truth,ref=[arrays(inp/n) for n in ['fit_input.npz','heldout.npz','evaluation_only.npz','reference_input.npz']]
            source={n:sha(inp/n) for n in ['fit_input.npz','heldout.npz','evaluation_only.npz','reference_input.npz']}
        structural=truth['rate_true'].copy()
        for sigma in sigmas:
            if sigma<0:raise ValueError('Negative epsilon SD')
            # Use the same independent standard-normal field across noise levels.
            stream=[seed,20261026,101]
            epsilon=sigma*np.random.default_rng(np.random.SeedSequence(stream)).normal(size=structural.shape)
            rate=structural*np.exp(epsilon)
            fit_new={k:v.copy() for k,v in fit.items()}
            held_new={k:v.copy() for k,v in held.items()}
            for split,(key,fraction) in enumerate(zip(['Y','Y_validation','Y_test'],[.6,.2,.2])):
                old=fit[key] if key=='Y' else held[key]
                if sigma==0:counts=old.copy()
                else:
                    rng=np.random.default_rng(np.random.SeedSequence([seed,20261026,201,split]))
                    counts=coupled_counts(old,fraction*structural,fraction*rate,rng)
                (fit_new if key=='Y' else held_new)[key]=counts
            truth_new={k:v.copy() for k,v in truth.items()}
            truth_new.update(rate_true=rate,structural_rate_true=structural,
                             epsilon_true=epsilon,epsilon_sigma_true=np.asarray(sigma))
            case=f'gradient_{seed}_depth4x'
            out=a.output/f'sigma_{sigma:g}'/case
            out.mkdir(parents=True,exist_ok=False)
            for name,payload in [('fit_input.npz',fit_new),('heldout.npz',held_new),
                                 ('evaluation_only.npz',truth_new),('reference_input.npz',ref)]:
                np.savez_compressed(out/name,**payload)
            row=dict(kind=a.kind,seed=seed,sigma_epsilon_DGP=sigma,variance_epsilon_DGP=sigma*sigma,
                epsilon_mean_correction=False,epsilon_rng_stream=stream,
                molecule_split_rng_stream='[seed,20261026,201,split]',
                epsilon_shared_across_molecule_splits=True,source=source,
                inputs_for_fit=['fit_input.npz'],reference_input='reference_input.npz',
                evaluation_only=['heldout.npz','evaluation_only.npz'],
                N=400,G=250,C=5,truth_K=[1,2,0,1,2],global_capacity=11,
                train_validation_test_fractions=[.6,.2,.2],
                latent_structure_and_reference_preserved=True,
                count_construction='Exact Poisson-marginal coupling by thinning and independent increments; noise SD zero retains original counts',
                source_code_sha256=sha(__file__),
                output_hashes={p.name:sha(p) for p in out.glob('*.npz')})
            (out/'input_manifest.json').write_text(json.dumps(row,indent=2)+'\n')
            records.append(dict(input_root=str(out),**row))
    (a.output/'generation_manifest.json').write_text(json.dumps(dict(kind=a.kind,cases=records),indent=2)+'\n')
    print(json.dumps(dict(kind=a.kind,cases=len(records),output=str(a.output))))


if __name__=='__main__':main()
