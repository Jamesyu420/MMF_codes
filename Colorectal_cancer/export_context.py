"""Export all original Xenium IDs and coordinates into the exact fit coordinate frame."""
import argparse,json
from pathlib import Path
import h5py
import numpy as np

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared',type=Path,required=True)
    p.add_argument('--raw',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    with np.load(a.prepared,allow_pickle=False) as z:
        frame={k:z[k] for k in ['coords','original_coords','spot_names','original_indices']}
    with h5py.File(a.raw,'r') as f:
        ids=f['obs'][f['obs'].attrs['_index']].asstr()[:].astype(str)
        xy=f['obsm/spatial'][:]
    ix=frame['original_indices']
    np.testing.assert_array_equal(ids[ix],frame['spot_names'])
    np.testing.assert_allclose(xy[ix],frame['original_coords'],rtol=0,atol=1e-5)
    design=np.column_stack([frame['original_coords'],np.ones(len(ix))])
    affine=np.linalg.lstsq(design,frame['coords'],rcond=None)[0]
    coords=np.column_stack([xy,np.ones(len(ids))])@affine
    assert float(np.abs(coords[ix]-frame['coords']).max())<1e-6
    coords[ix]=frame['coords']
    a.output.parent.mkdir(parents=True,exist_ok=True)
    if a.output.exists():raise FileExistsError(a.output)
    np.savez_compressed(a.output,spot_names=ids,coords=coords,original_coords=xy,
        original_indices=np.arange(len(ids)),prepared_original_indices=ix)

if __name__=='__main__':main()
