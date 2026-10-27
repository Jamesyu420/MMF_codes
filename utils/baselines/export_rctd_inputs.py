"""Export training counts and actual reference cells, never evaluation truth."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.io import mmwrite

p = argparse.ArgumentParser()
p.add_argument('--fit', type=Path, required=True)
p.add_argument('--reference', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
fit = np.load(a.fit, allow_pickle=False)
ref = np.load(a.reference, allow_pickle=False)
y, x, labels = fit['Y'], ref['X'], ref['cell_type']
assert y.shape[1] == x.shape[1] == len(ref['genes'])
assert np.all(y >= 0) and np.all(y == np.rint(y))
assert np.all(x >= 0) and np.all(x == np.rint(x))
mmwrite(a.output/'spatial_counts.mtx', sparse.coo_matrix(y.T))
mmwrite(a.output/'reference_counts.mtx', sparse.coo_matrix(x.T))
np.savetxt(a.output/'coords.tsv', fit['coords'], delimiter='\t', fmt='%.17g')
(a.output/'genes.txt').write_text('\n'.join(map(str, ref['genes']))+'\n')
(a.output/'reference_labels.txt').write_text('\n'.join(f'T{int(c)+1}' for c in labels)+'\n')
receipt = {'fit_input':str(a.fit.resolve()), 'reference_input':str(a.reference.resolve()),
    'fit_input_sha256':hashlib.sha256(a.fit.read_bytes()).hexdigest(),
    'reference_input_sha256':hashlib.sha256(a.reference.read_bytes()).hexdigest(),
    'n_spots':len(y),'n_reference_cells':len(x),'genes':y.shape[1],
    'reference_cell_counts':dict(zip(*[z.tolist() for z in np.unique(labels,return_counts=True)])),
    'evaluation_truth_read':False,'reference_is_actual_independent_cells':True}
(a.output/'input_receipt.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt))
