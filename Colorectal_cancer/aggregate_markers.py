"""Aggregate prespecified out-of-panel HD markers using the original fit membership."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--raw',type=Path,required=True)
ap.add_argument('--membership',type=Path,required=True)
ap.add_argument('--plot',type=Path,required=True)
ap.add_argument('--protocol',type=Path,required=True)
ap.add_argument('--output',type=Path,required=True)
a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
raw_path=a.raw/'hd_full18085_original_marker_counts.npz'
x=dict(np.load(raw_path,allow_pickle=False))
obs=pd.read_csv(a.raw/'hd_full18085_original_obs.csv.gz',index_col=0)
m=dict(np.load(a.membership,allow_pickle=False));p=dict(np.load(a.plot,allow_pickle=False))
assert np.array_equal(x['barcodes'],obs.index.to_numpy(dtype=str))
assert np.array_equal(m['original_indices'],np.flatnonzero(obs.high_quality.to_numpy()==1))
n_aggregate=len(m['aggregate_grid_coordinates'])
allcounts=np.zeros((n_aggregate,len(x['genes'])),dtype=np.int64)
alllibrary=np.zeros(n_aggregate,dtype=np.int64)
np.add.at(allcounts,m['aggregate_index'],x['counts'][m['original_indices']])
np.add.at(alllibrary,m['aggregate_index'],x['library_size_all_genes'][m['original_indices']])
assert np.array_equal(allcounts.sum(0),x['counts'][m['original_indices']].sum(0))
assert alllibrary.sum()==x['library_size_all_genes'][m['original_indices']].sum()
contributors=np.bincount(m['aggregate_index'])
centroid=np.zeros((n_aggregate,2))
np.add.at(centroid,m['aggregate_index'],x['coords'][m['original_indices']])
centroid/=contributors[:,None]
keep=p['original_indices']
assert np.allclose(centroid[keep],p['original_coords'],atol=1e-10,rtol=1e-10)
names=np.array([f'aggregate_{r}_{c}' for r,c in m['aggregate_grid_coordinates']])
assert np.array_equal(names[keep],p['spot_names']) and (alllibrary[keep]>0).all()
protocol=json.loads(a.protocol.read_text());gene_index={g:i for i,g in enumerate(x['genes'])}
cp10k=allcounts[keep]/alllibrary[keep,None]*10000
scores={};used={}
for group in ['secretory_goblet_associated','proliferation','epithelial_context']:
    used[group]=[g for g in protocol[group] if g in gene_index]
    scores[group]=np.log1p(cp10k[:,[gene_index[g] for g in used[group]]]).mean(1)
np.savez_compressed(a.output/'hd32_markers.npz',counts=allcounts[keep],
    library_size_all_genes=alllibrary[keep],genes=x['genes'],coords=p['coords'],
    original_coords=p['original_coords'],spot_names=p['spot_names'],**scores)
# Missing original cell-type annotations remain an explicit category.
annotation=obs.annotation.fillna('Unannotated').astype(str).to_numpy()
classes=sorted(set(annotation));fractions=np.zeros((n_aggregate,len(classes)))
for j,c in enumerate(classes):
    np.add.at(fractions[:,j],m['aggregate_index'],(annotation[m['original_indices']]==c).astype(int))
fractions/=contributors[:,None]
assert np.allclose(fractions.sum(1),1)
np.savez_compressed(a.output/'hd32_original_annotation_fractions.npz',fractions=fractions[keep],
    cell_types=np.array(classes),coords=p['coords'],spot_names=p['spot_names'])
audit=dict(passed=True,original_bins=len(obs),published_high_quality_bins=len(m['original_indices']),
    aggregated_bins=n_aggregate,final_retained_bins=len(keep),all_marker_integer_counts_conserved=True,
    full_gene_library_totals_conserved=True,source_bin_QC_membership_matches_original_obs_exactly=True,
    aggregate_barcodes_match_prepared_fit_order_exactly=True,
    max_centroid_coordinate_error=float(abs(centroid[keep]-p['original_coords']).max()),
    orientation='Exactly prepared HD32 coords[:,0],coords[:,1]; no inversion, rotation or registration assumption.',
    marker_groups_used=used,original_annotation_categories=classes,
    protocol_sha256=hashlib.sha256(a.protocol.read_bytes()).hexdigest(),
    raw_export_sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest(),
    membership_sha256=hashlib.sha256(a.membership.read_bytes()).hexdigest(),
    plot_input_sha256=hashlib.sha256(a.plot.read_bytes()).hexdigest(),new_fit_read=False)
(a.output/'preparation_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps(audit,indent=2))
