"""Export the original all-gene-normalized marker scores; no fitted values are used."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--markers',type=Path,required=True,help='Original full-18085-gene hd32_markers.npz')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    with np.load(a.markers,allow_pickle=False) as z:
        d={k:z[k] for k in ['spot_names','coords','secretory_goblet_associated','proliferation','library_size_all_genes']}
    assert len(d['spot_names'])==37678,'Retain the original template cohort when computing display caps'
    frame=pd.DataFrame(dict(spot=d['spot_names'],x=d['coords'][:,0],y=d['coords'][:,1],
        secretory_score=d['secretory_goblet_associated'],proliferation_score=d['proliferation'],
        library_all_assayed_genes=d['library_size_all_genes']))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    if a.output.exists():raise FileExistsError(a.output)
    frame.to_csv(a.output,index=False)

if __name__=='__main__':main()
