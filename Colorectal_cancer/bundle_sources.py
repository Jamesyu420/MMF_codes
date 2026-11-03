"""Create explicit source hashes for newly exported selected-version results."""
import argparse,json,shutil
from pathlib import Path
from Colorectal_cancer.io_utils import sha,write

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ['hd','xenium','context','marker-scores','output']:
        p.add_argument('--'+flag,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    cfg=json.loads(Path(__file__).with_name('figure7_config.json').read_text())
    for kind,source in [('HD',a.hd),('Xenium',a.xenium)]:
        out=a.output/'sources'/kind;out.mkdir(parents=True)
        for name in ['plot_source.npz','fit_run_manifest.json',
                     'plot_export_receipt.json' if kind=='HD' else 'summary.json']:
            shutil.copy2(source/name,out/name)
    shutil.copy2(a.context,a.output/'sources/Xenium/all_cell_context.npz')
    shutil.copy2(a.marker_scores,a.output/'sources/marker_scores.csv.gz')
    shutil.copy2(Path(__file__).with_name('marker_protocol.json'),a.output/'sources/marker_protocol.json')
    cfg['expected_sha256']={p:sha(a.output/p) for p in cfg['expected_sha256']}
    write(a.output/'config.json',cfg)
    shutil.copy2(Path(__file__).with_name('figure_contract.json'),a.output/'figure_contract.json')

if __name__=='__main__':main()
