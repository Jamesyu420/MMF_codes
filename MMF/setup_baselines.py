"""Fetch the pinned external baseline repositories into a separate directory."""
import argparse
from pathlib import Path
import subprocess

SOURCES = {
    'scTM': ('https://github.com/JinmiaoChenLab/scTM.git', '0099ab268c712c91f398a2862db59b1dc793f69b'),
    'SpiceMix': ('https://github.com/ma-compbio/SpiceMix.git', 'aea69f8081f33e8ca64b734c680804db907ed4eb'),
}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    for name,(url,commit) in SOURCES.items():
        path=a.output/name
        if path.exists():
            raise FileExistsError(f'{path} already exists; existing external sources are never replaced')
        subprocess.run(['git','clone','--no-checkout',url,str(path)],check=True)
        subprocess.run(['git','-C',str(path),'checkout','--detach',commit],check=True)
        actual=subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
        if actual!=commit:raise RuntimeError(f'Unexpected {name} commit: {actual}')


if __name__=='__main__':main()
