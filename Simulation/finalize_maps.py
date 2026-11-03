"""Reapply the submitted Fig 2/3 PDF label cleanup without moving any marks.

The submitted PDFs remove the three method-name asterisks and the descriptive
colorbar sentence from the original plot_note.py output. Their PDF page boxes,
map arrays, color scales, glyph origins and all numerical annotations stay fixed.
"""
import argparse
from pathlib import Path
import re

import fitz


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', type=Path, required=True)
    a = p.parse_args()
    label = 'Centered map / spatial RMS (display limits ±2.5)'
    for name in ['Fig2_pi_maps', 'Fig3_h_maps']:
        path = a.directory / f'{name}.pdf'
        doc = fitz.open(path)
        page = doc[0]
        contents = b'\n'.join(doc.xref_stream(i) for i in page.get_contents())
        if contents.count(b'\x00*') != 3:
            raise ValueError(f'{path}: expected three original method asterisks; retain the original file')
        contents = contents.replace(b'\x00*', b'')
        literal = label.encode('utf-16-be').replace(b'(', b'\\(').replace(b')', b'\\)')
        if contents.count(literal) != 1:
            raise ValueError(f'{path}: original descriptive colorbar label not found exactly once')
        contents = contents.replace(literal, b'')
        page.set_contents(page.get_contents()[0])
        doc.update_stream(page.get_contents()[0], contents)
        temporary = path.with_suffix('.tmp.pdf')
        doc.save(temporary, garbage=4, deflate=True)
        doc.close()
        temporary.replace(path)
        svg = a.directory / f'{name}.svg'
        if svg.exists():
            text = svg.read_text()
            for method in ['STAMP', 'SpiceMix', 'NMF']:
                text = text.replace(method + '*', method)
            text = re.sub(r'(<text[^>]*>)' + re.escape(label) + r'(</text>)', r'\1\2', text)
            svg.write_text(text)
        with fitz.open(path) as final:
            final[0].get_pixmap(matrix=fitz.Matrix(300/72, 300/72), alpha=False).save(a.directory / f'{name}.png')


if __name__ == '__main__':
    main()
