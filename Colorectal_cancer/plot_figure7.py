"""Render a verified epithelial HD/Xenium pair in the approved three-row layout.

Usage: python plot_main_figure.py /path/to/config.json
Only output/QA files are written; no fitting or source modification occurs.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/mmf_mpl_cache')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.ticker import FixedLocator, FormatStrFormatter


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('figure7_config.json'))
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    root = args.data.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True,exist_ok=True)
    cfg = json.loads(args.config.read_text())
    contract = json.loads((args.config.resolve().parent / 'figure_contract.json').read_text())
    for rel, expected in cfg['expected_sha256'].items():
        assert sha(root / rel) == expected, rel
    data = {}; epi = {}; receipts = {}; manifests = {}
    for group in ('HD', 'Xenium'):
        source = root / cfg[f'{group}_source']
        d = load(source)
        receipt = json.loads((root / cfg[f'{group}_receipt']).read_text())
        manifest = json.loads((root / cfg[f'{group}_manifest']).read_text())
        assert manifest['status'] == 'completed', group
        assert sha(source) == receipt['plot_source_sha256'], group
        assert all(np.isfinite(v).all() for v in d.values() if v.dtype.kind in 'fciub')
        assert len(set(d['spot_names'])) == len(d['spot_names'])
        np.testing.assert_allclose(d['pi'].sum(axis=1), 1, atol=1e-6)
        c = list(d['cell_types']).index('Epithelial')
        rows = np.flatnonzero(d['program_cell_type'] == c)
        rows = rows[np.argsort(d['program_index'][rows])]
        assert len(set(d['program_index'][rows])) == len(rows)
        assert manifest['configuration']['lengthscale'] == cfg[f'{group}_lengthscale']
        epi[group] = {'pi': d['pi'][:, c], 'h': d['h_mean'][rows],
                      'program_ids': d['program_index'][rows] + 1, 'rows': rows}
        data[group] = d; receipts[group] = receipt; manifests[group] = manifest
    hd, xe = data['HD'], data['Xenium']
    assert len(xe['cell_types']) == 1 and list(xe['cell_types']) == ['Epithelial']
    np.testing.assert_array_equal(xe['pi'], np.ones_like(xe['pi']))
    assert receipts['Xenium']['pi_max_abs_drift'] == 0
    assert receipts['Xenium']['no_pi_optimizer']
    assert receipts['HD']['pi_max_abs_drift'] == 0
    assert receipts['HD']['shared_effects_max_abs_drift'] == 0
    context = load(root / cfg['Xenium_context'])
    full_ids, full_xy = context['spot_names'], context['coords']
    assert len(set(full_ids)) == len(full_ids) and np.isfinite(full_xy).all()
    context_rows = pd.Index(full_ids).get_indexer(xe['spot_names'])
    assert (context_rows >= 0).all() and len(set(context_rows)) == len(xe['spot_names'])
    np.testing.assert_array_equal(context['original_indices'][context_rows], xe['original_indices'])
    np.testing.assert_array_equal(full_xy[context_rows], xe['coords'])
    is_fit = np.zeros(len(full_ids), dtype=bool)
    is_fit[context_rows] = True
    display_pi = np.zeros(len(full_ids), dtype=np.float32)
    display_pi[context_rows] = epi['Xenium']['pi']
    marker = pd.read_csv(root / cfg['marker_source'])
    assert marker.spot.is_unique
    joined = marker.set_index('spot').loc[hd['spot_names']]
    assert np.array_equal(joined.index.to_numpy(), hd['spot_names'])
    hxy = joined[['x', 'y']].to_numpy()
    design = np.column_stack([hd['coords'], np.ones(len(hxy))])
    affine = np.linalg.lstsq(design, hxy, rcond=None)[0]
    coord_error = float(np.abs(design @ affine - hxy).max())
    assert coord_error < 1e-6
    assert np.isfinite(joined[['x', 'y', 'secretory_score', 'proliferation_score']].to_numpy()).all()
    source_data = dict(HD_coords=hxy, HD_fit_coords=hd['coords'], HD_pi=epi['HD']['pi'],
                       HD_h_mean=epi['HD']['h'], HD_program_ids=epi['HD']['program_ids'],
                       HD_spot_names=hd['spot_names'], Xenium_coords=xe['coords'],
                       Xenium_pi=epi['Xenium']['pi'], Xenium_h_mean=epi['Xenium']['h'],
                       Xenium_program_ids=epi['Xenium']['program_ids'], Xenium_spot_names=xe['spot_names'],
                       secretory_score=joined.secretory_score.to_numpy(),
                       proliferation_score=joined.proliferation_score.to_numpy(),
                       library_all_assayed_genes=joined.library_all_assayed_genes.to_numpy())
    source_data.update(Xenium_all_coords=full_xy, Xenium_all_spot_names=full_ids,
                       Xenium_all_is_fit=is_fit, Xenium_fit_rows_in_all=context_rows,
                       Xenium_all_display_pi=display_pi)
    np.savez_compressed(output / 'figure_source.npz', **source_data)
    values = [item['h'][:, item['pi'] > .05] for item in epi.values() if item['h'].size]
    limit = max((float(np.abs(v).max()) for v in values if v.size), default=1.)
    normh = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.titlesize': 8,
                         'xtick.labelsize': 7, 'ytick.labelsize': 7, 'pdf.fonttype': 42,
                         'svg.fonttype': 'none', 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.linewidth': .6})
    fig = plt.figure(figsize=(7.2, 6.6))
    outer = fig.add_gridspec(3, 1, left=.075, right=.98, bottom=.075, top=.89,
                            height_ratios=[1, 1, 1.18], hspace=.20)
    panelmap = []

    def spatial(spec, xy, val, norm, cmap, title='', pi=None):
        ax = fig.add_subplot(spec)
        ax.set_aspect('equal'); ax.set_anchor('N'); ax.set_axis_off()
        keep = np.ones(len(xy), dtype=bool) if pi is None else pi > .05
        ax.scatter(xy[:, 0], xy[:, 1], s=.55, c='#DDDDDD', lw=0, rasterized=True)
        artist = ax.scatter(xy[keep, 0], xy[keep, 1], s=.55, c=val[keep],
                            norm=norm, cmap=cmap, lw=0, rasterized=True)
        np.testing.assert_array_equal(np.asarray(artist.get_array()), val[keep])
        assert len(artist.get_offsets()) == int(keep.sum())
        ax.set_title(title, pad=3)
        return ax, keep

    for i, (group, xy, label) in enumerate([('HD', hxy, 'Visium HD'),
                                          ('Xenium', xe['coords'], 'Xenium')]):
        row = outer[i].subgridspec(1, 2, width_ratios=[1, 3], wspace=.075)
        pi, h, ids = epi[group]['pi'], epi[group]['h'], epi[group]['program_ids']
        if group == 'Xenium':
            ax = fig.add_subplot(row[0])
            ax.set_aspect('equal'); ax.set_anchor('N'); ax.set_axis_off()
            background = ax.scatter(full_xy[:, 0], full_xy[:, 1], s=.55, c=display_pi,
                                    norm=Normalize(0, 1), cmap='viridis', lw=0, rasterized=True)
            foreground = ax.scatter(xy[:, 0], xy[:, 1], s=.55, c=pi,
                                    norm=Normalize(0, 1), cmap='viridis', lw=0, rasterized=True)
            assert len(background.get_offsets()) == len(full_ids)
            np.testing.assert_array_equal(np.asarray(background.get_array()), display_pi)
            assert len(foreground.get_offsets()) == len(pi)
            np.testing.assert_array_equal(np.asarray(foreground.get_array()), pi)
        else:
            ax, keep = spatial(row[0], xy, pi, Normalize(0, 1), 'viridis')
        ax.text(-.14, .5, label, transform=ax.transAxes, rotation=90, ha='right', va='center', fontsize=9)
        panelmap.append({'row': 'ab'[i], 'modality': group, 'kind': 'fitted_pi', 'N': len(pi),
                         'min': float(pi.min()), 'max': float(pi.max())})
        if group == 'Xenium':
            panelmap[-1].update(all_cells_displayed=len(full_ids), fit_cells_yellow=len(pi),
                               nonfit_cells_purple_display_zero=int((~is_fit).sum()),
                               shared_proportion_colormap='viridis', shared_proportion_limits=[0, 1],
                               background_pi_estimated=False, no_background_subsampling=True)
        if len(ids):
            sub = row[1].subgridspec(1, len(ids), wspace=.07)
            for j, program in enumerate(ids):
                ax, keep = spatial(sub[j], xy, h[j], normh, 'RdBu_r', f'P{program}', pi=pi)
                assert not (np.abs(h[j, keep]) > limit).any()
                panelmap.append({'row': 'ab'[i], 'modality': group, 'kind': 'h_mean',
                                 'program_id': int(program), 'N': len(pi), 'visible_N': int(keep.sum()),
                                 'source_row': int(epi[group]['rows'][j]), 'transform': 'none',
                                 'clipped_count': 0, 'limits': [-limit, limit]})
        else:
            ax = fig.add_subplot(row[1]); ax.set_axis_off()
    for x, width, norm, cmap, ticks, title in [
        (.125, .15, Normalize(0, 1), 'viridis', [0, .5, 1], 'Cell type\nproportion'),
        (.55, .31, normh, 'RdBu_r', [-limit, 0, limit], 'Program activity'),
    ]:
        cax = fig.add_axes([x, .935, width, .011])
        cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='horizontal', ticks=ticks)
        cb.ax.tick_params(labelsize=7, length=2)
        cb.ax.xaxis.set_major_locator(FixedLocator(ticks))
        cb.ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        fig.text(x + width / 2, .991 if '\n' in title else .971, title, ha='center',
                 va='top' if '\n' in title else 'baseline', fontsize=8)
    bottom = outer[2].subgridspec(1, 2, wspace=.15)
    marker_display = []
    for j, (key, title) in enumerate([('secretory_score', 'Secretory gene score'),
                                     ('proliferation_score', 'Proliferation gene score')]):
        oldvalues = marker[key].to_numpy()
        cap = float(np.quantile(oldvalues[oldvalues > 0], .995))
        score = joined[key].to_numpy()
        ax, keep = spatial(bottom[j], hxy, score, Normalize(0, cap), 'magma', title)
        cax = ax.inset_axes([.06, -.075, .88, .025])
        cb = fig.colorbar(ScalarMappable(norm=Normalize(0, cap), cmap='magma'), cax=cax,
                          orientation='horizontal', ticks=[0, cap])
        cb.ax.tick_params(labelsize=7, length=2)
        marker_display.append({'score': key, 'limits': [0, cap], 'above_display_cap': int((score > cap).sum()),
                               'N': len(score), 'unchanged_template_scale': True})
        panelmap.append({'row': 'c', **marker_display[-1]})
    for i, letter in enumerate('abc'):
        fig.text(.035, outer[i].get_position(fig).y1 + .01, letter, fontweight='bold', fontsize=10, va='bottom')
    fig.canvas.draw(); renderer = fig.canvas.get_renderer()
    bad = []
    for text in fig.findobj(matplotlib.text.Text):
        if text.get_visible() and text.get_text():
            box = text.get_window_extent(renderer)
            if box.width and box.height and (box.x0 < -1 or box.y0 < -1 or
                                             box.x1 > fig.bbox.width + 1 or box.y1 > fig.bbox.height + 1):
                bad.append(text.get_text())
    assert not bad, bad
    exports = {}
    for ext in ('pdf', 'svg', 'png'):
        target = output / f"{cfg['name']}.{ext}"
        fig.savefig(target, dpi=600 if ext != 'png' else 300, facecolor='white')
        exports[ext] = {'file': target.name, 'sha256': sha(target)}
    plt.close(fig)
    nk_hd, nk_xe = len(epi['HD']['program_ids']), len(epi['Xenium']['program_ids'])
    caption = (
        f"Epithelial proportions, native program activities and gene-score context. "
        f"(a) HD 32 × 32 µm bins, {len(hd['genes'])} fitting genes, {len(hd['coords']):,} bins, "
        f"{nk_hd} epithelial programs; panel-specific RCTD π/α/γ fixed, lengthscale {cfg['HD_lengthscale']}. "
        f"(b) Xenium, nominal {cfg['nominal_genes']} genes and {len(xe['genes'])} actual fitting genes, "
        f"{len(xe['coords']):,} selected cells, {nk_xe} epithelial programs; single Epithelial type, "
        f"π fixed to 1, α/γ estimated, lengthscale {cfg['Xenium_lengthscale']}. "
        f"The left Xenium panel displays all {len(full_ids):,} original cells using the same viridis 0–1 color scale as HD "
        f"and overlays the {len(xe['coords']):,} actual fitted cells in yellow at π=1. "
        "Cells not used in this fit are assigned display value 0 (purple) by the requested display convention; "
        "this display value is not an inferred MMF or RCTD proportion for those cells. "
        "The selected-cell π=1 values represent the fitted constraint, not the original RCTD proportions. "
        "Each modality displays every retained epithelial program in ascending native program-ID order; IDs do not imply cross-modality correspondence. "
        f"All activities use unchanged native signs and units and one full symmetric scale ±{limit:.8f}; no activity centering, clipping or subsampling. "
        "HD activity locations with π ≤ 0.05 are grey; all Xenium cells are visible. "
        "(c) Unchanged prespecified secretory and proliferation scores, aligned to the current HD bins by exact IDs, "
        "computed as mean log1p(CP10000) with the library denominator over all 18,085 assayed genes. "
        "The original template's positive-score 99.5th percentile color limits are retained: "
        + '; '.join(f"{m['score']}: 0–{m['limits'][1]:.8f} ({m['above_display_cap']} bins above the display cap)" for m in marker_display)
        + ". These gene-score display limits do not truncate program activity. "
        f"This epithelial main figure shows {nk_hd} of the HD fit's {len(hd['h_mean'])} total programs; the other cell types remain in the full source data and all-program companion figures. "
        "Native coordinate axes follow the supplied template's increasing-y display convention. "
        "No spatial registration or pointwise H correlation is performed. Different model constraints and lengthscales preclude single-factor causal attribution; this is not independent biological replication."
    )
    (output / f"{cfg['name']}.caption.txt").write_text(caption + '\n')
    audit = {'status': 'computed_pending_rendered_QA', 'template_sha256': contract['template_sha256'],
             'source_hashes_verified': True, 'source_sha256': cfg['expected_sha256'],
             'all_numeric_arrays_finite': True, 'input_completion_verified': True,
             'HD_total_programs': len(hd['h_mean']), 'HD_epithelial_programs': nk_hd,
             'Xenium_epithelial_programs': nk_xe, 'all_epithelial_programs_displayed': True,
             'HD_program_order': epi['HD']['program_ids'].tolist(),
             'Xenium_program_order': epi['Xenium']['program_ids'].tolist(),
             'native_activity_limits': [-limit, limit], 'native_h_transform': 'none',
             'activity_clipped_count': 0, 'subsampled': False, 'grey_mask': 'pi <= 0.05',
             'HD_coordinate_affine_to_template': affine.tolist(), 'HD_coordinate_max_error': coord_error,
             'marker_join_exact_bin_ids': True, 'marker_display': marker_display,
             'HD_N': len(hd['coords']), 'Xenium_N': len(xe['coords']),
             'HD_G': len(hd['genes']), 'Xenium_actual_G': len(xe['genes']),
             'Xenium_pi_is_exactly_one': True, 'program_ID_order_not_matching_order': True,
             'Xenium_all_cells_displayed': len(full_ids), 'Xenium_fit_overlay_cells': int(is_fit.sum()),
             'Xenium_nonfit_background_cells': int((~is_fit).sum()),
             'Xenium_all_cell_coordinates_and_ID_mapping_verified': True,
             'Xenium_background_model_pi_imputed': False, 'Xenium_background_display_value': 0,
             'shared_proportion_colormap': 'viridis', 'shared_proportion_limits': [0, 1],
             'figure_inches': [7.2, 6.6], 'only_template_labels': True,
             'no_text_outside_canvas': True, 'panel_map': panelmap, 'output_figures': exports,
             'source_figure_npz_sha256': sha(output / 'figure_source.npz'), 'producer_sha256': sha(__file__),
             'spatial_registration': False, 'H_correlation': False, 'fit_rerun': False}
    (output / 'figure_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'figure': cfg['name'], 'HD_epithelial_K': nk_hd, 'Xenium_epithelial_K': nk_xe,
                      'native_activity_limit': limit, 'marker_display': marker_display}, indent=2))


if __name__ == '__main__':
    main()
