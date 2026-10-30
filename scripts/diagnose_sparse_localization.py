"""Exact pairwise AUROC decomposition, not tuning or causal attribution."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from selection_core import selection_mask, normalize


def wins(positive, negative):
    """Count correctly ordered pairs, ties worth one half, without pair matrix."""
    negative = np.sort(negative)
    left = np.searchsorted(negative, positive, side='left')
    right = np.searchsorted(negative, positive, side='right')
    return float((left.astype(np.float64) + right).sum() / 2)


def decompose(amap, mask, selected):
    p = [amap[mask & selected], amap[mask & ~selected]]
    n = [amap[~mask & selected], amap[~mask & ~selected]]
    denom = int(mask.sum()) * int((~mask).sum())
    return {a+b: {'weight': len(p[i])*len(n[j])/denom,
                  'contribution': wins(p[i], n[j])/denom}
            for i, a in enumerate(['S', 'U']) for j, b in enumerate(['S', 'U'])}


def main():
    assert wins(np.array([1, 2]), np.array([0, 1])) == 3.5
    assert wins(np.array([]), np.array([0])) == 0
    assert wins(np.array([1]), np.array([])) == 0
    # Synthetic selected/unselected groups, including tied scores.
    test = np.array([[0., 1.], [1., 2.]])
    mask = np.array([[False, True], [False, True]])
    sel = np.array([[True, True], [False, False]])
    terms = decompose(test, mask, sel)
    assert sum(v['weight'] for v in terms.values()) == 1
    assert sum(v['contribution'] for v in terms.values()) == .875
    data = ROOT / 'data/mvtec_grid_mirror'
    source = ROOT / 'results/sparse_refinement_20261030_105631'
    maps = ROOT / 'results/resolution_comparison_20261030_104915'
    selections = ROOT / 'results/selection_20261030_015741'
    out = ROOT / 'results/sparse_localization_diagnosis_v1'
    out.mkdir(exist_ok=False)
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    plan = {'scope': 'post-hoc grid development diagnosis; no changes to scores/selection',
            'methods': ['coarse', 'fusion_global', 'fusion_texture'],
            'decomposition': 'AUC=sum over four positive/negative region pair groups; S selected, U unselected; weighted by pair counts within each image, then image mean',
            'loss': 'full fine AUC contribution minus sparse contribution; signed, positive is deterioration',
            'interpretation': 'UU changes are within unselected regions; SU/US are cross-region rankings, not proof of calibration causality',
            'script_sha256': sha(Path(__file__))}
    (out / 'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    manifest = json.loads((data / 'manifest.json').read_text())
    protocol = json.loads((source / 'protocol.json').read_text())
    assert sha(data / 'manifest.json') == protocol['manifest_sha256']
    assert sha(selections / 'selections.json') == protocol['selection_sha256']
    frozen = json.loads((source / 'predictions_complete.json').read_text())
    assert sha(source / 'calibration.json') == frozen['calibration_sha256']
    scales = json.loads((source / 'calibration.json').read_text())
    records = json.loads((selections / 'selections.json').read_text())
    assert [r['image'] for r in records] == [r['image'] for r in manifest['test']]
    hashes = {f['path']: f['sha256'] for f in manifest['files']}
    complete = json.loads((maps / 'predictions_complete.json').read_text())
    provenance = {}
    for name in ['low', 'native']:
        assert sha(maps / name / 'predictions.json') == complete[name]['predictions_sha256']
        provenance[name] = json.loads((maps / name / 'predictions.json').read_text())
    previous = json.loads((source / 'report.json').read_text())
    details = {name: [] for name in plan['methods']}
    for i, record in enumerate(records):
        target = manifest['test'][i]
        if not target['mask']:
            continue
        assert sha(data / target['mask']) == hashes[target['mask']]
        with Image.open(data / target['mask']) as im:
            mask = np.asarray(im.convert('L')) > 0
        arrays = {}
        for name in ['low', 'native']:
            path = maps / name / f'map_{i:03d}.npy'
            assert sha(path) == provenance[name][i]['map_sha256']
            arrays[name] = normalize(np.load(path), scales[name])
        for name in details:
            selected = selection_mask(record['selected']['4'][name])
            sparse = np.where(selected, arrays['native'], arrays['low'])
            full_terms = decompose(arrays['native'], mask, selected)
            sparse_terms = decompose(sparse, mask, selected)
            assert abs(sum(v['weight'] for v in full_terms.values())-1) < 1e-12
            losses = {k: full_terms[k]['contribution']-sparse_terms[k]['contribution'] for k in full_terms}
            assert losses['SS'] == 0
            actual = sum(v['contribution'] for v in sparse_terms.values())
            expected = next(r['auc'] for r in previous['pixel_details'][name] if r['image'] == record['image'])
            assert abs(actual-expected) < 1e-10
            details[name].append({'image': record['image'], 'defect': target['defect'],
                                  'selected_defect_fraction': float(mask[selected].sum()/mask.sum()),
                                  'full_terms': full_terms, 'sparse_terms': sparse_terms,
                                  'loss_contributions': losses})
        if (i+1) % 10 == 0:
            print('diagnosed', i+1, '/78', flush=True)
    summary = {}
    for name, rows in details.items():
        losses = {k: float(np.mean([r['loss_contributions'][k] for r in rows])) for k in ['SS', 'SU', 'US', 'UU']}
        total = sum(losses.values())
        expected = previous['metrics']['all_tiles']['abnormal_macro_pixel_auc']-previous['metrics'][name]['abnormal_macro_pixel_auc']
        assert abs(total-expected) < 1e-10
        summary[name] = {'mean_auc_loss': total, 'loss_by_pair_group': losses,
                         'cross_region_loss': losses['SU']+losses['US'],
                         'mean_defect_fraction_selected': float(np.mean([r['selected_defect_fraction'] for r in rows]))}
    (out / 'report.json').write_text(json.dumps({'status': 'PASS', 'summary': summary, 'details': details}, indent=2)+'\n')
    print(json.dumps(summary, indent=2)); print('OUTPUT:', out)


if __name__ == '__main__':
    main()
