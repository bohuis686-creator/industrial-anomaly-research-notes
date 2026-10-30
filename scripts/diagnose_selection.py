"""Post-hoc diagnosis only. Never modifies frozen predictions or tunes parameters."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from selection_core import tiles


def pair_auc(positive, negative):
    if not len(positive) or not len(negative):
        return None
    delta = np.asarray(positive)[:, None] - np.asarray(negative)[None, :]
    return float(np.mean((delta > 0) + 0.5 * (delta == 0)))


def main():
    assert pair_auc([2], [1]) == 1
    assert pair_auc([1], [2]) == 0
    assert pair_auc([1], [1]) == 0.5
    assert pair_auc([], [1]) is None
    run = ROOT / 'results/selection_20261030_015741'
    out = ROOT / 'results/diagnosis_selection_v1'
    out.mkdir(exist_ok=False)
    # Written before loading labels; this is a diagnostic plan, not preregistration.
    plan = {'scope': 'post-hoc grid development diagnosis; primary frozen K=4 only',
            'positive_tile': 'any defect pixel; boundary fragments count',
            'ranking': 'within image: missed defective tiles versus defect-free unselected tiles; ties=0.5',
            'recovery': 'missed defect pixels intersecting each frozen top-4 selection',
            'constraints': 'no tuning, no new selector, no causal claim about downsampling',
            'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    frozen = json.loads((run / 'selection_complete.json').read_text())
    raw = (run / 'selections.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == frozen['selection_sha256']
    records = json.loads(raw)
    data = ROOT / 'data/mvtec_grid_mirror'
    raw_manifest = (data / 'manifest.json').read_bytes()
    protocol = json.loads((run / 'protocol.json').read_text())
    assert hashlib.sha256(raw_manifest).hexdigest() == protocol['manifest_sha256']
    manifest = json.loads(raw_manifest)
    hashes = {f['path']: f['sha256'] for f in manifest['files']}
    assert [r['image'] for r in records] == [r['image'] for r in manifest['test']]
    methods = ['residual', 'corrected_residual', 'gradient', 'fusion_global', 'fusion_texture']
    rows = []
    for record, target in zip(records, manifest['test']):
        if not target['mask']:
            continue
        path = data / target['mask']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == hashes[target['mask']]
        with Image.open(path) as im:
            mask = np.asarray(im.convert('L')) > 0
        counts = tiles(mask).sum(axis=1)
        coarse = np.zeros(16, bool)
        coarse[record['selected']['4']['coarse']] = True
        missed = (counts > 0) & ~coarse
        negative = (counts == 0) & ~coarse
        total = int(counts.sum())
        row = {'image': record['image'], 'defect': target['defect'],
               'defect_pixels': total, 'missed_pixels': int(counts[missed].sum()),
               'missed_tiles': int(missed.sum()), 'methods': {}}
        for name in methods:
            scores = np.asarray(record['scores'][name])
            chosen = np.zeros(16, bool)
            chosen[record['selected']['4'][name]] = True
            gain = int(counts[~coarse & chosen].sum())
            loss = int(counts[coarse & ~chosen].sum())
            assert int(counts[chosen].sum() - counts[coarse].sum()) == gain - loss
            ranks = np.empty(16, int)
            ranks[np.argsort(-scores, kind='stable')] = np.arange(1, 17)
            row['methods'][name] = {
                'gain_pixels': gain, 'loss_pixels': loss,
                'missed_tiles_in_top4': int((missed & chosen).sum()),
                'missed_tile_ranks': ranks[missed].tolist(),
                'unselected_tile_auc': pair_auc(scores[missed], scores[negative])}
        rows.append(row)
    total_missed = sum(r['missed_pixels'] for r in rows)
    summary = {}
    for name in methods:
        aucs = [r['methods'][name]['unselected_tile_auc'] for r in rows
                if r['methods'][name]['unselected_tile_auc'] is not None]
        summary[name] = {
            'recovered_fraction_of_all_missed_pixels': sum(r['methods'][name]['gain_pixels'] for r in rows) / total_missed,
            'macro_gain': float(np.mean([r['methods'][name]['gain_pixels']/r['defect_pixels'] for r in rows])),
            'macro_loss': float(np.mean([r['methods'][name]['loss_pixels']/r['defect_pixels'] for r in rows])),
            'recovered_missed_tiles': sum(r['methods'][name]['missed_tiles_in_top4'] for r in rows),
            'mean_within_image_auc': float(np.mean(aucs)) if aucs else None,
            'auc_eligible_images': len(aucs)}
    report = {'selection_sha256': frozen['selection_sha256'], 'abnormal_images': len(rows),
              'images_with_missed_pixels': sum(r['missed_pixels'] > 0 for r in rows),
              'missed_tiles': sum(r['missed_tiles'] for r in rows),
              'missed_pixels': total_missed, 'summary': summary, 'rows': rows}
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'rows'}, indent=2))
    print('OUTPUT:', out)


if __name__ == '__main__':
    main()
