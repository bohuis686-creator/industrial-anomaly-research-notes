"""Frozen K=4 development evaluation; normal-only map calibration."""
import hashlib
import json
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch
from torchvision.models import resnet50, ResNet50_Weights

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from compare_resolution import auc
from smoke_native_tiles import stitch
from patchcore.patchcore import PatchCore
from patchcore.common import FaissNN
from selection_core import selection_mask, image_cues, tail_mean, topk, correct, normalize


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    data = ROOT / 'data/mvtec_grid_mirror'
    prior = ROOT / 'results/resolution_comparison_20261030_104915'
    selected = ROOT / 'results/selection_20261030_015741'
    base = ROOT / 'results/grid_smoke_20261030_014348'
    manifest = json.loads((data / 'manifest.json').read_text())
    records = json.loads((selected / 'selections.json').read_text())
    assert sha(selected / 'selections.json') == json.loads((selected / 'selection_complete.json').read_text())['selection_sha256']
    assert sha(data / 'manifest.json') == json.loads((prior / 'protocol.json').read_text())['manifest_sha256']
    assert [r['image'] for r in records] == [r['image'] for r in manifest['test']]
    assert not set(manifest['normal_validation']) & set(manifest['smoke_training'])
    hashes = {f['path']: f['sha256'] for f in manifest['files']}
    methods = list(records[0]['selected']['4'])
    methods.remove('all_tiles')
    names = ['low_only', 'all_tiles'] + methods
    out = ROOT / 'results' / datetime.now(ZoneInfo('Asia/Shanghai')).strftime('sparse_refinement_%Y%m%d_%H%M%S')
    out.mkdir(exist_ok=False)
    protocol = {'purpose': 'grid development only; frozen selections, no tuning', 'k': 4,
                'methods': names, 'calibration_images': manifest['normal_validation'],
                'map_calibration': 'each detector separately: q50 and q95-q50 of normal maps at fixed [8::16,8::16] pixel lattice; floor 1e-6; no clipping',
                'merge': 'replace calibrated coarse pixels inside selected tiles with calibrated fine pixels; outside retain coarse; no blend',
                'image_score': 'max of merged anomaly map for EVERY method, including low_only/all_tiles; differs from previous raw patch-score image AUROC',
                'localization': 'mean per-abnormal-image pixel AUROC at 1024, not pooled AUROC',
                'timing': 'actual coarse/texture-fusion/all-tiles on first 10 manifest test images, warm single pass; hash+decode+selection+inference+merge; excludes model load/save; different scope from full 78-image prior timing',
                'timing_order': ['coarse', 'fusion_texture', 'all_tiles'],
                'script_sha256': sha(Path(__file__)), 'selection_sha256': sha(selected / 'selections.json'),
                'manifest_sha256': sha(data / 'manifest.json'),
                'model_hashes': {k: sha(p / 'nnscorer_search_index.faiss') for k, p in
                                 [('low', base / 'model'), ('native', prior / 'native/model')]}}
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    print('OUTPUT:', out, flush=True)
    torch.set_num_threads(2); torch.manual_seed(1592); np.random.seed(1592)
    torch.hub.set_dir(str(ROOT / 'cache/torch'))
    device = torch.device('cuda:0')
    mean = torch.tensor([.485, .456, .406])[:, None, None]
    std = torch.tensor([.229, .224, .225])[:, None, None]
    models = {}
    for name, path in [('low', base / 'model'), ('native', prior / 'native/model')]:
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).eval()
        backbone.requires_grad_(False)
        model = PatchCore(device)
        model.load(backbone, ['layer2', 'layer3'], device, (3, 256, 256), 1024, 1024, nn_method=FaissNN(False, 2))
        model.anomaly_scorer.load(str(path)); models[name] = model

    def read(path):
        assert sha(data / path) == hashes[path]
        with Image.open(data / path) as im:
            return im.convert('RGB')

    def predict(image, name):
        arr = np.array(image, copy=True)
        x = ((torch.from_numpy(arr).permute(2, 0, 1).float()/255-mean)/std).unsqueeze(0)
        return np.asarray(models[name].predict(x)[1][0], np.float32)

    def fine(image, i):
        r, c = divmod(i, 4)
        return predict(image.crop((c*256, r*256, (c+1)*256, (r+1)*256)), 'native')

    def enlarge(amap):
        return np.asarray(Image.fromarray(amap).resize((1024, 1024), Image.Resampling.BILINEAR))

    samples = {'low': [], 'native': []}
    for i, path in enumerate(manifest['normal_validation']):
        im = read(path)
        low = enlarge(predict(im.resize((256, 256), Image.Resampling.BILINEAR), 'low'))
        high = stitch([fine(im, j) for j in range(16)])
        for name, amap in [('low', low), ('native', high)]:
            samples[name].append(amap[8::16, 8::16].ravel())
        if (i+1) % 10 == 0:
            print('normal calibration', i+1, '/53', flush=True)
    scales = {}
    for name, values in samples.items():
        q50, q95 = np.quantile(np.concatenate(values), [.5, .95])
        scales[name] = {'center': float(q50), 'scale': max(float(q95-q50), 1e-6)}
    (out / 'calibration.json').write_text(json.dumps(scales, indent=2)+'\n')

    def merged(low, high, ids):
        result = normalize(low, scales['low']).copy()
        mask = selection_mask(ids)
        result[mask] = normalize(high, scales['native'])[mask]
        return result

    # Freeze all image predictions and input hashes before reading defect masks.
    completion = json.loads((prior / 'predictions_complete.json').read_text())
    provenance = {}
    for name in ['low', 'native']:
        assert sha(prior / name / 'predictions.json') == completion[name]['predictions_sha256']
        provenance[name] = json.loads((prior / name / 'predictions.json').read_text())
    image_scores = {name: [] for name in names}
    for i, record in enumerate(records):
        maps = {}
        for name in ['low', 'native']:
            path = prior / name / f'map_{i:03d}.npy'
            assert sha(path) == provenance[name][i]['map_sha256']
            maps[name] = np.load(path)
        for name in names:
            ids = [] if name == 'low_only' else list(range(16)) if name == 'all_tiles' else record['selected']['4'][name]
            amap = merged(maps['low'], maps['native'], ids)
            assert np.isfinite(amap).all()
            image_scores[name].append(float(amap.max()))
    (out / 'image_scores.json').write_text(json.dumps(image_scores, indent=2)+'\n')
    (out / 'predictions_complete.json').write_text(json.dumps({'image_scores_sha256': sha(out / 'image_scores.json'), 'calibration_sha256': sha(out / 'calibration.json')}, indent=2)+'\n')

    # Execute real sparse pathways, not time slicing cached full-resolution maps.
    oldcal = json.loads((selected / 'calibration.json').read_text())
    timing = {}
    for name in ['coarse', 'fusion_texture', 'all_tiles']:
        rows = []
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        for i, record in enumerate(records[:10]):
            torch.cuda.synchronize(); start = time.perf_counter()
            im = read(record['image'])
            if name == 'all_tiles':
                result = normalize(stitch([fine(im, j) for j in range(16)]), scales['native'])
            else:
                if name == 'fusion_texture':
                    lowim, residual, texture, _ = image_cues(im)
                else:
                    lowim = im.resize((256, 256), Image.Resampling.BILINEAR)
                lowmap = predict(lowim, 'low')
                score = tail_mean(lowmap)
                if name == 'fusion_texture':
                    z = correct(residual, texture, oldcal['texture'])
                    score = .5*normalize(score, oldcal['scales']['a']) + .5*normalize(z, oldcal['scales']['z'])
                ids = topk(score, 4)
                result = normalize(enlarge(lowmap), scales['low']).copy()
                for j in ids:
                    r, c = divmod(j, 4)
                    result[r*256:(r+1)*256, c*256:(c+1)*256] = normalize(fine(im, j), scales['native'])
            torch.cuda.synchronize(); elapsed = (time.perf_counter()-start)*1000
            if name != 'all_tiles':
                assert ids == record['selected']['4'][name]
            expected = merged(np.load(prior / 'low' / f'map_{i:03d}.npy'), np.load(prior / 'native' / f'map_{i:03d}.npy'), list(range(16)) if name == 'all_tiles' else ids)
            np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)
            rows.append({'image': record['image'], 'ms': elapsed})
        timing[name] = {'rows': rows, 'median_ms': float(np.median([r['ms'] for r in rows])),
                        'peak_allocated_mib': torch.cuda.max_memory_allocated()/1024**2}
        print('timed', name, timing[name]['median_ms'], flush=True)
    (out / 'timing.json').write_text(json.dumps(timing, indent=2)+'\n')
    pixels = {name: [] for name in names}
    for i, record in enumerate(records):
        target = manifest['test'][i]
        if not target['mask']:
            continue
        assert sha(data / target['mask']) == hashes[target['mask']]
        with Image.open(data / target['mask']) as im:
            mask = np.asarray(im.convert('L')) > 0
        low = np.load(prior / 'low' / f'map_{i:03d}.npy')
        high = np.load(prior / 'native' / f'map_{i:03d}.npy')
        for name in names:
            ids = [] if name == 'low_only' else list(range(16)) if name == 'all_tiles' else record['selected']['4'][name]
            pixels[name].append({'image': record['image'], 'auc': auc(mask, merged(low, high, ids))})
        if (i+1) % 10 == 0:
            print('evaluated', i+1, '/78', flush=True)
    labels = [r['defect'] != 'good' for r in manifest['test']]
    metrics = {name: {'image_auroc_mapmax': auc(labels, image_scores[name]),
                      'abnormal_macro_pixel_auc': float(np.mean([r['auc'] for r in pixels[name]]))} for name in names}
    randoms = [name for name in names if name.startswith('random_')]
    metrics['random_mean'] = {k: float(np.mean([metrics[n][k] for n in randoms])) for k in metrics['low_only']}
    (out / 'report.json').write_text(json.dumps({'status': 'PASS', 'metrics': metrics, 'pixel_details': pixels, 'timing': timing}, indent=2)+'\n')
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == '__main__':
    main()
