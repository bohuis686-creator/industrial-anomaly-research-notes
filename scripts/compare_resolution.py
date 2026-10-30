"""Fixed development comparison; predictions frozen before labels are evaluated."""
import gc
import hashlib
import json
from datetime import datetime
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

import numpy as np
from PIL import Image
from scipy.stats import rankdata
import torch
from torchvision.models import resnet50, ResNet50_Weights

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from smoke_native_tiles import stitch
from patchcore.patchcore import PatchCore
from patchcore.common import FaissNN
from patchcore.sampler import ApproximateGreedyCoresetSampler


def auc(labels, scores):
    labels = np.asarray(labels, bool).ravel()
    scores = np.asarray(scores).ravel()
    p, n = int(labels.sum()), int((~labels).sum())
    assert p and n and np.isfinite(scores).all()
    return float((rankdata(scores)[labels].sum()-p*(p+1)/2)/(p*n))


def main():
    assert auc([0, 1], [0, 1]) == 1
    assert auc([0, 1], [1, 0]) == 0
    assert auc([0, 1], [1, 1]) == .5
    data = ROOT / 'data/mvtec_grid_mirror'
    old = ROOT / 'results/grid_smoke_20261030_014348'
    raw = (data / 'manifest.json').read_bytes()
    manifest = json.loads(raw)
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    hashes = {f['path']: f['sha256'] for f in manifest['files']}
    train = manifest['smoke_training']
    test = [r['image'] for r in manifest['test']]
    assert len(train) == 20 and not set(train) & set(test)
    assert hashlib.sha256(raw).hexdigest() == json.loads((old / 'config.json').read_text())['manifest_sha256']
    repo = ROOT / 'vendor/patchcore-inspection'
    commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    assert commit == 'fcaa92f124fb1ad74a7acf56726decd4b27cbcad'
    out = ROOT / 'results' / datetime.now(ZoneInfo('Asia/Shanghai')).strftime('resolution_comparison_%Y%m%d_%H%M%S')
    out.mkdir(exist_ok=False)
    protocol = {'scope': 'grid development; not independent final evaluation',
                'train': train, 'test_image_order': test, 'seed': 1592,
                'backbone': 'resnet50 IMAGENET1K_V1 layer2+layer3',
                'embedding_dim': 1024, 'coreset_fraction': .01, 'nn': 1,
                'low': 'reuse frozen 20-image memory bank; bilinear RGB resize to 256',
                'native': '20 images -> 320 nonoverlap 256 tiles; shared bank; batch=1',
                'image_score': 'maximum author patchcore image score across tiles',
                'localization': 'per-image pixel AUROC at 1024 resolution, mean over abnormal images only; NOT pooled pixel AUROC',
                'low_map_resize': 'PIL float bilinear 256 to 1024',
                'timing': 'one warm pass; SHA check+decode+preprocess+features+CPU FAISS+map stitching/resize; excludes saving',
                'caveat': 'same normal images but unequal feature counts, memory banks, compute and context; not causal isolation of resolution',
                'author_commit': commit, 'manifest_sha256': hashlib.sha256(raw).hexdigest(),
                'script_sha256': digest(Path(__file__)),
                'low_bank_sha256': digest(old / 'model/nnscorer_search_index.faiss')}
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    print('OUTPUT:', out, flush=True)
    torch.set_num_threads(2)
    torch.hub.set_dir(str(ROOT / 'cache/torch'))
    device = torch.device('cuda:0')
    assert torch.cuda.is_available()
    mean = torch.tensor([.485, .456, .406])[:, None, None]
    std = torch.tensor([.229, .224, .225])[:, None, None]

    def read(path):
        assert digest(data / path) == hashes[path]
        with Image.open(data / path) as im:
            image = im.convert('RGB')
        assert image.size == (1024, 1024)
        return image

    def tensors(image, mode):
        if mode == 'low':
            arrays = [np.array(image.resize((256, 256), Image.Resampling.BILINEAR), copy=True)]
        else:
            arr = np.asarray(image)
            arrays = (np.array(arr[r:r+256, c:c+256], copy=True)
                      for r in range(0, 1024, 256) for c in range(0, 1024, 256))
        for crop_array in arrays:
            assert crop_array.shape == (256, 256, 3)
            yield ((torch.from_numpy(crop_array).permute(2, 0, 1).float()/255-mean)/std).unsqueeze(0)

    runs = {}
    for mode in ['low', 'native']:
        np.random.seed(1592); torch.manual_seed(1592)
        target = out / mode
        target.mkdir()
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).eval()
        backbone.name = 'resnet50'; backbone.requires_grad_(False)
        model = PatchCore(device)
        model.load(backbone, ['layer2', 'layer3'], device, (3, 256, 256), 1024, 1024,
                   featuresampler=ApproximateGreedyCoresetSampler(.01, device), nn_method=FaissNN(False, 2))
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        if mode == 'low':
            model.anomaly_scorer.load(str(old / 'model'))
        else:
            model.fit((x for p in train for x in tensors(read(p), mode)))
            (target / 'model').mkdir()
            model.save_to_path(str(target / 'model'))
        torch.cuda.synchronize()
        preparation = {'seconds_including_load_or_save': time.perf_counter()-start,
                       'peak_allocated_mib': torch.cuda.max_memory_allocated()/1024**2,
                       'kind': 'load existing bank' if mode == 'low' else 'fit and save bank'}
        for _ in range(3):
            model.predict(next(tensors(read(train[0]), mode)))
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        records = []
        for i, path in enumerate(test):
            torch.cuda.synchronize(); start = time.perf_counter()
            scores, maps = [], []
            for x in tensors(read(path), mode):
                s, m = model.predict(x)
                scores.append(float(s[0])); maps.append(m[0])
            if mode == 'low':
                full = np.asarray(Image.fromarray(np.asarray(maps[0], np.float32)).resize((1024, 1024), Image.Resampling.BILINEAR))
            else:
                full = stitch(maps).astype(np.float32)
            torch.cuda.synchronize(); elapsed = (time.perf_counter()-start)*1000
            if mode == 'low':
                np.testing.assert_allclose(maps[0], np.load(old / f'map_{i:03d}.npy'), rtol=1e-5, atol=1e-5)
            assert full.shape == (1024, 1024) and np.isfinite(full).all()
            np.save(target / f'map_{i:03d}.npy', full)
            records.append({'image': path, 'score': max(scores), 'ms': elapsed,
                            'map_sha256': digest(target / f'map_{i:03d}.npy')})
            if (i+1) % 20 == 0:
                print(mode, i+1, '/', len(test), flush=True)
        (target / 'predictions.json').write_text(json.dumps(records, indent=2)+'\n')
        runs[mode] = {'preparation': preparation,
                      'memory_vectors': model.anomaly_scorer.nn_method.search_index.ntotal,
                      'inference_peak_allocated_mib': torch.cuda.max_memory_allocated()/1024**2,
                      'latency_median_ms': float(np.median([r['ms'] for r in records])),
                      'latency_p95_ms': float(np.percentile([r['ms'] for r in records], 95)),
                      'predictions_sha256': digest(target / 'predictions.json')}
        del model, backbone
        gc.collect(); torch.cuda.empty_cache()
    (out / 'predictions_complete.json').write_text(json.dumps(runs, indent=2)+'\n')
    # No ground-truth mask was opened before both prediction sets were frozen.
    labels = [r['defect'] != 'good' for r in manifest['test']]
    details = {}
    for mode in ['low', 'native']:
        target = out / mode
        assert digest(target / 'predictions.json') == runs[mode]['predictions_sha256']
        records = json.loads((target / 'predictions.json').read_text())
        rows = []
        for i, item in enumerate(manifest['test']):
            if not item['mask']:
                continue
            assert digest(data / item['mask']) == hashes[item['mask']]
            assert digest(target / f'map_{i:03d}.npy') == records[i]['map_sha256']
            with Image.open(data / item['mask']) as im:
                mask = np.asarray(im.convert('L')) > 0
            rows.append({'image': item['image'], 'defect': item['defect'],
                         'pixel_auc': auc(mask, np.load(target / f'map_{i:03d}.npy'))})
        details[mode] = rows
        runs[mode]['image_auroc'] = auc(labels, [r['score'] for r in records])
        runs[mode]['abnormal_image_macro_pixel_auroc'] = float(np.mean([r['pixel_auc'] for r in rows]))
    delta = np.array([r['pixel_auc'] for r in details['native']])-np.array([r['pixel_auc'] for r in details['low']])
    indices = np.random.default_rng(4312).integers(0, len(delta), (2000, len(delta)))
    report = {'status': 'PASS', 'runs': runs, 'pixel_macro_delta': float(delta.mean()),
              'pixel_delta_exploratory_bootstrap_95': np.quantile(delta[indices].mean(axis=1), [.025, .975]).tolist(),
              'pixel_improved_images': int((delta > 0).sum()), 'pixel_worsened_images': int((delta < 0).sum()),
              'details': details, 'limitations': protocol['caveat']}
    (out / 'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'details'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
