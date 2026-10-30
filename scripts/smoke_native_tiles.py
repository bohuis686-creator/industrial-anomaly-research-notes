"""Bounded normal-only native-resolution tiling integration test, not evaluation."""
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image
import torch
from torchvision.models import resnet50, ResNet50_Weights

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / 'vendor/patchcore-inspection'
sys.path.insert(0, str(REPO / 'src'))
from patchcore.patchcore import PatchCore
from patchcore.common import FaissNN
from patchcore.sampler import ApproximateGreedyCoresetSampler


def stitch(maps):
    maps = np.asarray(maps)
    assert maps.shape == (16, 256, 256)
    return maps.reshape(4, 4, 256, 256).transpose(0, 2, 1, 3).reshape(1024, 1024)


def main():
    known = stitch(np.stack([np.full((256, 256), i) for i in range(16)]))
    for i in range(16):
        row, col = divmod(i, 4)
        assert np.all(known[row*256:(row+1)*256, col*256:(col+1)*256] == i)
    data = ROOT / 'data/mvtec_grid_mirror'
    raw = (data / 'manifest.json').read_bytes()
    manifest = json.loads(raw)
    hashes = {f['path']: f['sha256'] for f in manifest['files']}
    commit = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()
    assert commit == 'fcaa92f124fb1ad74a7acf56726decd4b27cbcad'
    train = manifest['smoke_training'][:2]
    probe = manifest['normal_validation'][0]
    assert probe not in train
    out = ROOT / 'results' / datetime.now(ZoneInfo('Asia/Shanghai')).strftime('native_tile_smoke_%Y%m%d_%H%M%S')
    out.mkdir(exist_ok=False)
    config = {'purpose': 'normal-only integration/resource smoke; no accuracy claim',
              'training_images': train, 'probe_normal_image': probe,
              'tile_size': 256, 'tiles_per_image': 16, 'overlap': 0,
              'bank': 'shared across all tile positions', 'seed': 1592,
              'coreset_fraction': 0.01, 'batch_size': 1, 'backbone': 'resnet50 IMAGENET1K_V1',
              'author_commit': commit, 'manifest_sha256': hashlib.sha256(raw).hexdigest(),
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    np.random.seed(1592)
    torch.manual_seed(1592)
    torch.set_num_threads(2)
    assert torch.cuda.is_available()
    torch.hub.set_dir(str(ROOT / 'cache/torch'))
    mean = torch.tensor([.485, .456, .406])[:, None, None]
    std = torch.tensor([.229, .224, .225])[:, None, None]

    def read(path):
        assert hashlib.sha256((data / path).read_bytes()).hexdigest() == hashes[path]
        with Image.open(data / path) as im:
            arr = np.array(im.convert('RGB'), copy=True)
        assert arr.shape == (1024, 1024, 3)
        return arr

    def crops(arr):
        for row in range(4):
            for col in range(4):
                crop = np.array(arr[row*256:(row+1)*256, col*256:(col+1)*256], copy=True)
                yield ((torch.from_numpy(crop).permute(2, 0, 1).float()/255-mean)/std).unsqueeze(0)

    device = torch.device('cuda:0')
    backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).eval()
    backbone.name = 'resnet50'
    backbone.requires_grad_(False)
    model = PatchCore(device)
    model.load(backbone, ['layer2', 'layer3'], device, (3, 256, 256), 1024, 1024,
               featuresampler=ApproximateGreedyCoresetSampler(.01, device), nn_method=FaissNN(False, 2))
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    model.fit((crop for path in train for crop in crops(read(path))))
    torch.cuda.synchronize()
    fit_seconds = time.perf_counter()-start
    fit_peak = torch.cuda.max_memory_allocated()/1024**2
    (out / 'model').mkdir()
    model.save_to_path(str(out / 'model'))
    model.predict(next(crops(read(probe))))
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for repeat in range(3):
        start = time.perf_counter()
        maps, scores = [], []
        for crop in crops(read(probe)):
            score, amap = model.predict(crop)
            maps.append(amap[0]); scores.append(score[0])
        full = stitch(maps)
        torch.cuda.synchronize()
        times.append((time.perf_counter()-start)*1000)
        assert np.isfinite(full).all() and np.isfinite(scores).all()
        if repeat == 0:
            reference = full.copy()
        else:
            np.testing.assert_allclose(full, reference, rtol=1e-6, atol=1e-6)
    np.save(out / 'normal_probe_map.npy', reference)
    report = {'status': 'PASS', 'train_images': 2, 'train_tiles': 32,
              'probe_images': 1, 'probe_repeats': 3, 'test_images_used': 0,
              'fit_seconds': fit_seconds, 'fit_peak_allocated_mib': fit_peak,
              'inference_peak_allocated_mib': torch.cuda.max_memory_allocated()/1024**2,
              'inference_peak_reserved_mib': torch.cuda.max_memory_reserved()/1024**2,
              'latency_ms': times, 'latency_median_ms': float(np.median(times)),
              'memory_vectors': model.anomaly_scorer.nn_method.search_index.ntotal,
              'map_shape': list(reference.shape), 'stitch_known_answer': 'PASS',
              'repeat_consistency': 'PASS',
              'timing_scope': 'warm; file hash+decode+16 crops+features+CPU FAISS+stitch; excludes save',
              'limitations': 'two normal training images; one normal probe; not accuracy or fair speed comparison; reserved memory includes fit allocator cache'}
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2)); print('OUTPUT:', out)


if __name__ == '__main__':
    main()
