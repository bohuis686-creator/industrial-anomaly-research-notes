"""Small real-data integration baseline, not a paper-quality evaluation."""
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw
from scipy.stats import rankdata
import torch
from torchvision.models import resnet50, ResNet50_Weights

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/mvtec_grid_mirror'
REPO = ROOT / 'vendor/patchcore-inspection'
sys.path.insert(0, str(REPO / 'src'))
from patchcore.patchcore import PatchCore
from patchcore.common import FaissNN
from patchcore.sampler import ApproximateGreedyCoresetSampler

manifest_bytes = (DATA / 'manifest.json').read_bytes()
manifest = json.loads(manifest_bytes)
COMMIT = 'fcaa92f124fb1ad74a7acf56726decd4b27cbcad'
assert subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip() == COMMIT
OUT = ROOT / 'results' / datetime.now(ZoneInfo('Asia/Shanghai')).strftime('grid_smoke_%Y%m%d_%H%M%S')
OUT.mkdir(parents=True, exist_ok=False)
config = {'seed': 1592, 'input_size': 256, 'train_count': 20, 'coreset': 0.01,
          'backbone': 'resnet50', 'weights': 'IMAGENET1K_V1', 'embedding_dim': 1024,
          'patchsize': 3, 'nn': 1, 'backend': 'CPU FAISS', 'author_commit': COMMIT,
          'manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
          'purpose': '20-normal-image engineering baseline, not full-training paper results'}
(OUT / 'config.json').write_text(json.dumps(config, indent=2))
np.random.seed(1592)
torch.manual_seed(1592)
torch.set_num_threads(2)
torch.cuda.init()
torch.hub.set_dir(str(ROOT / 'cache/torch'))
mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]

def load(path):
    with Image.open(DATA / path) as im:
        arr = np.array(im.convert('RGB').resize((256, 256), Image.Resampling.BILINEAR), copy=True)
    return ((torch.from_numpy(arr).permute(2, 0, 1).float()/255 - mean) / std).unsqueeze(0)

device = torch.device('cuda:0')
backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).eval()
backbone.name = 'resnet50'
backbone.requires_grad_(False)
model = PatchCore(device)
model.load(backbone, ['layer2', 'layer3'], device, (3, 256, 256), 1024, 1024,
           featuresampler=ApproximateGreedyCoresetSampler(0.01, device), nn_method=FaissNN(False, 2))
torch.cuda.reset_peak_memory_stats(0)
start = time.perf_counter()
model.fit((load(p) for p in manifest['smoke_training']))
torch.cuda.synchronize()
fit_seconds = time.perf_counter() - start
fit_peak = torch.cuda.max_memory_allocated(0)/1024**2
(OUT / 'model').mkdir()
model.save_to_path(str(OUT / 'model'))
# Normal validation only determines visualization scale, not test labels.
validation = []
for p in manifest['normal_validation']:
    _, maps = model.predict(load(p))
    validation.append(np.asarray(maps[0]))
lo, hi = np.quantile(np.stack(validation), [0.01, 0.99])
assert hi > lo
for _ in range(3):
    model.predict(load(manifest['smoke_training'][0]))
torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats(0)
predictions = []
for i, record in enumerate(manifest['test']):
    torch.cuda.synchronize()
    start = time.perf_counter()
    scores, maps = model.predict(load(record['image']))
    torch.cuda.synchronize()
    elapsed = (time.perf_counter()-start)*1000
    amap = np.asarray(maps[0], dtype=np.float32)
    assert amap.shape == (256, 256) and np.isfinite(amap).all() and np.isfinite(scores).all()
    np.save(OUT / f'map_{i:03d}.npy', amap)
    predictions.append({'image': record['image'], 'score': float(scores[0]), 'milliseconds': elapsed})
inference_peak = torch.cuda.max_memory_allocated(0)/1024**2
(OUT / 'predictions.json').write_text(json.dumps(predictions, indent=2))
# Evaluation begins only after all predictions are saved. No feedback into fitting.
labels = np.array([r['defect'] != 'good' for r in manifest['test']])
scores = np.array([r['score'] for r in predictions])
n1, n0 = labels.sum(), (~labels).sum()
auc = (rankdata(scores)[labels].sum() - n1*(n1+1)/2)/(n1*n0)
# Fixed first example of each test category; not selected by quality of prediction.
seen = set()
for i, record in enumerate(manifest['test']):
    category = record['defect']
    if category in seen:
        continue
    seen.add(category)
    with Image.open(DATA / record['image']) as im:
        original = im.convert('RGB').resize((256,256))
    if record['mask']:
        with Image.open(DATA / record['mask']) as im:
            mask = im.convert('L').resize((256,256), Image.Resampling.NEAREST)
    else:
        mask = Image.new('L', (256,256))
    a = np.clip((np.load(OUT / f'map_{i:03d}.npy') - lo)/(hi-lo), 0, 1)
    # Fixed blue-to-red scale; bounds use normal validation only.
    heat = np.stack([a, np.zeros_like(a), 1-a], axis=-1)
    heat = Image.fromarray((heat*255).astype(np.uint8))
    canvas = Image.new('RGB', (768,300), 'white')
    canvas.paste(original, (0,24)); canvas.paste(Image.blend(original,heat,0.5),(256,24)); canvas.paste(mask.convert('RGB'),(512,24))
    draw = ImageDraw.Draw(canvas)
    draw.text((5,5), f'Input: {category}', fill='black')
    draw.text((261,5), 'Anomaly overlay (blue=low, red=high)', fill='black')
    draw.text((517,5), 'Ground truth (evaluation only)', fill='black')
    draw.text((5,283), 'MVTec AD / MVTec Software GmbH; CC BY-NC-SA 4.0; resized + overlay', fill='black')
    canvas.save(OUT / f'preview_{category}.png')
times = [p['milliseconds'] for p in predictions]
report = {'status':'PASS', 'config':config, 'train_images':20, 'held_out_normal':53,
          'test_images':len(predictions), 'image_auroc_exploratory':float(auc),
          'fit_seconds':fit_seconds, 'fit_peak_allocated_mib':fit_peak,
          'inference_peak_allocated_mib':inference_peak,
          'latency_median_ms':float(np.median(times)), 'latency_p95_ms':float(np.percentile(times,95)),
          'timing_scope':'single pass, warm; decode+resize+normalize+GPU features+CPU FAISS+map; excludes saving',
          'normal_only_display_range':[float(lo),float(hi)],
          'memory_vectors':model.anomaly_scorer.nn_method.search_index.ntotal,
          'limitations':'20 training images; one class; one seed; mirror; no pixel metrics; no region selection'}
(OUT / 'report.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2)); print('OUTPUT:',OUT)
