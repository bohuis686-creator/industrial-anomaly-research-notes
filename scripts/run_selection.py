"""Freeze rules and produce selections without consulting defect labels/masks."""
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'vendor/patchcore-inspection/src'))
import numpy as np
from PIL import Image
import torch
from torchvision.models import resnet50, ResNet50_Weights
from patchcore.patchcore import PatchCore
from patchcore.common import FaissNN
from selection_core import image_cues, tail_mean, fit_texture, correct, fit_scale, normalize, topk

DATA = ROOT / 'data/mvtec_grid_mirror'
BASE = ROOT / 'results/grid_smoke_20261030_014348'
COMMIT = 'fcaa92f124fb1ad74a7acf56726decd4b27cbcad'
assert subprocess.check_output(['git', '-C', str(ROOT/'vendor/patchcore-inspection'), 'rev-parse', 'HEAD'], text=True).strip() == COMMIT
manifest_bytes = (DATA/'manifest.json').read_bytes()
manifest = json.loads(manifest_bytes)
# Strip evaluation fields from the input contract. No masks are opened here.
inputs = {'train': manifest['normal_training'], 'validation': manifest['normal_validation'],
          'test': [r['image'] for r in manifest['test']]}
assert set(inputs['train']).isdisjoint(inputs['validation'])
assert set(inputs['train']+inputs['validation']).isdisjoint(inputs['test'])
file_hashes = {x['path']: x['sha256'] for x in manifest['files']}
del manifest
OUT = ROOT/'results'/datetime.now(ZoneInfo('Asia/Shanghai')).strftime('selection_%Y%m%d_%H%M%S')
OUT.mkdir(exist_ok=False)
protocol = {
    'version': 'selection_v1', 'created': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
    'category': 'grid (development only)', 'main_k': 4, 'secondary_k': 8,
    'tile_size': 256, 'grid': [4, 4], 'random_seeds': [1592,1593,1594,1595,1596],
    'coarse_model': str(BASE.relative_to(ROOT)), 'coarse_training_images': 20,
    'coarse_model_sha256': hashlib.sha256((BASE/'model/nnscorer_search_index.faiss').read_bytes()).hexdigest(),
    'manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(), 'author_commit': COMMIT,
    'residual': 'mean RGB abs(original - PIL bilinear up(down(original,256),1024)); top 1% mean per tile',
    'coarse_score': 'top 1% mean of the corresponding 64x64 area of the coarse 256 map',
    'texture': 'mean absolute horizontal/vertical grayscale gradient within each low-res 64x64 tile',
    'correction': '4 training-normal texture quartile bins; (residual-bin_median)/(1.4826*MAD); min 128 samples else global; floor 1e-6',
    'normal_calibration': '(score-normal_validation_q50)/(q95-q50); floor 1e-6; no clipping',
    'fusion_weights': [0.5,0.5], 'tie_break': 'row-major stable sort',
    'component_connectivity': 8, 'component_coverage_threshold': 0.5,
    'bootstrap': '2000 paired image resamples, seed 4312; exploratory intervals',
    'high_resolution_detection': 'NOT RUN; coverage only',
    'scripts_sha256': {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                       [Path(__file__), ROOT/'scripts/selection_core.py', ROOT/'scripts/evaluate_selection.py']},
}
(OUT/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
(OUT/'inputs.json').write_text(json.dumps(inputs, indent=2)+'\n')
torch.manual_seed(1592); np.random.seed(1592); torch.set_num_threads(2)
torch.cuda.init(); torch.hub.set_dir(str(ROOT/'cache/torch'))
device = torch.device('cuda:0')
backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).eval()
backbone.requires_grad_(False)
model = PatchCore(device)
model.load(backbone, ['layer2','layer3'], device, (3,256,256), 1024,1024,
           patchsize=3, anomaly_score_num_nn=1, nn_method=FaissNN(False,2))
model.anomaly_scorer.load(str(BASE/'model'))
mean = torch.tensor([0.485,0.456,0.406])[:,None,None]
std = torch.tensor([0.229,0.224,0.225])[:,None,None]

def cues(path, inference=True):
    assert hashlib.sha256((DATA/path).read_bytes()).hexdigest() == file_hashes[path]
    with Image.open(DATA/path) as im:
        image = im.convert('RGB')
    low, r, t, g = image_cues(image)
    a = None
    if inference:
        tensor = torch.from_numpy(np.array(low,copy=True)).permute(2,0,1).float()/255
        _, maps = model.predict(((tensor-mean)/std).unsqueeze(0))
        a = tail_mean(np.asarray(maps[0]))
    return {'a':a, 'r':r, 't':t, 'g':g}

train = [cues(p, False) for p in inputs['train']]
texture_model = fit_texture([x['r'] for x in train], [x['t'] for x in train])
validation = [cues(p) for p in inputs['validation']]
for x in validation:
    x['z'] = correct(x['r'], x['t'], texture_model)
scales = {k:fit_scale([x[k] for x in validation]) for k in ['a','r','z']}
(OUT/'calibration.json').write_text(json.dumps({'texture':texture_model,'scales':scales}, indent=2)+'\n')
print('Normal-only calibration complete', flush=True)
# Warm-up before collecting diagnostic shared-path timings.
for _ in range(3):
    cues(inputs['validation'][0])
torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(0)
records = []
for path in inputs['test']:
    start = time.perf_counter()
    x = cues(path)
    x['z'] = correct(x['r'],x['t'],texture_model)
    a,r,z = [normalize(x[k],scales[k]) for k in ['a','r','z']]
    scores = {'coarse':x['a'], 'residual':x['r'], 'gradient':x['g'],
              'corrected_residual':x['z'], 'fusion_global':0.5*a+0.5*r,
              'fusion_texture':0.5*a+0.5*z}
    choices = {}
    for k in [4,8]:
        choices[str(k)] = {name:topk(s,k) for name,s in scores.items()}
        for seed in protocol['random_seeds']:
            digest = hashlib.sha256(f'{seed}:{path}'.encode()).digest()
            rng = np.random.default_rng(int.from_bytes(digest[:8],'little'))
            choices[str(k)][f'random_{seed}'] = rng.permutation(16)[:k].tolist()
        choices[str(k)]['all_tiles'] = list(range(16))
    torch.cuda.synchronize()
    records.append({'image':path,'scores':{n:s.tolist() for n,s in scores.items()},
                    'texture':x['t'].tolist(), 'selected':choices,
                    'shared_path_ms':(time.perf_counter()-start)*1000})
(OUT/'selections.json').write_text(json.dumps(records,indent=2)+'\n')
done = {'status':'PASS','selection_sha256':hashlib.sha256((OUT/'selections.json').read_bytes()).hexdigest(),
        'images':len(records),'normal_stats_images':len(train),'normal_calibration_images':len(validation),
        'diagnostic_shared_path_median_ms':float(np.median([r['shared_path_ms'] for r in records])),
        'timing_note':'includes SHA verification and all selectors; not method-specific latency or end-to-end fine detection',
        'peak_allocated_mib':torch.cuda.max_memory_allocated(0)/1024**2,
        'completed_before_evaluation':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()}
(OUT/'selection_complete.json').write_text(json.dumps(done,indent=2)+'\n')
print(json.dumps(done,indent=2)); print('OUTPUT:',OUT)
