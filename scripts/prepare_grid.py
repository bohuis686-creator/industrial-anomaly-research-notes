"""Download only grid from the pinned Voxel51 mirror; preserve provenance."""
import concurrent.futures
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import random
import subprocess
import urllib.request
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'data/mvtec_grid_mirror'
REV = '30a183a3b96e3aef953f230784b123b719b09d97'
BASE = f'https://huggingface.co/datasets/Voxel51/mvtec-ad/resolve/{REV}/'

def fetch(path):
    parts = PurePosixPath(path)
    assert not parts.is_absolute() and '..' not in parts.parts
    target = DEST / path
    if target.exists():
        data = target.read_bytes()
    else:
        last = None
        for _ in range(3):
            try:
                data = subprocess.check_output([
                    'curl', '-fsSL', '--connect-timeout', '10', '--max-time', '40',
                    '--max-filesize', str(10 * 1024 * 1024), BASE + path])
                assert len(data) <= 10 * 1024 * 1024, 'unexpected file size'
                break
            except Exception as exc:
                last = exc
        else:
            raise last
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    info = {'path': path, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
    if path.endswith('.png'):
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            info['size'] = list(image.size)
    return info

DEST.mkdir(parents=True, exist_ok=True)
fetch('samples.json')
fetch('license.txt')
fetch('readme.txt')
samples = json.loads((DEST / 'samples.json').read_text())['samples']
samples = sorted((s for s in samples if s['category']['label'] == 'grid'), key=lambda s: s['filepath'])
records = [{'image': s['filepath'], 'split': s['split'], 'defect': s['defect']['label'],
            'mask': s.get('defect_mask', {}).get('mask_path')} for s in samples]
assert len(records) == 342
paths = sorted({p for s in records for p in (s['image'], s['mask']) if p})
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    files = list(pool.map(fetch, paths))
assert sum(f['bytes'] for f in files) < 500 * 1024**2
normal = [s['image'] for s in records if s['split'] == 'train']
assert len(normal) == 264 and all(s['defect'] == 'good' for s in records if s['split'] == 'train')
random.Random(1592).shuffle(normal)
manifest = {
    'source': 'Voxel51/mvtec-ad (converted mirror, not verified bitwise against official archive)',
    'revision': REV, 'license': 'CC-BY-NC-SA-4.0', 'seed': 1592,
    'normal_validation': sorted(normal[:53]), 'normal_training': sorted(normal[53:]),
    'smoke_training': normal[53:73],
    'test': [s for s in records if s['split'] == 'test'], 'files': files,
}
assert not set(manifest['normal_validation']) & set(manifest['normal_training'])
assert not set(normal) & {s['image'] for s in manifest['test']}
out = DEST / 'manifest.json'
if out.exists():
    assert json.loads(out.read_text()) == manifest, 'Existing manifest differs; do not replace'
else:
    out.write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps({'status': 'PASS', 'images': len(records), 'files': len(files),
                  'download_mib': sum(f['bytes'] for f in files)/1024**2,
                  'normal_training': 211, 'normal_validation': 53, 'test': len(manifest['test']),
                  'dimensions': sorted({tuple(f['size']) for f in files})}, indent=2))
