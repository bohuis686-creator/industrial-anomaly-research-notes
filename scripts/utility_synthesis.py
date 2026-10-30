"""Deterministic scientific perturbations; no learned generation or defect labels."""
import hashlib
import numpy as np
from PIL import Image, ImageFilter


def parameters(path, family, seed):
    key = f'{seed}:{path}:{family}'.encode('utf8')
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:8], 'little'))
    if family == 'clean':
        return {'family': family}
    if family == 'line':
        x, y = [int(v) for v in rng.integers(128, 896, size=2)]
        horizontal = bool(rng.integers(0, 2))
        length, width = int(rng.choice([32, 64, 128])), int(rng.choice([1, 2, 4]))
        delta = int(rng.choice([-1, 1])) * int(rng.choice([32, 64]))
        w, h = (length, width) if horizontal else (width, length)
        return {'family': family, 'box': [x,y,x+w,y+h], 'delta': delta,
                'horizontal': horizontal, 'length': length, 'width': width}
    if family not in ['patch_shift', 'blur']:
        raise ValueError(f'Unknown family {family}')
    side = int(rng.choice([16, 32, 64]))
    x, y = [int(v) for v in rng.integers(0, 1024-side+1, size=2)]
    result = {'family': family, 'box': [x,y,x+side,y+side]}
    result.update({'shift': max(1,side//4)} if family == 'patch_shift' else {'radius': 3})
    return result


def apply(original, params):
    original = np.asarray(original)
    assert original.shape == (1024,1024,3) and original.dtype == np.uint8
    result = original.copy()
    family = params['family']
    if family == 'clean':
        return result, np.zeros((1024,1024), bool)
    x1,y1,x2,y2 = params['box']
    assert 0 <= x1 < x2 <= 1024 and 0 <= y1 < y2 <= 1024
    if family == 'line':
        result[y1:y2,x1:x2] = np.clip(original[y1:y2,x1:x2].astype(np.int16)+params['delta'],0,255).astype(np.uint8)
    elif family == 'patch_shift':
        result[y1:y2,x1:x2] = np.roll(original[y1:y2,x1:x2], params['shift'], axis=1)
    elif family == 'blur':
        blurred = np.asarray(Image.fromarray(original).filter(ImageFilter.GaussianBlur(params['radius'])))
        result[y1:y2,x1:x2] = blurred[y1:y2,x1:x2]
    else:
        raise ValueError(f'Unknown family {family}')
    mask = np.any(result != original, axis=2)
    if not mask.any():
        raise ValueError('Empty perturbation mask; do not silently resample')
    return result, mask
