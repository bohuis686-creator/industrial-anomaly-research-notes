"""Label-free tile scores, calibration and deterministic selection."""
import numpy as np
from PIL import Image


def tiles(a):
    assert a.ndim == 2 and a.shape[0] == a.shape[1] and a.shape[0] % 4 == 0
    side = a.shape[0] // 4
    return a.reshape(4, side, 4, side).transpose(0, 2, 1, 3).reshape(16, -1)


def tail_mean(a):
    values = tiles(a)
    n = max(1, int(np.ceil(values.shape[1] * 0.01)))
    return np.partition(values, -n, axis=1)[:, -n:].mean(axis=1)


def image_cues(image):
    assert image.size == (1024, 1024), 'Protocol only covers 1024 square images'
    low = image.resize((256, 256), Image.Resampling.BILINEAR)
    restored = low.resize((1024, 1024), Image.Resampling.BILINEAR)
    original = np.asarray(image, dtype=np.float32) / 255
    residual = np.abs(original - np.asarray(restored, dtype=np.float32)/255).mean(axis=2)
    # Low-resolution texture descriptor, with gradients computed inside each tile.
    gray_low = np.asarray(low.convert('L'), dtype=np.float32)/255
    patches = tiles(gray_low).reshape(16, 64, 64)
    texture = (np.abs(np.diff(patches, axis=1)).mean(axis=(1, 2)) +
               np.abs(np.diff(patches, axis=2)).mean(axis=(1, 2))) / 2
    gray = np.asarray(image.convert('L'), dtype=np.float32)/255
    # Independent inexpensive gradient-energy baseline.
    dx = np.diff(gray, axis=1, append=gray[:, -1:])
    dy = np.diff(gray, axis=0, append=gray[-1:, :])
    gradient = tail_mean(np.sqrt(dx*dx + dy*dy))
    return low, tail_mean(residual), texture, gradient


def robust_location_scale(x):
    median = float(np.median(x))
    scale = max(float(1.4826*np.median(np.abs(x-median))), 1e-6)
    return median, scale


def fit_texture(residual, texture):
    r, t = np.asarray(residual).ravel(), np.asarray(texture).ravel()
    boundaries = np.quantile(t, [0.25, 0.5, 0.75])
    bins = np.searchsorted(boundaries, t, side='right')
    global_stats = robust_location_scale(r)
    groups = []
    for b in range(4):
        x = r[bins == b]
        loc, scale = robust_location_scale(x) if len(x) >= 128 else global_stats
        groups.append({'median': loc, 'scale': scale, 'count': len(x), 'fallback': len(x) < 128})
    return {'boundaries': boundaries.tolist(), 'groups': groups}


def correct(residual, texture, fitted):
    bins = np.searchsorted(fitted['boundaries'], texture, side='right')
    loc = np.array([fitted['groups'][int(b)]['median'] for b in bins])
    scale = np.array([fitted['groups'][int(b)]['scale'] for b in bins])
    return (np.asarray(residual)-loc)/scale


def fit_scale(x):
    q50, q95 = np.quantile(np.asarray(x).ravel(), [0.5, 0.95])
    return {'center': float(q50), 'scale': max(float(q95-q50), 1e-6)}


def normalize(x, fitted):
    # No clipping: preserve discrimination beyond normal calibration range.
    return (np.asarray(x)-fitted['center'])/fitted['scale']


def topk(scores, k):
    scores = np.asarray(scores)
    assert scores.shape == (16,) and np.isfinite(scores).all() and 1 <= k <= 16
    return np.argsort(-scores, kind='stable')[:k].tolist()


def selection_mask(indices, side=1024):
    assert len(indices) == len(set(indices)) and all(0 <= x < 16 for x in indices)
    out = np.zeros((side, side), dtype=bool)
    width = side//4
    for index in indices:
        row, col = divmod(index, 4)
        out[row*width:(row+1)*width, col*width:(col+1)*width] = True
    return out
