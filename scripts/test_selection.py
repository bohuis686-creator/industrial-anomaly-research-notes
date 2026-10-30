"""Small known-answer checks for selection and coverage semantics."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from selection_core import tiles,topk,selection_mask,fit_texture,correct
from evaluate_selection import measure

# Each block must map to the same row-major index at low and high resolution.
grid=np.repeat(np.repeat(np.arange(16).reshape(4,4),4,axis=0),4,axis=1)
assert np.array_equal(tiles(grid)[:,0],np.arange(16))
assert topk(np.arange(16),4)==[15,14,13,12]
assert topk(np.zeros(16),4)==[0,1,2,3]
assert topk((np.arange(16)-8)/3,4)==topk(np.arange(16),4)
assert selection_mask([0,1,2,3]).sum()==1024*1024//4
# A connected defect crosses the boundary of two tiles; selecting one covers half.
mask=np.zeros((16,16),bool); mask[1:3,3:5]=True
chosen=selection_mask([0],16)
r=measure(mask,chosen,chosen)
assert r['pixel_coverage']==0.5 and r['component_half']==1 and r['components']==1
assert measure(mask,selection_mask(list(range(16)),16),chosen)['pixel_coverage']==1
# Conditional background correction must remove a known background difference.
t=np.repeat(np.arange(4),160).astype(float)
r=np.repeat([1.,2.,3.,4.],160)+np.tile(np.linspace(-.1,.1,160),4)
f=fit_texture(r,t)
z=correct(np.array([1.,2.,3.,4.]),np.arange(4),f)
np.testing.assert_allclose(z,0,atol=1e-10)
assert all(not g['fallback'] for g in f['groups'])
print('PASS: tile coordinates, ties, affine ranking, exact K area, component coverage, conditional calibration')
