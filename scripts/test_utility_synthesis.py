"""Known-answer tests, no project image access."""
from pathlib import Path
import sys
import unittest
import numpy as np
from PIL import Image, ImageFilter
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utility_synthesis import apply, parameters


class SynthesisTests(unittest.TestCase):
    def setUp(self):
        self.a = np.repeat((np.indices((1024,1024)).sum(axis=0)%256).astype(np.uint8)[:,:,None],3,axis=2)

    def test_identity(self):
        image, mask = apply(self.a, {'family':'clean'})
        np.testing.assert_array_equal(image,self.a)
        self.assertFalse(mask.any())

    def test_line_clip_and_exact_mask(self):
        original = np.full((1024,1024,3),250,np.uint8)
        image,mask = apply(original,{'family':'line','box':[10,20,42,22],'delta':32})
        self.assertEqual(int(mask.sum()),64)
        self.assertTrue((image[20:22,10:42]==255).all())
        np.testing.assert_array_equal(image[~mask],original[~mask])

    def test_roll(self):
        image,mask = apply(self.a,{'family':'patch_shift','box':[0,0,16,16],'shift':4})
        np.testing.assert_array_equal(image[:16,:16], np.roll(self.a[:16,:16],4,axis=1))
        self.assertEqual(int(mask.sum()),256)

    def test_blur_whole_image_then_crop(self):
        params={'family':'blur','box':[992,992,1024,1024],'radius':3}
        image,mask = apply(self.a,params)
        expected=np.asarray(Image.fromarray(self.a).filter(ImageFilter.GaussianBlur(3)))
        np.testing.assert_array_equal(image[992:,992:],expected[992:,992:])
        np.testing.assert_array_equal(mask,np.any(image!=self.a,axis=2))
        self.assertFalse(mask[:992].any())

    def test_reject_empty(self):
        with self.assertRaises(ValueError):
            apply(np.zeros_like(self.a),{'family':'patch_shift','box':[0,0,16,16],'shift':4})

    def test_seed_and_bounds(self):
        textured=np.random.default_rng(9001).integers(0,256,(1024,1024,3),dtype=np.uint8)
        for family in ['line','patch_shift','blur']:
            for seed in range(100):
                p=parameters('fixed.png',family,seed)
                self.assertEqual(p,parameters('fixed.png',family,seed))
                x1,y1,x2,y2=p['box']
                self.assertTrue(0<=x1<x2<=1024 and 0<=y1<y2<=1024)
            p=parameters('fixed.png',family,1592)
            a,m=apply(textured,p); b,n=apply(textured,p)
            np.testing.assert_array_equal(a,b); np.testing.assert_array_equal(m,n)


if __name__=='__main__':
    unittest.main()
