"""Generate frozen pilot copies and exact changed-pixel masks, not model results."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0,str(Path(__file__).resolve().parent))
from utility_synthesis import apply, parameters
ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source=ROOT/'results/utility_pilot_protocol_v1'
    protocol=json.loads((source/'protocol.json').read_text())
    checks=json.loads((source/'checks.json').read_text())
    assert sha(source/'protocol.json')==checks['protocol_sha256']
    out=ROOT/'data/utility_synthetic_v1'
    out.mkdir(exist_ok=False)
    # Implementation details fixed before any actual synthesis; original protocol preserved.
    implementation={'protocol_sha256':sha(source/'protocol.json'),
                    'scripts_sha256':{p:sha(ROOT/'scripts'/p) for p in ['utility_synthesis.py','test_utility_synthesis.py','prepare_utility_synthetic.py']},
                    'clarifications':'line x,y are top-left; boxes half-open; discrete choices uniform; draw order fixed by script; retain original source copy; empty mask fails without retry',
                    'purpose':'synthetic feasibility data, NOT real industrial defect labels',
                    'license':'Derived from MVTec AD mirror; CC BY-NC-SA 4.0; original license retained in mvtec_grid_mirror'}
    (out/'generation_protocol.json').write_text(json.dumps(implementation,indent=2)+'\n')
    rows=[]
    for split,group in [('train','pilot_train'),('validation','pilot_validation')]:
        (out/split).mkdir()
        families=protocol['synthesis']['train_families' if split=='train' else 'validation_families']
        for index,path in enumerate(protocol['groups'][group]):
            filepath=ROOT/'data/mvtec_grid_mirror'/path
            assert sha(filepath)==protocol['selected_source_hashes'][path]
            with Image.open(filepath) as im:
                original=np.array(im.convert('RGB'),copy=True)
            variants=[('clean',None)]+[(f,s) for f in families for s in protocol['synthesis']['seeds']]
            for family,seed in variants:
                params=parameters(path,family,seed)
                result,mask=apply(original,params)
                repeated,repeated_mask=apply(original,parameters(path,family,seed))
                np.testing.assert_array_equal(result,repeated)
                np.testing.assert_array_equal(mask,repeated_mask)
                np.testing.assert_array_equal(mask,np.any(result!=original,axis=2))
                if family!='clean':
                    allowed=np.zeros(mask.shape,bool)
                    x1,y1,x2,y2=params['box']; allowed[y1:y2,x1:x2]=True
                    assert not (mask & ~allowed).any()
                prefix=f'{index:03d}_{family}_{seed if seed is not None else "original"}'
                ip=out/split/(prefix+'.png'); mp=out/split/(prefix+'_mask.png')
                Image.fromarray(result).save(ip); Image.fromarray(mask.astype(np.uint8)*255).save(mp)
                with Image.open(ip) as im:
                    np.testing.assert_array_equal(np.asarray(im),result)
                with Image.open(mp) as im:
                    np.testing.assert_array_equal(np.asarray(im)>0,mask)
                rows.append({'split':split,'source':path,'source_sha256':sha(filepath),
                             'family':family,'seed':seed,'parameters':params,'changed_pixels':int(mask.sum()),
                             'image':str(ip.relative_to(out)),'mask':str(mp.relative_to(out)),
                             'image_sha256':sha(ip),'mask_sha256':sha(mp)})
        print(split,'generated',sum(r['split']==split for r in rows),flush=True)
    assert len(rows)==192
    assert not {r['source_sha256'] for r in rows if r['split']=='train'} & {r['source_sha256'] for r in rows if r['split']=='validation'}
    assert not any(r['family']=='blur' for r in rows if r['split']=='train')
    (out/'manifest.json').write_text(json.dumps(rows,indent=2)+'\n')
    # Deterministic first validation source, first seed, each family. Crops are visual checks only.
    first=protocol['groups']['pilot_validation'][0]
    canvas=Image.new('RGB',(768,4*280),'white'); draw=ImageDraw.Draw(canvas)
    for j,family in enumerate(['clean','line','patch_shift','blur']):
        row=next(r for r in rows if r['source']==first and r['family']==family)
        with Image.open(ROOT/'data/mvtec_grid_mirror'/first) as im:
            original=im.convert('RGB')
        box=row['parameters'].get('box',[384,384,640,640])
        cx=(box[0]+box[2])//2; cy=(box[1]+box[3])//2
        x=max(0,min(768,cx-128)); y=max(0,min(768,cy-128)); crop=(x,y,x+256,y+256)
        canvas.paste(original.crop(crop),(0,j*280+24))
        for k,key in [(1,'image'),(2,'mask')]:
            with Image.open(out/row[key]) as im:
                canvas.paste(im.convert('RGB').crop(crop),(k*256,j*280+24))
        draw.text((5,j*280+5),family+' | original / synthetic / exact-change mask',fill='black')
    canvas.save(out/'preview.png')
    report={'status':'PASS_DATA_CHECKS_ONLY','counts':dict(Counter(r['split']+':'+r['family'] for r in rows)),
            'images':len(rows),'masks':len(rows),'bytes':sum(p.stat().st_size for p in out.rglob('*') if p.is_file()),
            'manifest_sha256':sha(out/'manifest.json'),'checks':['determinism','exact changed-pixel mask','no edits outside box','lossless PNG roundtrip','source split disjoint','blur validation only'],
            'model_fits':0,'real_defect_masks_read':0}
    (out/'checks.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2)); print('OUTPUT:',out)


if __name__=='__main__':
    main()
