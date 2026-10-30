"""Execute frozen synthetic feasibility pilot; never opens real test labels."""
from collections import defaultdict
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
from torchvision.models import resnet50, ResNet50_Weights

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from compare_resolution import auc
from smoke_native_tiles import stitch
from patchcore.patchcore import PatchCore
from patchcore.common import FaissNN
from selection_core import tiles, tail_mean, image_cues, selection_mask, topk, normalize


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def dump(path,obj):
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def features(image, coarse):
    low,residual_tail,texture,_=image_cues(image)
    restored=low.resize((1024,1024),Image.Resampling.BILINEAR)
    residual=np.abs(np.asarray(image,dtype=np.float32)/255-np.asarray(restored,dtype=np.float32)/255).mean(axis=2)
    c=tiles(coarse); r=tiles(residual); g=tiles(np.asarray(image.convert('L'),np.float32)/255)
    return np.stack([c.mean(1),c.std(1),c.max(1),np.quantile(c,.95,axis=1),tail_mean(coarse),
                     r.mean(1),np.quantile(r,.95,axis=1),residual_tail,texture,g.mean(1),g.std(1),
                     (np.arange(16)//4+.5)/4,(np.arange(16)%4+.5)/4],axis=1)


def errors(pred,mask):
    return np.stack([tiles(mask & ~pred).sum(1),tiles(~mask & pred).sum(1)],axis=1)


def targets(low,high,mask,thresholds):
    counts=np.array([int(mask.sum()),int((~mask).sum())])
    denom=2*np.maximum(counts,1)
    lc=errors(low>thresholds['low'],mask); hc=errors(high>thresholds['native'],mask)
    components=(lc-hc)/denom
    y=np.stack([tiles(mask).mean(1),components[:,0],components.sum(1)],axis=1)
    return y,lc,hc,counts


def ap(mask,scores):
    y=np.asarray(mask,bool).ravel(); s=np.asarray(scores).ravel()
    assert y.any()
    order=np.argsort(-s,kind='stable'); y=y[order]; s=s[order]
    ends=np.r_[np.flatnonzero(np.diff(s)!=0),len(s)-1]
    tp=np.cumsum(y,dtype=np.int64)[ends]
    return float(np.sum(np.diff(np.r_[0,tp])*(tp/(ends+1)))/y.sum())


def self_test():
    assert ap([1,0],[1,0])==1 and ap([1,0],[0,1])==.5 and ap([1,0],[1,1])==.5
    mask=np.zeros((4,4),bool); mask[0,0]=True
    low=np.zeros((4,4)); high=low.copy(); high[0,0]=2; high[3,3]=2
    y,lc,hc,counts=targets(low,high,mask,{'low':1,'native':1})
    assert y[0,2]==.5 and y[15,2]==-1/30
    for ids in [[],[0],[15],[0,15],list(range(16))]:
        mixed=lc.copy(); mixed[ids]=hc[ids]
        actual=(lc.sum(0)/(2*np.maximum(counts,1))).sum()-(mixed.sum(0)/(2*np.maximum(counts,1))).sum()
        assert abs(actual-y[ids,2].sum())<1e-12
    assert np.isfinite(targets(low,high,np.zeros_like(mask),{'low':1,'native':1})[0]).all()


def main():
    self_test()
    source=ROOT/'results/utility_pilot_protocol_v1'; data=ROOT/'data/utility_synthetic_v1'
    protocol=json.loads((source/'protocol.json').read_text())
    assert sha(source/'protocol.json')==json.loads((source/'checks.json').read_text())['protocol_sha256']
    assert sha(data/'manifest.json')==json.loads((data/'checks.json').read_text())['manifest_sha256']
    rows=json.loads((data/'manifest.json').read_text())
    assert len(rows)==192
    for split,group in [('train','pilot_train'),('validation','pilot_validation')]:
        assert {r['source'] for r in rows if r['split']==split}==set(protocol['groups'][group])
    assert not any(r['family']=='blur' for r in rows if r['split']=='train')
    old=ROOT/'results/sparse_refinement_20261030_105631'
    assert sha(old/'calibration.json')==json.loads((old/'predictions_complete.json').read_text())['calibration_sha256']
    scales=json.loads((old/'calibration.json').read_text())
    out=ROOT/'results'/datetime.now(ZoneInfo('Asia/Shanghai')).strftime('utility_pilot_%Y%m%d_%H%M%S')
    out.mkdir(exist_ok=False); (out/'maps').mkdir()
    dump(out/'execution_protocol.json',{'protocol_sha256':sha(source/'protocol.json'),'data_sha256':sha(data/'manifest.json'),
         'script_sha256':sha(Path(__file__)),'clarifications':{
         'image_false_alarm':'per-selector 99th percentile of full-map maxima on 53 normal calibration images, linear quantile, strict >; fitted after ridge weights fixed, before validation masks',
         'normal_FNR':'undefined, report null; contribution to primary loss is zero',
         'random':'sha256(seed:original-source:family:perturbation-seed), first8 little, default_rng permutation16',
         'cache':'offline fine tile predictions cached by exact RGB tile SHA; same frozen deterministic model; not runtime evidence',
         'bootstrap':'source-level mean over perturbation seeds, 8 validation sources',
         'timing':'no deployable runtime measured in this pilot'},'known_answer_tests':'PASS'})
    print('OUTPUT:',out,flush=True)
    torch.set_num_threads(2); torch.manual_seed(1592); np.random.seed(1592)
    torch.hub.set_dir(str(ROOT/'cache/torch'))
    device=torch.device('cuda:0'); models={}
    paths={'low':ROOT/'results/grid_smoke_20261030_014348/model',
           'native':ROOT/'results/resolution_comparison_20261030_104915/native/model'}
    for name,path in paths.items():
        assert sha(path/'nnscorer_search_index.faiss')==protocol['model_hashes'][name]
        backbone=resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).eval(); backbone.requires_grad_(False)
        model=PatchCore(device)
        model.load(backbone,['layer2','layer3'],device,(3,256,256),1024,1024,nn_method=FaissNN(False,2))
        model.anomaly_scorer.load(str(path)); models[name]=model
    mean=torch.tensor([.485,.456,.406])[:,None,None]; std=torch.tensor([.229,.224,.225])[:,None,None]
    cache={}; calls={'native':0,'cache_hits':0}

    def predict(image,name):
        arr=np.array(image,copy=True)
        key=hashlib.sha256(arr.tobytes()).hexdigest()
        if name=='native' and key in cache:
            calls['cache_hits']+=1
            return cache[key]
        x=((torch.from_numpy(arr).permute(2,0,1).float()/255-mean)/std).unsqueeze(0)
        amap=np.asarray(models[name].predict(x)[1][0],np.float32)
        assert amap.shape==(256,256) and np.isfinite(amap).all()
        if name=='native':
            calls['native']+=1; cache[key]=amap
        return amap

    def process(image):
        coarse=predict(image.resize((256,256),Image.Resampling.BILINEAR),'low')
        high=stitch([predict(image.crop((c*256,r*256,(c+1)*256,(r+1)*256)),'native') for r in range(4) for c in range(4)])
        low=np.asarray(Image.fromarray(coarse).resize((1024,1024),Image.Resampling.BILINEAR))
        return low,high,features(image,coarse)

    samples={'low':[],'native':[]}; calibration_records=[]
    for i,path in enumerate(protocol['groups']['calibration']):
        p=ROOT/'data/mvtec_grid_mirror'/path
        assert sha(p)==protocol['selected_source_hashes'][path]
        with Image.open(p) as im:
            low,high,x=process(im.convert('RGB'))
        key=f'cal_{i:03d}'
        np.savez(out/'maps'/(key+'.npz'),low=low,high=high)
        calibration_records.append({'key':key,'source':path,'x':x.tolist()})
        for name,amap in [('low',low),('native',high)]:
            samples[name].append(amap[8::16,8::16].ravel())
        cache.clear()
        if (i+1)%10==0: print('normal thresholds',i+1,'/53',flush=True)
    thresholds={k:float(np.quantile(np.concatenate(v),.99)) for k,v in samples.items()}
    dump(out/'thresholds.json',thresholds)
    dump(out/'calibration_features.json',calibration_records)
    records=[]; previous_source=None
    for i,row in enumerate(rows):
        if row['source']!=previous_source:
            cache.clear(); previous_source=row['source']
        assert sha(data/row['image'])==row['image_sha256']
        with Image.open(data/row['image']) as im:
            low,high,x=process(im.convert('RGB'))
        key=f'syn_{i:03d}'; p=out/'maps'/(key+'.npz')
        np.savez(p,low=low,high=high)
        record={**row,'key':key,'x':x.tolist(),'map_sha256':sha(p)}
        # Only training synthetic masks are read before fitting the regressors.
        if row['split']=='train':
            assert sha(data/row['mask'])==row['mask_sha256']
            with Image.open(data/row['mask']) as im:
                mask=np.asarray(im)>0
            y,_,_,_=targets(low,high,mask,thresholds); record['y']=y.tolist()
        records.append(record)
        if (i+1)%24==0: print('synthetic predictions',i+1,'/192',flush=True)
    dump(out/'features_and_training_targets.json',records)
    train=[r for r in records if r['split']=='train']
    x=np.concatenate([r['x'] for r in train]); y=np.concatenate([r['y'] for r in train])
    assert x.shape==(1792,13) and y.shape==(1792,3)
    center=x.mean(0); scale=np.maximum(x.std(0),1e-6); z=(x-center)/scale
    intercept=y.mean(0)
    slopes=np.linalg.solve(z.T@z/len(z)+np.eye(13),z.T@(y-intercept)/len(z))
    assert np.isfinite(slopes).all()
    dump(out/'ridge.json',{'center':center.tolist(),'scale':scale.tolist(),'slopes':slopes.tolist(),'intercept':intercept.tolist(),
                          'targets':['area','miss_correction','signed_gain'],'train_rows':len(x),'regularization':1.0})
    dump(out/'train_gain_distribution.json',{'negative':int((y[:,2]<0).sum()),'zero':int((y[:,2]==0).sum()),
           'positive':int((y[:,2]>0).sum()),'min':float(y[:,2].min()),'max':float(y[:,2].max())})

    def selections(record):
        x=np.asarray(record['x']); prediction=(x-center)/scale@slopes+intercept
        selected={f'ridge_{name}':topk(prediction[:,i],4) for i,name in enumerate(['area','miss_correction','signed_gain'])}
        selected.update(coarse_top1pct=topk(x[:,4],4),residual_top1pct=topk(x[:,7],4),low_only=[],all_tiles=list(range(16)))
        for seed in [1592,1593,1594,1595,1596]:
            key=f"{seed}:{record['source']}:{record.get('family','calibration')}:{record.get('seed')}"
            rng=np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'little'))
            selected[f'random_{seed}']=rng.permutation(16)[:4].tolist()
        return selected

    def merged(low,high,ids):
        return np.where(selection_mask(ids),normalize(high,scales['native']),normalize(low,scales['low']))

    maximums=defaultdict(list)
    for record in calibration_records:
        arrays=np.load(out/'maps'/(record['key']+'.npz'))
        for name,ids in selections(record).items():
            maximums[name].append(float(merged(arrays['low'],arrays['high'],ids).max()))
    image_thresholds={k:float(np.quantile(v,.99)) for k,v in maximums.items()}
    dump(out/'image_thresholds.json',image_thresholds)
    validation=[r for r in records if r['split']=='validation']
    frozen_selections=[{'key':r['key'],'selected':selections(r)} for r in validation]
    dump(out/'validation_selections.json',frozen_selections)
    dump(out/'prediction_complete.json',{'ridge_sha256':sha(out/'ridge.json'),'selections_sha256':sha(out/'validation_selections.json'),
         'thresholds_sha256':sha(out/'thresholds.json'),'image_thresholds_sha256':sha(out/'image_thresholds.json'),
         'cache_counts':calls,'validation_masks_read_before_freeze':0,'real_test_masks_read':0})
    print('Ridge models and validation selections frozen; starting synthetic evaluation',flush=True)
    details=[]
    for i,(record,sel) in enumerate(zip(validation,frozen_selections)):
        path=out/'maps'/(record['key']+'.npz'); assert sha(path)==record['map_sha256']
        arrays=np.load(path); low,high=arrays['low'],arrays['high']
        assert sha(data/record['mask'])==record['mask_sha256']
        with Image.open(data/record['mask']) as im:
            mask=np.asarray(im)>0
        yy,lc,hc,counts=targets(low,high,mask,thresholds)
        selected={**sel['selected'],'label_oracle_diagnostic_only':topk(yy[:,2],4)}
        metrics={}
        for name,ids in selected.items():
            mixed=lc.copy(); mixed[ids]=hc[ids]
            total=mixed.sum(0); denom=np.maximum(counts,1)
            loss=float((total/(2*denom)).sum())
            assert abs(float((lc.sum(0)/(2*denom)).sum())-loss-yy[ids,2].sum())<1e-12
            # Independent pixel replacement check, not only tile-sum reconstruction.
            binary=np.where(selection_mask(ids),high>thresholds['native'],low>thresholds['low'])
            np.testing.assert_array_equal(total,[int((mask&~binary).sum()),int((~mask&binary).sum())])
            amap=merged(low,high,ids)
            metrics[name]={'loss':loss,'fnr':float(total[0]/counts[0]) if counts[0] else None,
                           'fpr':float(total[1]/counts[1]),'ap':ap(mask,amap) if counts[0] else None,
                           'pixel_auc':auc(mask,amap) if counts[0] else None,'image_score':float(amap.max()),
                           'false_alarm':float(amap.max()>image_thresholds[name]) if not counts[0] and name in image_thresholds else None}
        details.append({'source':record['source'],'family':record['family'],'seed':record['seed'],'metrics':metrics})
        if (i+1)%10==0: print('synthetic evaluation',i+1,'/80',flush=True)
    dump(out/'evaluation_details.json',details)
    names=list(details[0]['metrics']); summaries={}
    for family in ['clean','line','patch_shift','blur']:
        summaries[family]={}
        for name in names:
            metrics={}
            for metric in ['loss','fnr','fpr','ap','pixel_auc','false_alarm']:
                bysource=defaultdict(list)
                for r in details:
                    v=r['metrics'][name][metric]
                    if r['family']==family and v is not None: bysource[r['source']].append(v)
                metrics[metric]=float(np.mean([np.mean(v) for v in bysource.values()])) if bysource else None
            summaries[family][name]=metrics
    sources=protocol['groups']['pilot_validation']; rng=np.random.default_rng(4312)
    indices=rng.integers(0,len(sources),(2000,len(sources))); comparisons={}
    for baseline in ['ridge_area','coarse_top1pct']:
        delta=np.array([np.mean([r['metrics']['ridge_signed_gain']['loss']-r['metrics'][baseline]['loss'] for r in details if r['source']==s and r['family']=='blur']) for s in sources])
        ci=np.quantile(delta[indices].mean(1),[.025,.975])
        comparisons[baseline]={'mean_loss_delta':float(delta.mean()),'source_bootstrap_95':ci.tolist()}
    normal_ok=summaries['clean']['ridge_signed_gain']['fpr']<=summaries['clean']['coarse_top1pct']['fpr']
    passed=normal_ok and all(r['mean_loss_delta']<0 and r['source_bootstrap_95'][1]<0 for r in comparisons.values())
    report={'execution_status':'PASS','feasibility_gate':'PASS' if passed else 'NOT_PASSED',
            'heldout_blur_comparisons':comparisons,'normal_fpr_condition':normal_ok,
            'summary':summaries,'model_fits':3,'real_test_images_or_masks_read':0,
            'deployment_timing':'NOT_RUN; inference used offline exact tile cache; no speed claim',
            'limitations':'8 validation sources; synthetic only; known grid development; no evidence of real generalization or novelty'}
    dump(out/'report.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='summary'},indent=2),flush=True)


if __name__=='__main__':
    main()
