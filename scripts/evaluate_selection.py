"""Read ground truth only after frozen selections exist; no tuning here."""
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from selection_core import selection_mask


def measure(mask, selected, coarse):
    total = int(mask.sum())
    assert total > 0
    labels,n = ndimage.label(mask, structure=np.ones((3,3)))
    areas = np.bincount(labels.ravel())[1:]
    overlaps = np.bincount(labels[selected], minlength=n+1)[1:]
    fractions = overlaps/areas
    return {'pixel_coverage':float((mask & selected).sum()/total),
            'covered_pixels':int((mask & selected).sum()),'defect_pixels':total,
            'component_any':int((fractions>0).sum()),
            'component_half':int((fractions>=0.5).sum()), 'components':int(n),
            'component_fraction_sum':float(fractions.sum()),
            'gained_fraction_vs_coarse':float((mask & selected & ~coarse).sum()/total),
            'lost_fraction_vs_coarse':float((mask & ~selected & coarse).sum()/total)}


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('run',type=Path); args=parser.parse_args()
    out=args.run.resolve(); data=ROOT/'data/mvtec_grid_mirror'
    frozen=json.loads((out/'selection_complete.json').read_text())
    assert hashlib.sha256((out/'selections.json').read_bytes()).hexdigest()==frozen['selection_sha256']
    protocol=json.loads((out/'protocol.json').read_text())
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==protocol['scripts_sha256'][Path(__file__).name]
    manifest_raw=(data/'manifest.json').read_bytes()
    assert hashlib.sha256(manifest_raw).hexdigest()==protocol['manifest_sha256']
    manifest=json.loads(manifest_raw); records=json.loads((out/'selections.json').read_text())
    assert [r['image'] for r in records]==[r['image'] for r in manifest['test']]
    hashes={f['path']:f['sha256'] for f in manifest['files']}
    # Lock the prediction artifact before any ground truth is loaded.
    targets=[]
    for r in manifest['test']:
        if r['mask']:
            assert hashlib.sha256((data/r['mask']).read_bytes()).hexdigest()==hashes[r['mask']]
            with Image.open(data/r['mask']) as im:
                mask=np.asarray(im.convert('L'))>0
            assert mask.shape==(1024,1024) and mask.any()
        else:
            mask=np.zeros((1024,1024),dtype=bool)
        assert bool(mask.any())==(r['defect']!='good')
        targets.append(mask)
    abnormal=[i for i,m in enumerate(targets) if m.any()]
    details={}; summaries={}; comparisons={}
    for k in ['4','8']:
        names=list(records[0]['selected'][k])
        details[k]={name:[] for name in names}
        for i in abnormal:
            base=selection_mask(records[i]['selected'][k]['coarse'])
            for name in names:
                ids=records[i]['selected'][k][name]
                assert len(ids)==(16 if name=='all_tiles' else int(k))
                result=measure(targets[i],selection_mask(ids),base)
                result.update(image=records[i]['image'],defect=manifest['test'][i]['defect'])
                details[k][name].append(result)
        summaries[k]={}
        for name, rows in details[k].items():
            total_components=sum(r['components'] for r in rows)
            summaries[k][name]={
                'pixel_coverage_macro':float(np.mean([r['pixel_coverage'] for r in rows])),
                'pixel_coverage_micro':sum(r['covered_pixels'] for r in rows)/sum(r['defect_pixels'] for r in rows),
                'component_any_recall':sum(r['component_any'] for r in rows)/total_components,
                'component_half_recall':sum(r['component_half'] for r in rows)/total_components,
                'component_mean_fraction':sum(r['component_fraction_sum'] for r in rows)/total_components,
                'gain_vs_coarse':float(np.mean([r['gained_fraction_vs_coarse'] for r in rows])),
                'loss_vs_coarse':float(np.mean([r['lost_fraction_vs_coarse'] for r in rows])),
                'by_defect_macro':{d:float(np.mean([r['pixel_coverage'] for r in rows if r['defect']==d]))
                                   for d in sorted({r['defect'] for r in rows})}}
        random_names=[n for n in names if n.startswith('random_')]
        summaries[k]['random_mean']={key:float(np.mean([summaries[k][n][key] for n in random_names]))
                                     for key in ['pixel_coverage_macro','pixel_coverage_micro','component_any_recall','component_half_recall']}
        summaries[k]['random_mean']['macro_sd_across_5_seeds']=float(np.std([summaries[k][n]['pixel_coverage_macro'] for n in random_names],ddof=1))
        comparisons[k]={}
        for baseline in ['coarse','fusion_global']:
            delta=np.array([r['pixel_coverage'] for r in details[k]['fusion_texture']])-np.array([r['pixel_coverage'] for r in details[k][baseline]])
            indices=np.random.default_rng(4312).integers(0,len(delta),(2000,len(delta)))
            lo,hi=np.quantile(delta[indices].mean(axis=1),[0.025,0.975])
            comparisons[k]['fusion_texture_minus_'+baseline]={
                'mean_delta':float(delta.mean()),'paired_image_bootstrap_95pct':[float(lo),float(hi)],
                'improved_images':int((delta>1e-10).sum()),'worsened_images':int((delta < -1e-10).sum()),
                'unchanged_images':int((np.abs(delta)<=1e-10).sum()),
                'different_selection_sets_all_images':sum(set(r['selected'][k]['fusion_texture'])!=set(r['selected'][k][baseline]) for r in records)}
    assert all(abs(summaries[k]['all_tiles']['pixel_coverage_macro']-1)<1e-12 for k in ['4','8'])
    report={'status':'PASS','evaluation_time':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
            'selection_sha256':frozen['selection_sha256'],'abnormal_images':len(abnormal),
            'normal_images_excluded_from_coverage':len(records)-len(abnormal),
            'components':sum(r['components'] for r in details['4']['coarse']),
            'main_k':4,'summaries':summaries,'paired_comparisons':comparisons,
            'limitations':'grid development only; 20-image coarse model; conditional texture v1; no fine detection; no full end-to-end timing; intervals exploratory'}
    (out/'coverage_details.json').write_text(json.dumps(details,indent=2)+'\n')
    (out/'coverage_report.json').write_text(json.dumps(report,indent=2)+'\n')
    methods=['random_mean','coarse','residual','gradient','corrected_residual','fusion_global','fusion_texture','all_tiles']
    lines=['# Grid 区域选择开发实验','', '主预算 K=4；K=8 为预设补充。只测选区覆盖率，不是精检检测精度。', '',
           f'异常图 {len(abnormal)} 张；正常图 {len(records)-len(abnormal)} 张不进入覆盖率分母。', '']
    for k in ['4','8']:
        lines += [f'## K={k}', '', '| 方法 | 图像平均缺陷像素覆盖率 | 连通缺陷至少覆盖一半 |', '| --- | ---: | ---: |']
        for name in methods:
            s=summaries[k][name]
            lines.append(f"| {name} | {s['pixel_coverage_macro']:.2%} | {s['component_half_recall']:.2%} |")
        lines.append('')
        for name,s in comparisons[k].items():
            ci=s['paired_image_bootstrap_95pct']
            lines.append(f"{name}: 平均差 {s['mean_delta']*100:.2f} 个百分点；探索性配对 bootstrap 95% 区间 [{ci[0]*100:.2f}, {ci[1]*100:.2f}]。")
        lines.append('')
    lines += ['结果未用于重新选择参数。随机行是 5 个预设种子的平均；all_tiles 使用 16 块，不属于同预算竞争方案。', '',
              '峰值显存和 shared_path_ms 只覆盖选区准备路径，包含所有方案及文件校验；不能据此宣称端到端加速。']
    (out/'SUMMARY.md').write_text('\n'.join(lines)+'\n')
    # First anomalous image in the pre-existing file order; no best-case selection.
    i=abnormal[0]
    with Image.open(data/records[i]['image']) as im:
        original=im.convert('RGB').resize((512,512))
    mask=targets[i]; contour=mask & ~ndimage.binary_erosion(mask)
    # Max-pool the boundary to avoid erasing narrow defects when displaying.
    boundary=contour.reshape(512,2,512,2).any(axis=(1,3))
    array=np.array(original); array[boundary]=[255,0,0]
    original=Image.fromarray(array)
    preview_names=['coarse','residual','gradient','fusion_global','fusion_texture','random_1592']
    canvas=Image.new('RGB',(1536,1120),'white'); draw=ImageDraw.Draw(canvas)
    for j,name in enumerate(preview_names):
        x=(j%3)*512; y=(j//3)*548
        canvas.paste(original,(x,y+26))
        draw.text((x+5,y+5),name+' (K=4)',fill='black')
        for index in records[i]['selected']['4'][name]:
            row,col=divmod(index,4)
            draw.rectangle((x+col*128,y+26+row*128,x+(col+1)*128-1,y+26+(row+1)*128-1),outline='yellow',width=3)
    draw.text((5,1097),'Yellow=selected tiles; red=GT boundary (evaluation only). MVTec / CC BY-NC-SA 4.0; resized + overlay.',fill='black')
    canvas.save(out/'selection_preview.png')
    print('\n'.join(lines)); print('OUTPUT:',out)


if __name__=='__main__':
    main()
