"""Freeze pilot metadata only: no image synthesis, inference, fitting or test labels."""
import hashlib
import itertools
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    path = ROOT / 'data/mvtec_grid_mirror/manifest.json'
    m = json.loads(path.read_text())
    bank = list(m['smoke_training'])
    calibration = list(m['normal_validation'])
    pool = sorted(set(m['normal_training'])-set(bank))
    random.Random(271828).shuffle(pool)
    groups = {'bank': bank, 'calibration': calibration,
              'pilot_train': pool[:16], 'pilot_validation': pool[16:24],
              'unused_normals': pool[24:]}
    hashes = {r['path']: r['sha256'] for r in m['files']}
    for a, b in itertools.combinations(groups, 2):
        assert not set(groups[a]) & set(groups[b])
        assert not {hashes[p] for p in groups[a]} & {hashes[p] for p in groups[b]}
    test_paths = {r['image'] for r in m['test']}
    for values in groups.values():
        assert not set(values) & test_paths
        assert not {hashes[p] for p in values} & {hashes[p] for p in test_paths}
    # Decode no images. Verify files assigned to the actual pilot by digest only.
    for group in ['bank', 'calibration', 'pilot_train', 'pilot_validation']:
        for p in groups[group]:
            assert sha(ROOT / 'data/mvtec_grid_mirror' / p) == hashes[p]
    banks = {'low': ROOT/'results/grid_smoke_20261030_014348/model/nnscorer_search_index.faiss',
             'native': ROOT/'results/resolution_comparison_20261030_104915/native/model/nnscorer_search_index.faiss'}
    protocol = {
        'version': 'utility_pilot_v1', 'status': 'PROTOCOL_ONLY_NOT_RUN',
        'question': 'Does signed correction gain predict useful refinement better than anomaly area with matched inputs and ridge model?',
        'scope': 'grid synthetic feasibility only; no real defect evaluation in this pilot',
        'manifest_sha256': sha(path), 'script_sha256': sha(Path(__file__)),
        'split_seed': 271828, 'groups': groups,
        'selected_source_hashes': {p: hashes[p] for k in ['bank','calibration','pilot_train','pilot_validation'] for p in groups[k]},
        'model_hashes': {k: sha(p) for k,p in banks.items()},
        'leakage_note': 'grid was already development; old texture calibration used all training normals. Do not use that fitted texture correction as input to new pilot. No independent real-test claim.',
        'geometry': {'image': [1024,1024], 'grid': [4,4], 'tile': 256, 'overlap': 0, 'k': 4},
        'calibration': {'source_group': 'calibration', 'sample_lattice': '[8::16,8::16]',
                        'threshold_quantile': .99, 'quantile_method': 'linear',
                        'positive_rule': 'raw map > detector-specific threshold',
                        'map_alignment': 'coarse float PIL bilinear to 1024; fine row-major stitch',
                        'meaning': 'empirical normal pixel reference, not guaranteed test FPR',
                        'continuous_score': 'reuse frozen normal q50/q95 map calibration from sparse_refinement_20261030_105631'},
        'synthesis': {'seeds': [1592,1593,1594],
                      'rng': 'default_rng(int.from_bytes(sha256(UTF8(seed:path:family))[:8], little))',
                      'train_families': ['line','patch_shift'],
                      'validation_families': ['line','patch_shift','blur'],
                      'clean': 'one unchanged copy per source, not duplicated per seed',
                      'line': 'x,y uniform integers 128..895; horizontal/vertical equiprobable; length in {32,64,128}, width in {1,2,4}; add signed 32 or 64 to all RGB channels in rectangle; clip uint8',
                      'patch_shift': 'square side in {16,32,64}; top-left uniform over legal image coordinates; cyclic horizontal roll of source square by max(1,side//4); no blending',
                      'blur': 'held out from fitting: square side in {16,32,64}; top-left uniform legal coordinates; replace square with corresponding crop of whole image PIL GaussianBlur radius=3',
                      'mask': 'actual changed pixels: any RGB channel differs; reject empty mask as generation failure, do not silently resample',
                      'artifacts': 'source identity, seed, parameters, image/mask SHA; no source image crosses splits'},
        'features': ['coarse_mean','coarse_std','coarse_max','coarse_q95','coarse_top1pct_mean',
                     'residual_mean','residual_q95','residual_top1pct_mean','lowres_texture',
                     'gray_mean','gray_std','row_center','col_center'],
        'feature_definitions': 'coarse per64x64 tile; residual RGB mean abs(original-bilinear up/down); texture selection_core image_cues; gray full-res PIL L /255; std ddof0; row/col centers (index+.5)/4; q95 linear; top1% ceil',
        'regression': {'model': 'numpy ridge, same 13 inputs and capacity for all three targets',
                       'objective': 'mean squared error + 1.0 * squared slope norm; unpenalized intercept',
                       'standardization': 'train rows only, std floor 1e-6',
                       'selection': 'exactly K=4 descending predicted score, stable row-major ties; retain negative scores'},
        'targets': {'area': 'defect pixels in tile / 65536',
                    'miss_correction': '(FN_coarse_in_tile-FN_fine_in_tile)/(2*max(P_image,1)); signed',
                    'signed_gain': 'miss_correction+(FP_coarse_in_tile-FP_fine_in_tile)/(2*max(N_image,1))'},
        'loss': 'FN/(2*max(P,1))+FP/(2*max(N,1)); fixed thresholds, disjoint replacement',
        'comparators': ['ridge_area','ridge_miss_correction','ridge_signed_gain','coarse_top1pct','residual_top1pct','random_1592_to_1596','low_only','all_tiles','label_oracle_diagnostic_only'],
        'oracle': 'top4 true signed gains; nondeployable diagnostic, never a model input or performance claim',
        'evaluation': 'aggregate repeats by source image first; show clean, line, patch_shift, held-out blur separately; primary loss, FNR and FPR; continuous per-image AP/AUROC on anomalous copies; normal image max-score false-alarm rate using normal-only calibration; no true test masks',
        'decision': 'proceed only if held-out blur source-mean loss for signed gain beats area and coarse, paired source-bootstrap 95% upper delta <0 for both, and clean FPR <= coarse; otherwise not passed, not proof of universal failure',
        'bootstrap': {'resamples': 2000, 'seed': 4312, 'unit': '8 validation source images, not tiles/pixels'},
        'bounds': 'single frozen ridge setting, no hyperparameter sweep; no new backbone/dependencies; stop on integrity error; no true defect evaluation; feasibility passing does not establish novelty/real generalization',
        'future_confirmation': 'AUPRO/AP, real unseen categories, matched real runtime and multiple bank seeds required later; not run in pilot',
    }
    out = ROOT / 'results/utility_pilot_protocol_v1'
    out.mkdir(exist_ok=False)
    target = out / 'protocol.json'
    target.write_text(json.dumps(protocol, indent=2, ensure_ascii=False)+'\n')
    check = {'status': 'PASS_PROTOCOL_CHECKS_ONLY', 'counts': {k:len(v) for k,v in groups.items()},
             'path_and_content_disjoint': True, 'selected_files_hash_valid': True,
             'protocol_sha256': sha(target), 'generated_images': 0, 'model_fits': 0,
             'real_test_masks_read': 0}
    (out / 'checks.json').write_text(json.dumps(check, indent=2)+'\n')
    print(json.dumps(check, indent=2)); print('OUTPUT:', out)


if __name__ == '__main__':
    main()
