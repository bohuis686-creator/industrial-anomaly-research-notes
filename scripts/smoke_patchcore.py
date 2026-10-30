"""Run the pinned author core on synthetic tensors; not a detection benchmark."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "vendor/patchcore-inspection"
COMMIT = "fcaa92f124fb1ad74a7acf56726decd4b27cbcad"
assert subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip() == COMMIT
sys.path.insert(0, str(REPO / "src"))

import numpy as np
import torch
from torchvision.models import resnet50, ResNet50_Weights
from patchcore.patchcore import PatchCore
from patchcore.common import FaissNN
from patchcore.sampler import ApproximateGreedyCoresetSampler

torch.manual_seed(1592)
np.random.seed(1592)
torch.set_num_threads(2)
assert torch.cuda.is_available()
torch.cuda.init()
torch.hub.set_dir(str(ROOT / "cache/torch"))
weights = ResNet50_Weights.IMAGENET1K_V1
backbone = resnet50(weights=weights).eval()
backbone.requires_grad_(False)
device = torch.device("cuda:0")
model = PatchCore(device)
model.load(
    backbone=backbone, layers_to_extract_from=["layer2", "layer3"],
    device=device, input_shape=(3, 256, 256), pretrain_embed_dimension=1024,
    target_embed_dimension=1024, patchsize=3, anomaly_score_num_nn=1,
    featuresampler=ApproximateGreedyCoresetSampler(0.1, device),
    nn_method=FaissNN(False, 2),
)
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
images = (torch.rand(2, 3, 256, 256) - mean) / std
torch.cuda.reset_peak_memory_stats(0)
model.fit([images[i:i+1] for i in range(2)])
scores, maps = model.predict(images[:1])
torch.cuda.synchronize(0)
maps = np.asarray(maps)
assert maps.shape == (1, 256, 256), maps.shape
assert np.isfinite(maps).all() and np.isfinite(scores).all()
# Independent nearest-neighbour sanity check: FAISS uses squared L2 distance.
nn = FaissNN(False, 2)
bank = np.array([[0, 0], [2, 0], [0, 3]], dtype=np.float32)
query = np.array([[0.1, 0], [1.9, 0]], dtype=np.float32)
nn.fit(bank)
distances, indices = nn.run(1, query)
np.testing.assert_array_equal(indices[:, 0], [0, 1])
np.testing.assert_allclose(distances[:, 0], [0.01, 0.01], atol=1e-6)
checkpoint = ROOT / "cache/torch/checkpoints" / weights.url.rsplit("/", 1)[-1]
result = {
    "status": "PASS", "time": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
    "purpose": "synthetic tensor integration test; no real defect metrics",
    "author_commit": COMMIT, "weights": str(weights), "weights_url": weights.url,
    "weights_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    "backbone": "resnet50 (not paper default wide_resnet50_2)",
    "training_tensors": 2, "image_shape": [3, 256, 256],
    "coreset_fraction": 0.1,
    "memory_bank_vectors": model.anomaly_scorer.nn_method.search_index.ntotal,
    "map_shape": list(maps.shape), "faiss_known_answer": "PASS",
    "peak_allocated_mib": torch.cuda.max_memory_allocated(0) / 1024**2,
    "peak_reserved_mib": torch.cuda.max_memory_reserved(0) / 1024**2,
}
(ROOT / "results").mkdir(exist_ok=True)
(ROOT / "results/smoke_patchcore.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
