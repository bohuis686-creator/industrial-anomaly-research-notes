"""无下载的特征提取检查；随机权重不能用于检测效果评价。"""
import json
import torch
import torchvision
from torchvision.models import resnet50
from torchvision.models.feature_extraction import create_feature_extractor

torch.manual_seed(1592)
assert torch.cuda.is_available()
torch.cuda.init()
model = create_feature_extractor(
    resnet50(weights=None), return_nodes={"layer2": "layer2", "layer3": "layer3"}
).eval().cuda()
torch.cuda.reset_peak_memory_stats(0)
with torch.inference_mode():
    features = model(torch.randn(1, 3, 256, 256, device="cuda"))
torch.cuda.synchronize(0)
assert all(torch.isfinite(value).all().item() for value in features.values())
assert features["layer2"].shape == (1, 512, 32, 32)
assert features["layer3"].shape == (1, 1024, 16, 16)
print(json.dumps({
    "status": "PASS", "weights": "random; not an accuracy experiment",
    "torch": torch.__version__, "torchvision": torchvision.__version__,
    "gpu": torch.cuda.get_device_name(0),
    "features": {k: list(v.shape) for k, v in features.items()},
    "peak_allocated_mib": torch.cuda.max_memory_allocated(0) / 1024**2,
    "peak_reserved_mib": torch.cuda.max_memory_reserved(0) / 1024**2,
}, indent=2))
