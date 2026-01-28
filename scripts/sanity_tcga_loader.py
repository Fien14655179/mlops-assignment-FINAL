from pathlib import Path

import yaml
from ml_core.data import get_tcga_dataloaders

cfg_path = Path("experiments/configs/train_config.yaml")
cfg = yaml.safe_load(cfg_path.read_text())

train_loader, val_loader, test_loader = get_tcga_dataloaders(cfg)

xb, yb, pidb = next(iter(train_loader))
print("Batch x:", xb.shape, xb.dtype)
print("Batch y:", yb.shape, yb.dtype)
print("Example pid:", pidb[0])

print(
    "Train/Val/Test sizes:",
    len(train_loader.dataset),
    len(val_loader.dataset),
    len(test_loader.dataset),
)
print("Vis dim:", train_loader.dataset.vis_dim)
print("Num classes:", len(train_loader.dataset.class_to_idx))
