from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

from torch.utils.data import DataLoader

from .tcga import TCGADataset


def _load_pids(path: Path) -> list[str]:
    return json.loads(path.read_text())


def get_tcga_dataloaders(cfg: dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    data_cfg = cfg["data"]
    base = Path(data_cfg.get("data_dir", "data"))
    tcga_dir = base / "tcga"
    splits = base / "splits"

    train_pids = _load_pids(splits / "train_pids.json")
    val_pids = _load_pids(splits / "val_pids.json")
    test_pids = _load_pids(splits / "test_pids.json")

    train_ds = TCGADataset(train_pids, tcga_dir=tcga_dir)
    val_ds = TCGADataset(val_pids, tcga_dir=tcga_dir)
    test_ds = TCGADataset(test_pids, tcga_dir=tcga_dir)

    batch_size = int(data_cfg.get("batch_size", 64))
    num_workers = int(data_cfg.get("num_workers", 2))

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader
