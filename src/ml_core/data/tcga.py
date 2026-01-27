from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pickle
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class TCGARecord:
    pid: str
    y: int


class TCGADataset(Dataset):
    """
    Patient-level TCGA dataset.
    Returns (x_vis [768], y, pid).

    Notes:
    - One sample = one patient (pid)
    - Embeddings may be stored as (768,) or (1, 768); we squeeze to 1D.
    """

    def __init__(self, pids: List[str], tcga_dir: str | Path) -> None:
        self.tcga_dir = Path(tcga_dir)

        emb_path = self.tcga_dir / "tcga_titan_embeddings.pkl"
        label_path = self.tcga_dir / "tcga_patient_to_cancer_type.csv"

        if not emb_path.exists():
            raise FileNotFoundError(f"Missing embeddings file: {emb_path}")
        if not label_path.exists():
            raise FileNotFoundError(f"Missing labels file: {label_path}")

        # Load embeddings dict: pid -> {"embeddings": np.ndarray}
        with open(emb_path, "rb") as f:
            self.emb: Dict[str, Dict] = pickle.load(f)

        # Load labels (columns: patient_id, cancer_type)
        df = pd.read_csv(label_path)
        df["patient_id"] = df["patient_id"].astype(str)
        df["cancer_type"] = df["cancer_type"].astype(str)

        # Intersection: only patients that have embeddings
        df = df[df["patient_id"].isin(self.emb.keys())].copy()

        # Stable label mapping (sorted for reproducibility)
        classes = sorted(df["cancer_type"].unique().tolist())
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}

        pid_to_y = dict(
            zip(
                df["patient_id"].tolist(),
                df["cancer_type"].map(self.class_to_idx).tolist(),
            )
        )

        # Restrict to split pids + intersection
        keep = [pid for pid in pids if pid in pid_to_y and pid in self.emb]
        self.records = [TCGARecord(pid=pid, y=pid_to_y[pid]) for pid in keep]

        if len(self.records) == 0:
            raise RuntimeError("TCGADataset has 0 records after filtering/intersection.")

        # Infer visual embedding dimension (robust to (1,768))
        any_pid = self.records[0].pid
        vec = np.asarray(self.emb[any_pid]["embeddings"])
        vec = np.squeeze(vec)
        if vec.ndim != 1:
            raise ValueError(
                f"Expected 1D embedding after squeeze, got {vec.shape} for pid={any_pid}"
            )
        self.vis_dim = int(vec.shape[0])

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        r = self.records[idx]
        x = np.asarray(self.emb[r.pid]["embeddings"], dtype=np.float32)
        x = np.squeeze(x)  # handle (1,768) -> (768,)
        x_t = torch.from_numpy(x)
        y_t = torch.tensor(r.y, dtype=torch.long)
        return x_t, y_t, r.pid
