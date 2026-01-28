import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder


class TCGAMultimodalDataset(Dataset):
    """
    TCGA multimodal dataset:
    - Visual embeddings (TITAN)
    - Text embeddings (Qwen encoder output from 1.4)
    - Patient-aware splits
    """

    def __init__(self, split: str, cfg: dict):
        assert split in {"train", "val", "test"}, f"Invalid split: {split}"
        self.split = split

        # -------------------------
        # Config
        # -------------------------
        data_cfg = cfg["data"]
        model_cfg = cfg["model"]

        self.visual_embeddings_path = data_cfg["visual_embeddings_path"]
        self.text_embeddings_path = data_cfg["text_embeddings_path"]
        self.labels_path = data_cfg["labels_path"]
        self.splits_dir = Path(data_cfg["splits_dir"])

        self.modality = model_cfg["modality"]
        self.num_classes = model_cfg["num_classes"]

        # -------------------------
        # Load splits
        # -------------------------
        with open(self.splits_dir / f"{split}.json") as f:
            self.patient_ids = json.load(f)

        # -------------------------
        # Load embeddings
        # -------------------------
        with open(self.visual_embeddings_path, "rb") as f:
            self.visual_embeddings = pickle.load(f)

        with open(self.text_embeddings_path, "rb") as f:
            self.text_embeddings = pickle.load(f)

        # -------------------------
        # Load labels
        # -------------------------
        labels_df = pd.read_csv(self.labels_path)
        self.pid_to_label = dict(
            zip(labels_df["pid"], labels_df["cancer_type"])
        )

        # Encode labels to integers [0, 32]
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(labels_df["cancer_type"])

        # -------------------------
        # Input dimension
        # -------------------------
        if self.modality == "multimodal":
            self.input_dim = 768 + 768
        elif self.modality == "vision":
            self.input_dim = 768
        elif self.modality == "text":
            self.input_dim = 768
        else:
            raise ValueError(f"Unknown modality: {self.modality}")

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        pid = self.patient_ids[idx]

        # --- Visual embedding ---
        visual_emb = self.visual_embeddings[pid]
        if isinstance(visual_emb, list):
            visual_emb = visual_emb[0]
        visual_emb = torch.tensor(visual_emb, dtype=torch.float32)

        # --- Text embedding ---
        text_emb = self.text_embeddings[pid]
        if isinstance(text_emb, list):
            text_emb = text_emb[0]
        text_emb = torch.tensor(text_emb, dtype=torch.float32)

        # --- Fuse modalities ---
        if self.modality == "multimodal":
            x = torch.cat([visual_emb, text_emb], dim=0)
        elif self.modality == "vision":
            x = visual_emb
        elif self.modality == "text":
            x = text_emb
        else:
            raise RuntimeError("Invalid modality")

        # --- Label ---
        label_str = self.pid_to_label[pid]
        y = self.label_encoder.transform([label_str])[0]
        y = torch.tensor(y, dtype=torch.long)

        return x, y