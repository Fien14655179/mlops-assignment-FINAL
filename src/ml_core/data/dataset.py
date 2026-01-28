import json
import pickle
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd

class TCGAMultimodalDataset(Dataset):
    def __init__(self, split: str, cfg: dict):
        self.split = split
        data_cfg = cfg["data"]
        model_cfg = cfg["model"]

        # Paths
        self.visual_path = data_cfg["visual_embeddings_path"]
        self.text_path = data_cfg["text_embeddings_path"]
        self.splits_dir = Path(data_cfg["splits_dir"])
        self.modality = model_cfg.get("modality", "multimodal")
        
        # 1. Load PIDs
        with open(self.splits_dir / f"{split}.json", "r") as f:
            self.patient_ids = json.load(f)

        # 2. Load Visual Data (Pickle)
        with open(self.visual_path, "rb") as f:
            self.visual_data = pickle.load(f)

        # 3. Load Text Data (Pickle)
        with open(self.text_path, "rb") as f:
            self.text_data = pickle.load(f)

        # 4. Load Labels
        with open(data_cfg["label_mapping"], "r") as f:
            self.class_to_idx = json.load(f)
            
        df = pd.read_csv(data_cfg["labels_path"])
        self.pid_to_label = dict(zip(df["patient_id"], df["cancer_type"]))

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        pid = self.patient_ids[idx]

        # A. Visual (Mean Pooling)
        vis_emb = self.visual_data.get(pid, np.zeros(768))
        if isinstance(vis_emb, list): # If multiple images
            vis_emb = np.mean(vis_emb, axis=0)
        x_vis = torch.tensor(vis_emb, dtype=torch.float32)

        # B. Text
        txt_emb = self.text_data.get(pid, np.zeros(768))
        x_txt = torch.tensor(txt_emb, dtype=torch.float32)

        # C. Label
        label_str = self.pid_to_label.get(pid)
        y = torch.tensor(self.class_to_idx.get(label_str, 0), dtype=torch.long)

        return x_vis, x_txt, y, pid
