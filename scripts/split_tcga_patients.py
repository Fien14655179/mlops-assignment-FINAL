from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
import pickle


def main() -> None:
    # Paths (relative, TA-proof)
    data_root = Path("data/tcga")
    emb_path = data_root / "tcga_titan_embeddings.pkl"
    labels_path = data_root / "tcga_patient_to_cancer_type.csv"
    out_dir = Path("data/splits")
    out_dir.mkdir(parents=True, exist_ok=True)

    seed = 42
    test_size = 0.15
    val_size = 0.15
    # train will be 0.70

    # --- Load pids that actually have embeddings ---
    with open(emb_path, "rb") as f:
        emb = pickle.load(f)
    pids_with_emb = set(emb.keys())

    # --- Load labels and filter to available embeddings ---
    df = pd.read_csv(labels_path)
    df = df[df["patient_id"].isin(pids_with_emb)].copy()

    # Safety checks
    assert df["patient_id"].nunique() == len(df), (
        "Expected 1 label per patient_id after filtering. "
        "If duplicates exist, we need to decide aggregation."
    )
    assert df["patient_id"].nunique() == len(pids_with_emb), (
        "Mismatch between pids in embeddings and filtered label table."
    )

    # --- Stratified split on cancer_type ---
    pids = df["patient_id"].tolist()
    y = df["cancer_type"].tolist()

    p_train, p_temp, y_train, y_temp = train_test_split(
        pids, y,
        test_size=(test_size + val_size),
        random_state=seed,
        stratify=y,
    )

    # Split temp into val and test (val proportion inside temp)
    val_ratio_in_temp = val_size / (test_size + val_size)
    p_val, p_test, y_val, y_test = train_test_split(
        p_temp, y_temp,
        test_size=(1 - val_ratio_in_temp),
        random_state=seed,
        stratify=y_temp,
    )

    # --- No-overlap asserts ---
    s_train, s_val, s_test = set(p_train), set(p_val), set(p_test)
    assert s_train.isdisjoint(s_val)
    assert s_train.isdisjoint(s_test)
    assert s_val.isdisjoint(s_test)
    assert len(s_train) + len(s_val) + len(s_test) == len(pids_with_emb)

    # --- Write outputs ---
    (out_dir / "train_pids.json").write_text(json.dumps(sorted(s_train), indent=2))
    (out_dir / "val_pids.json").write_text(json.dumps(sorted(s_val), indent=2))
    (out_dir / "test_pids.json").write_text(json.dumps(sorted(s_test), indent=2))

    # --- Print summary (screenshot-worthy) ---
    print("Split written to:", out_dir.resolve())
    print("Patients (train/val/test):", len(s_train), len(s_val), len(s_test))
    print("Classes:", df["cancer_type"].nunique())


if __name__ == "__main__":
    main()
