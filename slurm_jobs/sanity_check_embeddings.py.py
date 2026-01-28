#!/usr/bin/env python3
import json
import pickle

import numpy as np

# --- Config --- 
ENCODER_PKL = "./data/processed/text/v1/encoder_embeddings_test.pkl"
DECODER_JSONL = "./data/processed/text/v1/decoder_output_test_v3.jsonl"

# --- Load embeddings ---
print("[INFO] Loading embeddings .pkl file...")
try:
    with open(ENCODER_PKL, "rb") as f:
        embeddings = pickle.load(f)
except Exception as e:
    print("[ERROR] Failed to load .pkl:", e)
    exit(1)

print(f"[INFO] Total embeddings loaded: {len(embeddings)}")

# --- Load decoder records to compare count ---
print("[INFO] Loading decoder output JSONL...")
with open(DECODER_JSONL, "r", encoding="utf-8") as f:
    decoder_records = [json.loads(line) for line in f if line.strip()]

print(f"[INFO] Total decoder records: {len(decoder_records)}")

if len(embeddings) > len(decoder_records):
    print("[WARNING] More embeddings than decoder records!")

# --- Check each embedding ---
all_shapes_correct = True
all_values_valid = True

print("\n[INFO] Checking individual embeddings (first 5 PIDs)...")
for pid in list(embeddings.keys())[:5]:
    emb = embeddings[pid]
    if not isinstance(emb, np.ndarray):
        print(f"[ERROR] PID {pid} embedding is not a numpy array!")
        all_shapes_correct = False
        continue
    if emb.shape != (1024,):
        print(f"[WARNING] PID {pid} embedding shape {emb.shape} != (1024,)")
        all_shapes_correct = False
    # Check min/max/mean
    min_val, max_val, mean_val = emb.min(), emb.max(), emb.mean()
    print(f"PID {pid}: shape={emb.shape}, min={min_val:.6f}, max={max_val:.6f}, mean={mean_val:.6f}")
    if np.all(emb == 0):
        print(f"[WARNING] PID {pid} embedding is all zeros!")
        all_values_valid = False

# --- Summary ---
print("\n[SUMMARY]")
print(f"Total embeddings: {len(embeddings)}")
print(f"Total decoder records: {len(decoder_records)}")
print(f"All shapes correct: {all_shapes_correct}")
print(f"All values valid: {all_values_valid}")

if len(embeddings) == len(decoder_records) and all_shapes_correct and all_values_valid:
    print("[SUCCESS] Embeddings appear to be correct and ready for downstream use!")
else:
    print("[WARNING] Some issues detected in embeddings. Check warnings above.")
