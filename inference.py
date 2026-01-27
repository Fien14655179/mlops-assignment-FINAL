import argparse
import json
import os
import torch


def load_model(model_path: str):
    raise NotImplementedError("Model loading not yet implemented")


def load_data(split: str):
    raise NotImplementedError("Data loading not yet implemented")


def run_inference(model, data, config):
    raise NotImplementedError("Inference logic not yet implemented")


def main():
    parser = argparse.ArgumentParser(description="Run inference with trained fusion model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model weights (.pt)")
    parser.add_argument("--split", type=str, choices=["val", "test"], required=True, help="Which split to run inference on")
    parser.add_argument("--config", type=str, required=True, help="Path to model config JSON")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = json.load(f)
    model = load_model(args.model_path)
    data = load_data(args.split)
    predictions = run_inference(model, data, config)
    print("Predictions ready (not implemented).")


if __name__ == "__main__":
    main()
