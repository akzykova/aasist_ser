import argparse
import json
import os
import random
from pathlib import Path
import torch
import numpy as np
from typing import Dict, List
from data_utils import DatasetCustom
from utils import set_seed
from importlib import import_module


def get_model(model_config: Dict, device: torch.device):
    module = import_module("models.{}".format(model_config["architecture"]))
    _model = getattr(module, "Model")
    model = _model(model_config).to(device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    print("no. model params:{}".format(nb_params))

    if model_config.get("model_path"):
        state_dict = torch.load(model_config["model_path"], map_location=device)
        model_state_dict = model.state_dict()
        filtered_state_dict = {
            k: v for k, v in state_dict.items() 
            if k in model_state_dict and v.shape == model_state_dict[k].shape
        }
        
        model.load_state_dict(filtered_state_dict, strict=False)
        print("Weights are downloaded from ", model_config["model_path"])

    return model

def save_results(results: List[tuple], output_file: str):
    with open(output_file, "w") as f:
        for utt_id, score in results:
            f.write(f"{utt_id}: {score}\n")

def run_inference(args: argparse.Namespace):    
    with open(args.config, "r") as f_json:
        config = json.loads(f_json.read())
    
    set_seed(args.seed, config)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Using seed: {args.seed}")

    model = get_model(config["model_config"], device)
    model.eval()

    test_dataset = DatasetCustom(audio_dir=Path(args.test_dir))
    
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        persistent_workers=False,
        generator=generator
    )

    print("Running inference on test files...")
    results = []
    for batch_x, batch_emo, filenames in test_loader:
        batch_x = batch_x.float().to(device)
        batch_emo = batch_emo.float().to(device)
        
        _, batch_out = model(batch_x, batch_emo)
        scores = batch_out[:, 1].data.cpu().numpy().ravel()
        results.extend(zip(filenames, map(float, scores)))
            

    save_results(results, args.output_file)
    print(f"Inference complete. Results saved to {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AASIST Inference with Seed Fixation")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--test_dir", type=str, required=True, help="Directory with test audio files")
    parser.add_argument("--output_file", type=str, default="inference_results.txt", help="Output file path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()
    run_inference(args)