import argparse
import json
import os
import random
from pathlib import Path
import torch
import numpy as np
from typing import Dict, List
from data_utils import Dataset_Custom
from utils import set_seed

from models.AASIST_GFILM import AASISTGFILM
from models.AASIST_Concat import AASISTConcat
from models.AASIST_FILM import AASISTFILM
from models.AMSDF import Module
from models.AASIST import Model
from models.AASIST_WAV2VEC import WAV2VECModel


def get_model(model_name: str, model_config: Dict, device: torch.device):
    print(f'Getting the model {model_name}....')

    model_map = {
        "AASIST": Model,
        "AASIST_Concat": AASISTConcat,
        "AASIST_FILM": AASISTFILM,
        "AASIST_GFILM": AASISTGFILM,
        "AMSDF": Module,
        "AASIST_WAV2VEC": WAV2VECModel
    }
    
    if model_name not in model_map:
        raise ValueError(f"Model {model_name} is not recognized!")

    model_class = model_map[model_name]

    if model_name in ["AASIST_Concat", "AASIST_FILM", "AASIST_GFILM"]:
        model = model_class(
            aasist_config=model_config["aasist_config"],
            ser_config=model_config["ser_config"]
        ).to(device)
    else:
        model = model_class().to(device)
        
    if "model_path" in model_config:
        print(f"\nLoading weights from {model_config['model_path']}")
        try:
            state_dict = torch.load(model_config["model_path"], map_location=device)
            model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            print(f"Error loading model weights: {e}")
    
    return model

def save_results(results: List[tuple], output_file: str):
    with open(output_file, "w") as f:
        for utt_id, score in results:
            f.write(f"{utt_id}: {score}\n")

def run_inference(args: argparse.Namespace):
    set_seed(args.seed)
    
    with open(args.config, "r") as f_json:
        config = json.loads(f_json.read())
    
    seed = config.get("seed", args.seed)
    set_seed(seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Using seed: {seed}")

    model = get_model(args.model_choice, config["model_config"], device)
    model.eval()

    test_dataset = Dataset_Custom(audio_dir=Path(args.test_dir))
    
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    num_workers = min(4, os.cpu_count() // 2)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        persistent_workers=num_workers > 0,
        generator=generator
    )

    print("Running inference on test files...")
    results = []
    for batch_x, batch_emo, _, filenames in test_loader:
        try:
            batch_x = batch_x.float().cuda()
            batch_emo = batch_emo.float().cuda()
            
            _, batch_out = model(batch_x, batch_emo)
            scores = batch_out[:, 1].data.cpu().numpy().ravel()
            results.extend(zip(filenames, map(float, scores)))
            
            del batch_x, batch_emo, batch_out
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"\nError processing batch: {str(e)}")
            results.extend((f, None) for f in filenames)
            

    save_results(results, args.output_file)
    print(f"Inference complete. Results saved to {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AASIST Inference with Seed Fixation")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--test_dir", type=str, required=True, help="Directory with test audio files")
    parser.add_argument("--output_file", type=str, default="inference_results.txt", help="Output file path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--model_choice", type=str, required=True, choices=["AASIST", "AASIST_Concat", "AASIST_FILM", "AASIST_GFILM", "AMSDF", "AASIST_WAV2VEC"],
                        help="Choose model for inference")

    args = parser.parse_args()
    run_inference(args)