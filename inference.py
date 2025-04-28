import argparse
import json
import os
import random
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Union
from models.AASIST_SER import AASISTWithEmotion
from data_utils import Dataset_Custom

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_model(model_config: Dict, device: torch.device) -> AASISTWithEmotion:
    model = AASISTWithEmotion(
        aasist_config=model_config["aasist_config"],
        ser_config=model_config["ser_config"]
    ).to(device)
    
    if "model_path" in model_config:
        print(f"\nLoading weights from {model_config['model_path']}")
        try:
            state_dict = torch.load(model_config["model_path"], map_location=device)

            # if 'aasist' in state_dict:
            #     model.aasist.load_state_dict(state_dict['aasist'])
            #     print("✓ AASIST weights loaded")
            
            if 'film' in state_dict:
                model.film.load_state_dict(state_dict['film'])
                print("✓ FiLM block weights loaded")
            
            if 'gated_block' in state_dict:
                model.gated_block.load_state_dict(state_dict['gated_block'])
                print("✓ Post-FiLM block weights loaded")
            
            if 'classifier' in state_dict:
                model.classifier.load_state_dict(state_dict['classifier'])
                print("✓ Classifier weights loaded")
            
        except Exception as e:
            print(f"\nError loading weights: {str(e)}")
            raise
    
    def count_params(module):
        return sum(p.numel() for p in module.parameters())
    
    total_params = count_params(model)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\nModel summary:")
    #print(f"- AASIST: {count_params(model.aasist):,} params")
    print(f"- FiLM block: {count_params(model.film):,} params")
    print(f"- Gated block: {count_params(model.gated_block):,} params")
    print(f"- Classifier: {count_params(model.classifier):,} params")
    print(f"Total: {total_params:,} params (Trainable: {trainable_params:,})")
    
    return model

def save_results(results: List[tuple], output_file: str):
    """Сохраняет результаты инференса в файл"""
    with open(output_file, "w") as f:
        for utt_id, score in results:
            f.write(f"{utt_id}: {score}\n")

def run_inference(args: argparse.Namespace):
    """Основная функция для выполнения инференса"""
    # Установка seed для воспроизводимости
    set_seed(args.seed)
    
    # Загрузка конфигурации
    with open(args.config, "r") as f_json:
        config = json.loads(f_json.read())
    
    # Переопределение seed из конфига, если он там указан
    seed = config.get("seed", args.seed)
    set_seed(seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Using seed: {seed}")

    # Инициализация модели
    model = get_model(config["model_config"], device)
    model.eval()

    # Подготовка данных
    audio_files = [f.stem for f in Path(args.test_dir).glob("*.flac")]
    test_dataset = Dataset_Custom(list_IDs=audio_files, base_dir=Path(args.test_dir))
    
    # DataLoader с фиксированным seed для воспроизводимости
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        generator=generator
    )

    # Выполнение инференса
    print("Running inference on test files...")
    results = []
    for batch_x, utt_id in test_loader:
        batch_x = batch_x.to(device)
        with torch.no_grad():
            _, batch_out = model(batch_x)
            batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()
        results.append((utt_id, batch_score))

    # Сохранение результатов
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