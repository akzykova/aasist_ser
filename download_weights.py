import argparse
import os
import gdown

MODEL_WEIGHTS = {
    "AASIST": ["1G0XElZOCmM69HxfopTUM7EegqgftFH5R"],
    "AASIST_Concat": ["1wG7O46BN7PPKZi_w-PDxfvF_enBZZNNd"],
    "AASIST_FILM": ["1UsZSQ8C37Wbc4nSrRpmAwrXZ76NLaFVU"],
    "AASIST_GFILM": ["1gWLT6d6HYiCVXBSSj6yPWfpTmH0dqdQF"],
    "AASIST_WAV2VEC": ["1G0XElZOCmM69HxfopTUM7EegqgftFH5R", "13bmoy2pfZVGsmyJyy9BFYHF9KLXU9WDv", "1w2A6PAATLJ85qIMFkVUuzY-Ku3roegiG"],
    "AMSDF": ["1G0XElZOCmM69HxfopTUM7EegqgftFH5R", "1yZGhXZCQRRXZtBmuvsWglZ_Twb21g6dY", "1w2A6PAATLJ85qIMFkVUuzY-Ku3roegiG"],
}

def download_weight(model_name, dest_folder="./models/weights"):
    os.makedirs(dest_folder, exist_ok=True)

    if model_name not in MODEL_WEIGHTS:
        print(f"Модель '{model_name}' не найдена. Доступны: {', '.join(MODEL_WEIGHTS.keys())}")
        return

    file_ids = MODEL_WEIGHTS[model_name]
    for file_id in file_ids:
        url = f"https://drive.google.com/uc?id={file_id}"

        output_path = os.path.join(dest_folder, f"{model_name}.pth")
        if os.path.exists(output_path):
            print(f"⚠️  Файл для {model_name} уже существует: {output_path}")
            continue

        print(f"Скачивание весов для {model_name}...")
        gdown.download(url, output_path, quiet=False)
        print(f"Скачано в {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Скачивание весов для предобученных моделей")
    parser.add_argument("--model", type=str, required=True, choices=["AASIST", "AASIST_Concat", "AASIST_FILM", "AASIST_GFILM", "AMSDF", "AASIST_WAV2VEC"],
                        help="Выберите модель из предложенных: AASIST, AASIST_Concat, AASIST_FILM, AASIST_GFILM, AMSDF, AASIST_WAV2VEC")
    parser.add_argument("--outpit_dir", type=str, help="Введите название директории для скачивания весов")
    args = parser.parse_args()

    download_weight(args.model, args.outpit_dir)
