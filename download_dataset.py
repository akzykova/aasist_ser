import argparse
import os
import subprocess
import gdown

def download_la_dataset():
    print("Downloading ASVspoof19 LA dataset...")
    url = "https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip?sequence=3&isAllowed=y"
    zip_path = "./dataset/LA.zip"

    subprocess.run(["curl", "-L", "-o", zip_path, "-#", url])
    subprocess.run(["unzip", "-q", zip_path, "-d", "./dataset"])
    os.remove(zip_path)

def download_emotional_dataset():
    print("Downloading Emotional Speech Dataset...")
    file_id = "1vDCHiMKfJiylum_IHZL3uZXVJkQpdrQv"
    zip_path = "emotional_speech.zip"
    
    gdown.download(
        f"https://drive.google.com/uc?id={file_id}",
        output=zip_path,
        quiet=False
    )
    subprocess.run(["unzip", "-q", zip_path, "-d", "./dataset"])
    os.remove(zip_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Скачивание аудио-датасетов")
    parser.add_argument(
        "--dataset", 
        type=str, 
        required=True, 
        choices=["LA", "emotional"],
        help="Выберите датасет: 'LA' (ASVspoof19) или 'emotional' (Датасет эмоциональной сгенерированной речи)"
    )
    args = parser.parse_args()
    os.makedirs("./dataset", exist_ok=True) 

    if args.dataset == "LA":
        download_la_dataset()
    else:
        download_emotional_dataset()