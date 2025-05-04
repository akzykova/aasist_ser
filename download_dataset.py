import argparse
import os
import subprocess
import gdown

def download_la_dataset():
    url = "https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip?sequence=3&isAllowed=y"
    zip_path = "LA.zip"

    print("⬇️  Скачивание ASVspoof2019 LA.zip...")
    subprocess.run(["curl", "-L", "-o", zip_path, "-#", url])
    print("📦 Распаковка LA.zip...")
    subprocess.run(["unzip", "-q", zip_path])
    os.remove(zip_path)
    print("✅ LA датасет скачан и распакован.")

def download_emotional_dataset(folder_url):
    os.makedirs("datasets", exist_ok=True)
    print("Downloading Dataset of Synthesized Emotional Speech...")
    gdown.download_folder(folder_url, output="datasets", quiet=False, use_cookies=False)
    print("✅ Эмоциональный датасет скачан.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Скачивание аудио-датасетов")
    # parser.add_argument("--dataset", type=str, required=True, choices=["LA", "emotional"],
    #                     help="Выберите датасет: 'LA' или 'emotional'")
    # parser.add_argument("--drive-url", type=str, help="URL Google Drive папки (для emotional)")

    args = parser.parse_args()
    download_la_dataset()

