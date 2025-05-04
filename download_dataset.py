import argparse
import os
import subprocess
import gdown

def download_la_dataset():
    url = "https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip?sequence=3&isAllowed=y"
    zip_path = "LA.zip"

    subprocess.run(["curl", "-L", "-o", zip_path, "-#", url])
    subprocess.run(["unzip", "-q", zip_path])
    os.remove(zip_path)

def download_emotional_dataset(folder_url):
    os.makedirs("dataset", exist_ok=True)
    print("Downloading Dataset of Synthesized Emotional Speech...")
    gdown.download_folder(folder_url, output="dataset", quiet=False, use_cookies=False)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Скачивание аудио-датасетов")
    # parser.add_argument("--dataset", type=str, required=True, choices=["LA", "emotional"],
    #                     help="Выберите датасет: 'LA' или 'emotional'")
    # parser.add_argument("--drive-url", type=str, help="URL Google Drive папки (для emotional)")

    args = parser.parse_args()
    download_la_dataset()

