import os
import argparse
from pydub import AudioSegment

def convert_wav_to_flac(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for filename in os.listdir(input_dir):
        if filename.endswith(".wav"):  
            input_file = os.path.join(input_dir, filename)
            output_file = os.path.join(output_dir, os.path.splitext(filename)[0] + ".flac")
            
            audio = AudioSegment.from_wav(input_file)
            audio = audio.set_frame_rate(16000)
            
            audio.export(output_file, format="flac")
            print(f"Converted {filename} to FLAC")
    
    print("All WAV files have been converted to FLAC.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert WAV files to FLAC format.")
    parser.add_argument("--input_dir", required=True, help="Path to the directory containing WAV files.")
    parser.add_argument("--output_dir", required=True, help="Path to the directory where FLAC files will be saved.")
    
    args = parser.parse_args()
    convert_wav_to_flac(args.input_dir, args.output_dir)
