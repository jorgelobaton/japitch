import os
import glob
import time
import pandas as pd
import jiwer
from faster_whisper import WhisperModel
from tqdm import tqdm

# --- CONFIGURATION ---
SPEEDSPEECH_ROOT = "./data/speedspeech_ja_2022_v1.0.0" 
TRANSCRIPT_PATH = "./data/recitation_transcript_utf8.txt"
MODELS_TO_COMPARE = ["large-v3", "medium", "small"] # Comparing 3 models
DEVICE = "cpu"
COMPUTE_TYPE = "int8"
CPU_THREADS = 8
BEAM_SIZE = 1

def load_ground_truth(path):
    truth = {}
    try:
        with open(path, 'r', encoding='utf-8-sig') as f:
            for line in f:
                if ':' in line: parts = line.strip().split(':')
                elif '\t' in line: parts = line.strip().split('\t')
                else: parts = [line.strip()]
                if len(parts) >= 2:
                    truth[parts[0].strip()] = parts[1].split(',')[0].strip()
    except FileNotFoundError:
        print("Transcript not found.")
    return truth

def get_file_metadata(filepath):
    parts = filepath.replace("\\", "/").split("/")
    filename = parts[-1]
    basename = os.path.splitext(filename)[0]
    try:
        number_part = basename.split('_')[-1]
        clean_number = f"{int(number_part):03d}"
        file_id = f"RECITATION324_{clean_number}"
    except ValueError:
        file_id = basename
    try:
        speed = parts[-2].split("_")[1] if "_" in parts[-2] else "unknown"
    except IndexError:
        speed = "unknown"
    return file_id, speed

def evaluate():
    ground_truth = load_ground_truth(TRANSCRIPT_PATH)
    wav_files = glob.glob(os.path.join(SPEEDSPEECH_ROOT, "**/*.wav"), recursive=True)
    
    all_results = []

    # --- LOOP THROUGH MODELS ---
    for model_name in MODELS_TO_COMPARE:
        print(f"\n\n=== EVALUATING MODEL: {model_name} ===")
        model = WhisperModel(model_name, device=DEVICE, compute_type=COMPUTE_TYPE, cpu_threads=CPU_THREADS)
        
        start_time_global = time.time()
        
        for wav_path in tqdm(wav_files, desc=f"{model_name}"):
            file_id, speed = get_file_metadata(wav_path)
            if file_id not in ground_truth: continue
            
            reference = ground_truth[file_id]
            
            # Transcribe
            start_t = time.time()
            segments, _ = model.transcribe(wav_path, language="ja", beam_size=BEAM_SIZE)
            hypothesis = "".join([s.text for s in segments])
            proc_time = time.time() - start_t
            
            # Metrics
            output = jiwer.process_characters(reference, hypothesis)
            cer = output.cer
            
            # Calculate "Course Accuracy": (N - S - I - D) / N
            # jiwer provides these counts directly
            N = len(reference)
            S, I, D = output.substitutions, output.insertions, output.deletions
            # Note: Accuracy can be negative if Insertions are high!
            course_accuracy = (N - S - I - D) / N if N > 0 else 0
            
            all_results.append({
                "Model": model_name,
                "Speed_Category": speed,
                "File_ID": file_id,
                "CER": cer,
                "Course_Accuracy": course_accuracy,
                "Processing_Time": proc_time
            })

    # Save and Summarize
    df = pd.DataFrame(all_results)
    df.to_csv("comparison_results.csv", index=False)
    
    print("\n=== FINAL COMPARISON (Average Accuracy) ===")
    print(df.groupby(["Model", "Speed_Category"])["Course_Accuracy"].mean())
    
    print("\n=== SPEED COMPARISON (Avg Time per File) ===")
    print(df.groupby(["Model"])["Processing_Time"].mean())

if __name__ == "__main__":
    evaluate()
