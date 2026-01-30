import os
import glob
import numpy as np
import pandas as pd
import librosa
import unicodedata
from faster_whisper import WhisperModel
from sudachipy import tokenizer, dictionary
import difflib

# Configuration
JVS_ROOT = "data/jvs_ver1"
SUBSET = "parallel100"
SAMPLE_LIMIT = 200  # Set to None to run full dataset

# --- 1. JVS Phoneme to Kana Mapping ---
MORA_MAP = {
    'a': 'あ', 'i': 'い', 'u': 'う', 'e': 'え', 'o': 'お',
    'I': 'い', 'U': 'う',
    'ka': 'か', 'ki': 'き', 'ku': 'く', 'ke': 'け', 'ko': 'こ',
    'kI': 'き', 'kU': 'く', 'kya': 'きゃ', 'kyu': 'きゅ', 'kyo': 'きょ',
    'sa': 'さ', 'si': 'し', 'su': 'す', 'se': 'せ', 'so': 'そ',
    'sI': 'し', 'sU': 'す', 'sha': 'しゃ', 'shi': 'し', 'shu': 'しゅ', 'she': 'シェ', 'sho': 'しょ',
    'ta': 'た', 'ti': 'ち', 'tu': 'つ', 'te': 'て', 'to': 'と',
    'tI': 'ち', 'tU': 'つ', 'tsu': 'つ', 'tsU': 'つ', 'tsa': 'つぁ',
    'chi': 'ち', 'cha': 'ちゃ', 'chu': 'ちゅ', 'che': 'チェ', 'cho': 'ちょ',
    'na': 'な', 'ni': 'に', 'nu': 'ぬ', 'ne': 'ね', 'no': 'の',
    'nya': 'にゃ', 'nyu': 'にゅ', 'nyo': 'にょ',
    'ha': 'は', 'hi': 'ひ', 'hu': 'ふ', 'he': 'へ', 'ho': 'ほ',
    'fa': 'ふぁ', 'fi': 'ふぃ', 'fu': 'ふ', 'fe': 'ふぇ', 'fo': 'ふぉ',
    'hya': 'ひゃ', 'hyu': 'ひゅ', 'hyo': 'ひょ',
    'ma': 'ま', 'mi': 'み', 'mu': 'む', 'me': 'め', 'mo': 'も',
    'mya': 'みゃ', 'myu': 'みゅ', 'myo': 'みょ',
    'ya': 'や', 'yu': 'ゆ', 'yo': 'よ',
    'ra': 'ら', 'ri': 'り', 'ru': 'る', 're': 'れ', 'ro': 'ろ',
    'rya': 'りゃ', 'ryu': 'りゅ', 'ryo': 'りょ',
    'wa': 'わ', 'wo': 'を',
    'ga': 'が', 'gi': 'ぎ', 'gu': 'ぐ', 'ge': 'げ', 'go': 'ご',
    'gya': 'ぎゃ', 'gyu': 'ぎゅ', 'gyo': 'ぎょ',
    'za': 'ざ', 'zi': 'じ', 'zu': 'ず', 'ze': 'ぜ', 'zo': 'ぞ',
    'ja': 'じゃ', 'ji': 'じ', 'ju': 'じゅ', 'jo': 'じょ',
    'da': 'だ', 'di': 'ぢ', 'du': 'づ', 'de': 'で', 'do': 'ど',
    'ba': 'ば', 'bi': 'び', 'bu': 'ぶ', 'be': 'べ', 'bo': 'ぼ',
    'bya': 'びゃ', 'byu': 'びゅ', 'byo': 'びょ',
    'pa': 'ぱ', 'pi': 'ぴ', 'pu': 'ぷ', 'pe': 'ぺ', 'po': 'ぽ',
    'pya': 'ぴゃ', 'pyu': 'ぴゅ', 'pyo': 'ぴょ',
    'N': 'ん', 'cl': 'っ'
}

def parse_jvs_lab(lab_path):
    """Parses JVS .lab file into a list of {kana, start, end}."""
    with open(lab_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    phonemes = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 3:
            phonemes.append({
                'start': float(parts[0]),
                'end': float(parts[1]),
                'ph': parts[2]
            })
            
    # Group phonemes into Morae -> Kana
    kanas = []
    i = 0
    while i < len(phonemes):
        curr = phonemes[i]
        p = curr['ph']
        
        # Ignorables
        if p in ['sil', 'pau']:
            i += 1
            continue
            
        mora_ph = p
        start = curr['start']
        end = curr['end']
        
        # Lookahead for vowel
        next_ph = phonemes[i+1] if i + 1 < len(phonemes) else None
        
        # If consonant, try to merge with next vowel
        if p not in ['a','i','u','e','o','I','U','N','cl'] and next_ph:
            # Simple check: if next is vowel-like or 'y'+vowel or 'w'+vowel
            # A simpler heuristic for this script: greedy match against map keys
            combined = p + next_ph['ph']
            if combined in MORA_MAP:
                mora_ph = combined
                end = next_ph['end']
                i += 2 # Consume two
            elif p in MORA_MAP:
                 # Standalone consonant mapped? (Could be 'n' mistake?)
                 # Fallback: Assume the single char maps or skip
                 i += 1
            else:
                 # Unknown
                 i += 1
        else:
            i += 1
            
        if mora_ph in MORA_MAP:
            kanas.append({
                'text': MORA_MAP[mora_ph],
                'start': start,
                'end': end
            })
            
    return kanas


# Initialize Models Once
# Match the App's configuration explicitly!
print("Loading Whisper...")
whisper = WhisperModel("medium", device="cpu", compute_type="int8") # App uses "medium"

# Import process logic from App
# We need to add the current dir to path
import sys
sys.path.append(os.getcwd())
from app import process_transcription_results
# Note: process_transcription_results returns a flask Response object in my last edit to app.py.
# We need to handle that.

def detect_pitch_core_python(frame, sr):
    # Port of app.js detectPitchCore
    # frame: numpy array of floats
    n = len(frame)
    mean = np.mean(frame)
    
    # RMS
    energy = np.sum((frame - mean)**2)
    rms = np.sqrt(energy / n)
    if rms <= 1e-6:
        return -1, 0
        
    # Autocorrelation (Direct port of JS logic)
    # x = ((frame - mean) / rms) * w
    # w = 0.5 - 0.5 * cos(...)
    
    window = 0.5 - 0.5 * np.cos((2 * np.pi * np.arange(n)) / (n - 1))
    x = ((frame - mean) / rms) * window
    
    # To speed up in Python, use numpy correlate
    # full correlation
    correlations = np.correlate(x, x, mode='full')
    # The center is at index n-1. 
    # Offset k corresponds to index (n-1) + k
    
    min_offset = int(sr / 500)
    max_offset = int(sr / 50)
    
    lags = np.arange(min_offset, max_offset)
    if len(lags) == 0: return -1, 0
    
    # Slice valid lags from full correlation
    # correlate mode='full' result size is 2*n - 1.
    center = n - 1
    corr_slice = correlations[center + lags]
    
    # Normalize by (n - lag)
    divisors = n - lags
    corr_norm = corr_slice / divisors
    
    best_idx = np.argmax(corr_norm)
    best_corr = corr_norm[best_idx]
    best_offset = lags[best_idx]
    
    # Parabolic interpolation
    if best_offset > min_offset and best_offset < max_offset - 1:
        if 0 < best_idx < len(corr_norm) - 1:
            y1 = corr_norm[best_idx - 1]
            y2 = corr_norm[best_idx]
            y3 = corr_norm[best_idx + 1]
            
            d = (y3 - y1) / (2 * (2 * y2 - y1 - y3))
            refined_offset = best_offset + d
            
            return sr / refined_offset, best_corr
            
    return sr / best_offset, best_corr

def run_pipeline(wav_path):
    # A. Pitch Detection & Chunks (Visual Trails)
    # Emulates app.js logic
    y, sr = librosa.load(wav_path, sr=None) # Load full
    
    # App params
    buffer_length = 2048
    hop = 256 # App uses overlapping loop with step? 
    # app.js: for (start=0; start+frameSize<=len; start+=hop)
    
    trail = []
    
    # Constants from app.js
    rmsSpeechThresholdOff = 0.010
    minFreq = 50
    maxFreq = 500
    plotRows = 3
    
    pixelsPerSecond = 100
    canvas_width = 1000 # Guess
    
    last_plotted_ms = None
    last_row_index = None
    stable = 0
    smooth = None
    
    # Loop like app.js
    # Converted to range for performance
    num_frames = (len(y) - buffer_length) // hop
    
    for i in range(num_frames + 1):
        start = i * hop
        frame = y[start : start + buffer_length]
        
        # RMS
        rms = np.sqrt(np.mean(frame**2))
        
        t_sec = start / sr
        x_total = t_sec * pixelsPerSecond
        row_index = int(x_total / canvas_width)
        
        if row_index >= 3: # 3 rows max
             break
             
        # Gating
        if rms < rmsSpeechThresholdOff:
            smooth = None
            stable = 0
            continue
            
        pitch, corr = detect_pitch_core_python(frame, sr)
        
        if not (minFreq < pitch < maxFreq) or corr < 0.28:
            stable = 0
            continue
            
        stable = min(stable + 1, 10)
        if stable < 2:
            continue
            
        # Smoothing
        alpha = 0.35
        if smooth is None:
            smooth = pitch
        else:
            smooth = alpha * pitch + (1 - alpha) * smooth
            
        # Gap Logic
        gap_ms = 120
        now_ms = t_sec * 1000
        
        gap = False
        if last_plotted_ms is not None:
            if (now_ms - last_plotted_ms) > gap_ms:
                gap = True
            if row_index != last_row_index:
                gap = True
                
        trail.append({
            't': t_sec,
            'pitch': smooth,
            'gap': gap,
            'row': row_index
        })
        
        last_plotted_ms = now_ms
        last_row_index = row_index

    # Extract Chunks (Python port of extractTrailChunks)
    if not trail: return []
    
    chunks = []
    current_chunk_pts = []
    current_row = None
    
    for p in trail:
        starts_new = (not current_chunk_pts) or p['gap'] or (current_row is not None and p['row'] != current_row)
        
        if starts_new:
            if current_chunk_pts:
                chunks.append({
                    'start': current_chunk_pts[0]['t'],
                    'end': current_chunk_pts[-1]['t']
                })
            current_chunk_pts = [p]
            current_row = p['row']
        else:
            current_chunk_pts.append(p)
            
    if current_chunk_pts:
        chunks.append({
            'start': current_chunk_pts[0]['t'],
            'end': current_chunk_pts[-1]['t']
        })
        
    chunks_df = pd.DataFrame(chunks)
    if chunks_df.empty: return []

    # B. STT & Linguistics (Using App Logic)
    segments_iter, info = whisper.transcribe(wav_path, language="ja", word_timestamps=True)
    segments_list = list(segments_iter)
    
    # CALL APP LOGIC
    # We use a dummy Flask context to handle the response object if it uses jsonify
    from flask import Flask
    dummy = Flask(__name__)
    with dummy.app_context():
        response = process_transcription_results(segments_list, info)
        import json
        data = json.loads(response.get_data(as_text=True))
        
    stt_segments = data['segments'] # list of {start, end, text, reading, force_break}

    # C. Midpoint Alignment (Python port of recomputeChunkLabels)
    predictions = []
    
    # We need to split segments into Chars if we want to predict per-Char
    # OR we follow the App's "Label" logic which assigns the whole READING to the chunk.
    # The JVS GT is per-kana.
    # If the app assigns "わたし" to Chunk A, then "わ", "た", "し" are all predicted on Chunk A.
    
    for seg in stt_segments:
        s_start = seg['start']
        s_end = seg['end']
        reading = seg['reading']
        
        s_mid = (s_start + s_end) * 0.5
        
        # Find best chunk
        best_chunk = None
        min_dist = float('inf')
        
        for _, chunk in chunks_df.iterrows():
            c_mid = (chunk['start'] + chunk['end']) * 0.5
            dist = abs(s_mid - c_mid)
            if dist < min_dist:
                min_dist = dist
                best_chunk = chunk
        
        if best_chunk is not None:
            for char in reading:
                # App logic: The whole label is placed on the chunk.
                # So every char in that label effectively predicts that chunk window.
                predictions.append({
                    'kana': char,
                    'pred_start': float(best_chunk['start']),
                    'pred_end': float(best_chunk['end'])
                })
                
    return predictions

# --- 3. Evaluation Loop ---


def evaluate():
    results = []
    
    # Find JVS folders
    speaker_dirs = glob.glob(os.path.join(JVS_ROOT, 'jvs*'))
    print(f"Found {len(speaker_dirs)} speakers.")
    
    count = 0
    
    for spk_dir in speaker_dirs:
        wav_dir = os.path.join(spk_dir, SUBSET, 'wav24kHz16bit')
        lab_dir = os.path.join(spk_dir, SUBSET, 'lab/mon')
        
        wav_files = sorted(glob.glob(os.path.join(wav_dir, '*.wav')))
        
        for wav_path in wav_files:
            if SAMPLE_LIMIT is not None and count >= SAMPLE_LIMIT:
                break
            
            basename = os.path.basename(wav_path).replace('.wav', '')
            lab_path = os.path.join(lab_dir, f"{basename}.lab")
            
            if not os.path.exists(lab_path):
                continue
                
            print(f"Processing {basename}...")
            
            # Get Ground Truth
            gt_kanas = parse_jvs_lab(lab_path)
            
            # Get Predictions
            try:
                preds = run_pipeline(wav_path)
            except Exception as e:
                print(f"Pipeline failed for {wav_path}: {e}")
                continue
                
            # Alignment / Scoring
            # We align the lists of tokens (not characters) to handle multi-char kanas like 'きゃ'
            gt_tokens = [x['text'] for x in gt_kanas]
            pred_tokens = [x['kana'] for x in preds]
            
            matcher = difflib.SequenceMatcher(None, gt_tokens, pred_tokens)
            
            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                if tag == 'equal':
                    # Match found! Check timestamps.
                    for k in range(i2-i1):
                        gt_idx = i1 + k
                        pred_idx = j1 + k
                        
                        gt_item = gt_kanas[gt_idx]
                        pred_item = preds[pred_idx]
                        
                        # HIT LOGIC: GT Start strictly within Predicted Chunk Window
                        gt_s = gt_item['start']
                        p_s = pred_item['pred_start']
                        p_e = pred_item['pred_end']
                        
                        hit = (p_s <= gt_s <= p_e)
                        
                        # Deviation logic
                        deviation = 0.0
                        if not hit:
                            # Distance to closest boundary (Start or End)
                            dist_start = abs(gt_s - p_s)
                            dist_end = abs(gt_s - p_e)
                            deviation = min(dist_start, dist_end)
                        
                        results.append({
                            'filename': basename,
                            'kana_gt': gt_item['text'],
                            'kana_pred': pred_item['kana'],
                            'text_match': True, # By definition of 'equal' tag
                            'gt_start': gt_s,
                            'pred_window_start': p_s,
                            'pred_window_end': p_e,
                            'hit': hit,
                            'deviation': deviation
                        })
            count += 1
            
    # Summary
    df = pd.DataFrame(results)
    if not df.empty:
        acc = df['hit'].mean()
        # Avg deviation only for False hits
        miss_df = df[df['hit'] == False]
        avg_dev = miss_df['deviation'].mean() if not miss_df.empty else 0.0
        
        print("\n--- Evaluation Results ---")
        print(df.head())
        print(f"\nTotal Kana Evaluated: {len(df)}")
        print(f"Global Alignment Accuracy (Hit Rate): {acc:.2%}")
        print(f"Average Time Deviation on Misses: {avg_dev:.4f} sec")
        df.to_csv("evaluation_results.csv", index=False)
    else:
        print("No results generated.")

if __name__ == "__main__":
    evaluate()
