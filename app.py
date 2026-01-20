from flask import Flask, send_from_directory, request, jsonify, Response
import os
import tempfile
import json
import time
import uuid
import html
from sudachipy import tokenizer
from sudachipy import dictionary

_whisper_model = None
_sudachi_tokenizer = None


def _get_sudachi():
    global _sudachi_tokenizer
    if _sudachi_tokenizer is not None:
        return _sudachi_tokenizer
    try:
        _sudachi_tokenizer = dictionary.Dictionary().create()
    except Exception as e:
        print(f"Warning: Failed to load Sudachi: {e}")
    return _sudachi_tokenizer



def _get_whisper_model():
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model

    try:
        from faster_whisper import WhisperModel
    except Exception as e:
        raise RuntimeError(
            "Missing dependency 'faster-whisper'. Install it with: pip install faster-whisper"
        ) from e

    # Use a small-ish model by default for speed. You can change to 'small' or 'medium'.
    _whisper_model = WhisperModel("large-v3", device="auto", compute_type="int8")
    return _whisper_model

app = Flask(__name__, static_folder='static', static_url_path='/static', template_folder='.')

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), 'uploads')
RECORDINGS_DIR = os.path.join(os.path.dirname(__file__), 'data', 'recordings')
META_PATH = os.path.join(UPLOAD_DIR, 'metadata.json')
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RECORDINGS_DIR, exist_ok=True)


@app.route('/')
def index():
    # serve the workspace root index.html
    return send_from_directory('.', 'index.html')


@app.route('/uploads/<path:filename>')
def uploads(filename):
    # Backward compatible: older items live in ./uploads, newer recordings in ./data/recordings
    upload_path = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(upload_path):
        return send_from_directory(UPLOAD_DIR, filename)

    rec_path = os.path.join(RECORDINGS_DIR, filename)
    if os.path.exists(rec_path):
        return send_from_directory(RECORDINGS_DIR, filename)

    return jsonify({'error': 'Not found'}), 404


@app.route('/uploads/')
def uploads_index():
        items = _load_meta()
        items.sort(key=lambda x: x.get('createdAt', 0), reverse=True)

        rows = []
        for item in items:
                display = html.escape(item.get('displayName') or item.get('originalName', ''))
                kind = html.escape(item.get('kind', ''))
                url = item.get('url', '')
                created_at = html.escape(item.get('createdAtText') or str(item.get('createdAt', 0)))
                rows.append(
                        f"<tr>"
                        f"<td>{kind}</td>"
                        f"<td>{display}</td>"
                        f"<td><a href=\"{url}\">download</a></td>"
                        f"<td style=\"color:#666\">{created_at}</td>"
                        f"</tr>"
                )

        body = "\n".join(rows) if rows else "<tr><td colspan=\"4\">No saved files yet.</td></tr>"

        page = f"""<!doctype html>
<html lang=\"en\">
<head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>Saved Pitch Audio</title>
    <style>
        body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; margin: 24px; color: #111; }}
        h1 {{ margin: 0 0 8px; }}
        p {{ margin: 0 0 16px; color: #444; }}
        table {{ border-collapse: collapse; width: 100%; max-width: 980px; }}
        th, td {{ border-bottom: 1px solid #e7e7e7; padding: 10px 8px; text-align: left; }}
        th {{ font-size: 12px; color: #666; text-transform: uppercase; letter-spacing: 0.04em; }}
        a {{ color: #3b5bdb; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
        .top {{ display: flex; gap: 12px; align-items: baseline; flex-wrap: wrap; }}
    </style>
</head>
<body>
    <div class=\"top\">
        <h1>Saved audio</h1>
        <a href=\"/\">Back to app</a>
    </div>
    <p>These files are saved on the server (uploads/). Use the app’s “Saved Overlay” dropdown to apply them as overlays.</p>
    <table>
        <thead>
            <tr><th>Kind</th><th>Name</th><th>File</th><th>Created</th></tr>
        </thead>
        <tbody>
            {body}
        </tbody>
    </table>
</body>
</html>"""

        return Response(page, mimetype='text/html')


def _load_meta():
    if not os.path.exists(META_PATH):
        return []
    try:
        with open(META_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return []


def _save_meta(items):
    with open(META_PATH, 'w', encoding='utf-8') as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


def _add_audio_item(kind: str, filename: str, original_name: str, storage: str = 'uploads'):
    items = _load_meta()
    item_id = str(uuid.uuid4())
    created_at = int(time.time())
    created_text = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(created_at))
    display_name = f"{original_name} ({created_text})"

    item = {
        'id': item_id,
        'kind': kind,  # 'upload' | 'recording'
        'filename': filename,
        'originalName': original_name,
        'displayName': display_name,
        'createdAt': created_at,
        'createdAtText': created_text,
        'storage': storage,
        'url': f'/uploads/{filename}',
    }
    items.append(item)
    _save_meta(items)
    return item


@app.route('/api/audio/list', methods=['GET'])
def audio_list():
    items = _load_meta()
    # Sort alphabetically by originalName (case-insensitive)
    items.sort(key=lambda x: (x.get('originalName', '').lower(), x.get('createdAt', 0)))
    return jsonify({'items': items})


@app.route('/api/audio/save-upload', methods=['POST'])
def audio_save_upload():
    if 'file' not in request.files:
        return jsonify({'error': 'Missing file'}), 400
    f = request.files['file']
    if not f or not f.filename:
        return jsonify({'error': 'Empty file'}), 400

    suffix = os.path.splitext(f.filename)[1] or '.bin'
    safe_name = f"upload_{int(time.time())}_{uuid.uuid4().hex}{suffix}"
    path = os.path.join(UPLOAD_DIR, safe_name)
    f.save(path)
    item = _add_audio_item('upload', safe_name, f.filename, storage='uploads')
    return jsonify(item)


@app.route('/api/audio/save-recording', methods=['POST'])
def audio_save_recording():
    if 'file' not in request.files:
        return jsonify({'error': 'Missing file'}), 400
    f = request.files['file']
    if not f or not f.filename:
        return jsonify({'error': 'Empty file'}), 400

    suffix = os.path.splitext(f.filename)[1] or '.webm'
    safe_name = f"rec_{int(time.time())}_{uuid.uuid4().hex}{suffix}"
    path = os.path.join(RECORDINGS_DIR, safe_name)
    f.save(path)
    item = _add_audio_item('recording', safe_name, f.filename, storage='data/recordings')
    return jsonify(item)


@app.route('/api/stt', methods=['POST'])
def stt():
    if 'file' not in request.files:
        return jsonify({"error": "Missing file"}), 400

    f = request.files['file']
    if not f or not f.filename:
        return jsonify({"error": "Empty file"}), 400

    try:
        model = _get_whisper_model()
    except Exception as e:
        return jsonify({"error": str(e)}), 501

    suffix = os.path.splitext(f.filename)[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = tmp.name
        f.save(tmp_path)

    try:
        segments_iter, info = model.transcribe(tmp_path, language="ja", word_timestamps=True)
        segments_list = list(segments_iter)
        print(segments_list)

        full_text = "".join(seg.text for seg in segments_list).strip()

        # Sudachi for high-quality readings
        sudachi = _get_sudachi()
        tokens = []
        if sudachi:
            mode = tokenizer.Tokenizer.SplitMode.C
            tokens = sudachi.tokenize(full_text, mode)

        def is_katakana_word(s):
            # Regex basic check if string is all katakana (including long vowel -)
            import re
            return bool(re.fullmatch(r'[ァ-ンー]+', s))

        def kata2hira(s):
            # Simple Katakana (0x30A1-0x30F6) to Hiragana conversion
            res = []
            for c in s:
                code = ord(c)
                if 0x30A1 <= code <= 0x30F6:
                    res.append(chr(code - 0x60))
                else:
                    res.append(c)
            return "".join(res)

        full_reading = ""
        context_words = [] # Each item: {text, reading}
        
        import re
        # Filter out purely punctuation/symbol tokens if needed, or handle in loop
        # Punctuation regex
        punct_re = re.compile(r'^[。、！？「」『』・,.\?!]+$')

        # Common Japanese particles that should merge with previous word
        particles = {'は', 'を', 'が', 'に', 'へ', 'と', 'で', 'や', 'の', 'も', 'か', 'な', 'ね', 'よ', 'わ', 'から', 'まで', 'より'}
        
        def starts_with_small_tsu(s):
            return s and (s[0] == 'っ' or s[0] == 'ッ')

        def should_break_after_n(current_hira, next_hira):
            """
            Japanese 'n' (ん/ン) should cause a break unless followed by n-row kana (na, ni, nu, ne, no).
            """
            if not current_hira or (current_hira[-1] != 'ん' and current_hira[-1] != 'ン'):
                return False
            if not next_hira:
                return True # End of sentence/phrase
            
            n_row = {'な', 'に', 'ぬ', 'ね', 'の', 'ナ', 'ニ', 'ヌ', 'ネ', 'ノ'}
            return next_hira[0] not in n_row

        def split_on_special_kana(word_obj, next_word_reading=None):
            """
            Splits words on small 'tsu' (っ/ッ) and 'n' (ん/ン) (moraic nasal).
            """
            reading = word_obj['reading']
            text = word_obj['text']
            
            # Identify indices where we should split
            # 1. Before every 'っ' or 'ッ'
            # 2. After 'ん' or 'ン' IF condition is met
            splits = []
            
            for i in range(len(reading)):
                char = reading[i]
                # Small tsu break (starts a new chunk)
                if (char == 'っ' or char == 'ッ') and i > 0:
                    splits.append(i)
                # Moraic nasal 'n' break (ends a chunk)
                if char == 'ん' or char == 'ン':
                    next_char = reading[i+1] if i + 1 < len(reading) else (next_word_reading[0] if next_word_reading else None)
                    if should_break_after_n(char, next_char):
                        splits.append(i + 1)
            
            # Deduplicate and sort
            splits = sorted(list(set(splits)))
            if not splits:
                return [word_obj]
            
            res = []
            last_r_idx = 0
            last_t_idx = 0
            r_len = len(reading)
            t_len = len(text)
            
            for r_idx in splits:
                if r_idx > last_r_idx:
                    t_take = int((r_idx / r_len) * t_len)
                    t_take = max(last_t_idx + 1, t_take) if (t_len > 1 and r_idx < r_len) else t_take
                    
                    # If this split was caused by 'っ' or 'ッ', the NEXT part has the break.
                    # If caused by 'ん' or 'ン', THIS part ends with a break.
                    is_n_split = reading[r_idx-1] in ('ん', 'ン')
                    
                    res.append({
                        'text': text[last_t_idx:t_take],
                        'reading': reading[last_r_idx:r_idx],
                        'force_break': is_n_split
                    })
                    last_r_idx = r_idx
                    last_t_idx = t_take
            
            if last_r_idx < r_len:
                res.append({
                    'text': text[last_t_idx:],
                    'reading': reading[last_r_idx:],
                    'force_break': False
                })
            
            # Ensure the very last part knows if it should break based on the NEXT word
            if res and reading[-1] in ('ん', 'ン') and should_break_after_n(reading[-1], next_word_reading):
                res[-1]['force_break'] = True

            return res
        
        if tokens:
            temp_words = []
            for i in range(len(tokens)):
                surface = tokens[i].surface()
                if punct_re.match(surface):
                    continue

                reading_kata = tokens[i].reading_form()
                if is_katakana_word(surface):
                    final_reading = reading_kata
                else:
                    final_reading = kata2hira(reading_kata)

                # Peek at next token for 'n' logic
                next_reading = None
                if i + 1 < len(tokens):
                    nr = tokens[i+1].reading_form()
                    next_reading = kata2hira(nr)

                split_parts = split_on_special_kana({'text': surface, 'reading': final_reading}, next_reading)
                for p in split_parts:
                    p['force_break'] = p.get('force_break', False) or starts_with_small_tsu(p['reading'])
                    temp_words.append(p)
            
            # Merge particles with previous words
            merged_words = []
            i = 0
            while i < len(temp_words):
                current = temp_words[i]
                
                # Check if this is a particle and should be merged with previous
                # BUT: if the particle starts with a small tsu (rare) or follows a force_break, 
                # we might want to be careful. For now, keep user request: particles don't start chunks.
                is_particle = current['text'] in particles and len(merged_words) > 0 and not current['force_break']
                
                if is_particle:
                    # Merge with previous word
                    prev = merged_words[-1]
                    prev['text'] += current['text']
                    prev['reading'] += current['reading']
                else:
                    merged_words.append(current)
                
                i += 1
            
            for word in merged_words:
                full_reading += word['reading']
                context_words.append({'text': word['text'], 'reading': word['reading']})
        else:
            context_words.append({'text': full_text, 'reading': full_text})
            full_reading = full_text

        # Create a unified character stress from Whisper words for precise alignment
        # We need to map every character in the clean text to a timestamp.
        # Since Whisper gives word timestamps, we assume linear distribution within the word.
        
        char_timeline = [] # [{char, start, end}]
        current_whisper_idx = 0
        
        # Helper: Clean a string of punctuation for alignment
        def clean_for_align(s):
            return re.sub(r'[^\w]', '', s) # Naive clean, might be too aggressive
        
        # Better: iterate segments and just use what we have, skipping punctuation manually
        for seg in segments_list:
            source = seg.words if (hasattr(seg, "words") and seg.words) else [seg]
            for w in source:
                w_text = w.word if hasattr(w, "word") else w.text
                if not w_text: continue
                
                # Filter punctuation from the whisper word itself for char calculation?
                # Actually, we should keep it to consume 'surface' properly in some cases?
                # But context_words (Sudachi) has punctuation removed.
                # So we should skip punctuation chars tokens here too.
                
                w_start = w.start
                w_end = w.end
                duration = w_end - w_start
                
                # Identify valid chars
                # We want to keep Kana/Kanji/Alphanum, discard symbols for alignment mapping
                import unicodedata
                valid_chars = []
                for c in w_text:
                    if not punct_re.match(c) and not c.isspace():
                        valid_chars.append(c)
                        
                if not valid_chars:
                    continue
                    
                char_len = len(valid_chars)
                if char_len == 0: continue
                
                avg_char_dur = duration / char_len
                
                for i, c in enumerate(valid_chars):
                    c_start = w_start + (i * avg_char_dur)
                    c_end = c_start + avg_char_dur
                    char_timeline.append({'char': c, 'start': c_start, 'end': c_end})

        segments_payload = []
        c_idx = 0
        
        # Map Sudachi tokens to Char Timeline
        for item in context_words:
            orig = item['text']
            hira = item['reading']
            
            # Clean 'orig' of punctuation just in case (though we filtered tokens)
            orig_clean = "".join([c for c in orig if not punct_re.match(c) and not c.isspace()])
            if not orig_clean:
                continue

            # We need to find `len(orig_clean)` characters in `char_timeline` starting at `c_idx`
            if c_idx >= len(char_timeline):
                break
                
            # Align: consume chars from timeline
            matched_chars = []
            
            # Simple greedy match: assume correct order.
            # We take the next N valid chars from timeline.
            # (In a perfect world, char_timeline[c_idx].char == orig_clean[0])
            
            # We will just take the next len(orig_clean) items from timeline
            needed = len(orig_clean)
            
            if c_idx + needed <= len(char_timeline):
                slice_segment = char_timeline[c_idx : c_idx + needed]
                
                # Start time is start of first char, End time is end of last char
                seg_start = slice_segment[0]['start']
                seg_end = slice_segment[-1]['end']
                
                segments_payload.append({
                    "start": float(seg_start),
                    "end": float(seg_end),
                    "text": orig,
                    "reading": hira,
                    "force_break": bool(item.get('force_break', False))
                })
                
                c_idx += needed
            else:
                # Run out of timeline?
                break

        return jsonify({
            "text": full_text,
            "reading": full_reading,
            "segments": segments_payload,
            "language": getattr(info, "language", "ja")
        })
    except Exception as e:
        return jsonify({"error": f"Transcription failed: {e}"}), 500
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


if __name__ == '__main__':
    # Bind to all interfaces so it's easy to access from the host/browser
    app.run(host='0.0.0.0', port=5001, debug=True)
