# Japanese Pitch Accent Trainer

A real-time Japanese pitch accent visualizer for language learners. Record full sentences and compare your pitch trails against native speakers'. Captions are automatically generated and placed on their respective trails.

## Features

- **Precision Pitch Tracking**: Uses autocorrelation with parabolic interpolation for sub-sample accuracy, eliminating visual "stair-stepping."
- **Timeline**: Continuous visualization over long phrases.
- **Japanese STT Integration**: Automatically transcribes audio using `faster-whisper` and provides Hiragana/Katakana readings via `SudachiPy`.
- **Caption Mapping**: Intelligently aligns transcription readings to pitch trail segments using a midpoint-based best-fit algorithm, ensuring lyrics appear directly over the corresponding vocalizations.
- **Overlay and Practice**: Save recordings, upload reference files, and overlay them to visually compare your pitch contours.

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: FFmpeg is required for audio processing.*

2. **Run the App**:
   ```bash
   python app.py
   ```

3. **Open in Browser**:
   Navigate to `http://localhost:5001`.

## Usage

- **Start Mic**: Begin real-time pitch feedback.
- **Record**: Save your practice sessions to compare later.
- **Upload Reference**: Upload any audio to get a "truth" trail with automatically assigned readings.
- **Apply Overlay**: Selection a previous recording or upload from the dropdown to see it on the canvas.

## Technical Details

### Pitch Detection Engine
- **Algorithm**: Autocorrelation-based frequency estimation refined with **parabolic interpolation**. This achieves sub-sample precision, removing the "stair-step" artifacts common in digital pitch detection.
- **Stability Logic**: Implements a 2-frame onset stability check and a 120ms speech hangover to prevent trail flickering during fast consonants or glottal stops.
- **Raw Audio Pipeline**: Explicitly disables browser-level `echoCancellation`, `noiseSuppression`, and `autoGainControl` to preserve raw harmonic content for the detector.

### STT Alignment
- **Linguistic Logic**: The backend (SudachiPy) identifies moraic structures that require physical gaps in the pitch trail, such as the sokuon (っ) and the moraic nasal (ん). Timing from `faster-whisper` is mapped to acoustic "chunks" in the pitch history using a midpoint-based best-fit algorithm.

## Local CLI Tool

For quick local testing without a browser, you can use the secondary Python script:

```bash
python realtime_spectrogram.py
```

### Script Options:
- `--seconds` (`-s`): rolling window length in seconds (default 5.0).
- `--device` (`-d`): numeric input device ID.
- `--samplerate` (`-r`): override hardware samplerate.

*Note: The CLI tool uses `librosa.yin` for pitch estimation and is separate from the high-precision Web Audio engine used in the Flask app.*
