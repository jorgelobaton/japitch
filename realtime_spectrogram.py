#!/usr/bin/env python3
"""Realtime spectrogram + pitch (F0) visualization.

Usage: python realtime_spectrogram.py

Requires a working microphone and the packages in requirements.txt.
"""
import argparse
import queue
import sys
import threading
import time

import numpy as np
import sounddevice as sd
import librosa
import matplotlib.pyplot as plt


def int_or_none(x):
    return None if x is None else int(x)


def run(realtime_seconds=5.0, device=None, samplerate=None):
    channels = 1
    # audio params
    if samplerate is None:
        samplerate = int(sd.query_devices(device, 'input')['default_samplerate'])

    samplerate = int(samplerate)
    frame_size = 2048
    hop_length = 256
    window_seconds = float(realtime_seconds)
    buffer_size = int(window_seconds * samplerate)

    q = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        # ensure mono
        if indata.ndim > 1:
            data = np.mean(indata, axis=1)
        else:
            data = indata
        q.put(data.copy())

    stream = sd.InputStream(callback=callback, channels=channels, samplerate=samplerate, device=device)

    # circular buffer
    buffer = np.zeros(buffer_size, dtype='float32')
    buff_lock = threading.Lock()

    running = True

    def audio_consumer():
        nonlocal buffer
        while running:
            try:
                chunk = q.get(timeout=0.1)
            except queue.Empty:
                continue
            chunk = np.asarray(chunk, dtype='float32')
            with buff_lock:
                if chunk.size >= buffer.size:
                    buffer = chunk[-buffer.size : ]
                else:
                    buffer = np.roll(buffer, -chunk.size)
                    buffer[-chunk.size:] = chunk

    t = threading.Thread(target=audio_consumer, daemon=True)
    t.start()

    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 5))

    img = None
    line, = ax.plot([], [], color='cyan', lw=2)
    ax.set_title('Realtime spectrogram (dB) + F0 (Hz)')

    freqs = librosa.fft_frequencies(sr=samplerate, n_fft=frame_size)

    try:
        stream.start()
        print('Listening... press Ctrl-C to stop')
        last_update = 0.0
        update_interval = 0.06  # ~16 FPS
        while True:
            start = time.time()
            with buff_lock:
                y = buffer.copy()

            # compute STFT magnitude dB
            S = np.abs(librosa.stft(y, n_fft=frame_size, hop_length=hop_length))
            S_db = librosa.amplitude_to_db(S, ref=np.max)

            # compute times for x axis
            n_frames = S_db.shape[1]
            times = np.linspace(-window_seconds, 0.0, n_frames)

            if img is None:
                extent = [times[0], times[-1], freqs[0], freqs[-1]]
                img = ax.imshow(S_db, origin='lower', aspect='auto', extent=extent, cmap='magma')
                ax.set_ylabel('Frequency (Hz)')
                ax.set_xlabel('Time (s)')
                cb = fig.colorbar(img, ax=ax)
                cb.set_label('dB')
                ax.set_ylim(50, 5000)
            else:
                img.set_data(S_db)
                img.set_extent([times[0], times[-1], freqs[0], freqs[-1]])

            # estimate F0 using librosa.yin on the buffer
            # yin returns f0 per frame; we use same hop_length
            try:
                f0 = librosa.yin(y, fmin=50, fmax=1000, sr=samplerate, frame_length=frame_size, hop_length=hop_length)
                f0_times = np.linspace(-window_seconds, 0.0, f0.size)
                # convert NaNs to nan so plotting won't join them
            except Exception:
                f0 = np.array([])
                f0_times = np.array([])

            if f0.size:
                line.set_data(f0_times, f0)
            else:
                line.set_data([], [])

            ax.relim()
            # keep the frequency limits reasonable
            ax.set_xlim(-window_seconds, 0.0)

            fig.canvas.draw()
            fig.canvas.flush_events()

            elapsed = time.time() - start
            to_sleep = update_interval - elapsed
            if to_sleep > 0:
                time.sleep(to_sleep)

    except KeyboardInterrupt:
        print('\nStopping...')
    finally:
        running = False
        try:
            stream.stop()
            stream.close()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seconds', '-s', type=float, default=5.0, help='Seconds of rolling window (default 5.0)')
    parser.add_argument('--device', '-d', type=int_or_none, default=None, help='Input device ID')
    parser.add_argument('--samplerate', '-r', type=int_or_none, default=None, help='Samplerate override')
    args = parser.parse_args()
    run(realtime_seconds=args.seconds, device=args.device, samplerate=args.samplerate)


if __name__ == '__main__':
    main()
