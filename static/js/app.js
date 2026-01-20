const canvas = document.getElementById('spectrogram');
const ctx = canvas.getContext('2d');
const pitchDisplay = document.getElementById('pitch-display');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const clearBtn = document.getElementById('clearBtn');
const recordBtn = document.getElementById('recordBtn');
const audioFileInput = document.getElementById('audioFile');
const sttText = document.getElementById('stt-text');
const overlaySelect = document.getElementById('overlaySelect');
const applyOverlayBtn = document.getElementById('applyOverlayBtn');
const playOverlayBtn = document.getElementById('playOverlayBtn');
const refreshOverlaysBtn = document.getElementById('refreshOverlaysBtn');
const minFreqSlider = document.getElementById('minFreq');
const maxFreqSlider = document.getElementById('maxFreq');

let audioContext, analyser, microphone, dataArray, animationId, mediaStream;
const bufferLength = 2048;
const spectrogramHeight = canvas.height;
let sampleRate = null; // set when AudioContext created
let smoothedPitch = null;
let lastPitchTime = 0;
let startTime = 0;
const pixelsPerSecond = 100; // adjust speed of trail
const plotRows = 3;
const rmsSpeechThreshold = 0.015; // increased to reject breath/noise on mic
const rmsSpeechThresholdOff = 0.010; // hysteresis
const speechHangoverMs = 120; // slightly shorter to avoid trailing noise
const minPitchCorrelation = 0.28; // higher threshold to reject rumble/breath
const pitchHoldMs = 60; // hold last pitch briefly through dips
let isSpeaking = false;
let stablePitchFrames = 0;
let lastPlottedTime = 0;
let lastPlottedRow = 0;
let lastVoiceActivityMs = 0;
let lastValidPitchMs = 0;

let mediaRecorder = null;
let recordedAudioChunks = [];

const RECORD_ICON = '⏺';
const STOP_ICON = '⏹';

let overlayAudioEl = null;
let overlayAudioUrl = null;
let overlayScaleX = 1.0;
let overlayAnimationId = null;

let sttSegments = null; // [{start,end,text,reading}]
let referenceChunks = null; // computed from referenceTrail
let chunkLabels = null; // [{chunk, label}]

function rowHeight() {
    return canvas.height / plotRows;
}

function mapPitchToY(pitchHz, minFreq, maxFreq, rowIndex) {
    const h = rowHeight();
    const yInRow = h - ((pitchHz - minFreq) / (maxFreq - minFreq)) * h;
    return rowIndex * h + yInRow;
}

const pitchHistory = []; // stores {x, y, gap: boolean}
let referenceTrail = null; // stores [{x, y, gap: boolean}]
let recording = false;
let recordedTrail = [];

function setReference(pattern) {
    const minFreq = parseInt(minFreqSlider.value);
    const maxFreq = parseInt(maxFreqSlider.value);
    const width = canvas.width;
    const rowIndex = 0;
    
    // Simple helper to create a pitch pattern based on a few points
    // pattern: [{t: 0-1, f: Hz}]
    referenceTrail = pattern.map(p => ({
        x: p.t * width,
        y: mapPitchToY(p.f, minFreq, maxFreq, rowIndex),
        gap: false
    }));
}

function drawRowSeparators() {
    const h = rowHeight();
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.12)';
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 6]);
    ctx.beginPath();
    for (let i = 1; i < plotRows; i++) {
        const y = Math.round(i * h) + 0.5;
        ctx.moveTo(0, y);
        ctx.lineTo(canvas.width, y);
    }
    ctx.stroke();
    ctx.setLineDash([]);
}

function drawPlayhead(x, rowIndex) {
    const h = rowHeight();
    const y0 = rowIndex * h;
    const y1 = y0 + h;

    // subtle glow
    ctx.strokeStyle = 'rgba(0, 212, 255, 0.22)';
    ctx.lineWidth = 6;
    ctx.beginPath();
    ctx.moveTo(x, y0);
    ctx.lineTo(x, y1);
    ctx.stroke();

    ctx.strokeStyle = 'rgba(255, 255, 255, 0.55)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(x, y0);
    ctx.lineTo(x, y1);
    ctx.stroke();
}

function renderIdleFrame() {
    // Draw the current canvas state even if the microphone isn't running.
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    drawRowSeparators();
    drawTrail(referenceTrail, { strokeStyle: 'rgba(255, 255, 255, 0.35)', lineWidth: 4, dash: null});
    drawTrail(pitchHistory, { strokeStyle: '#00f', lineWidth: 6, dash: null });

    drawChunkLabels();
    
    // Mic playhead (usually 0 if idle)
    if (mediaStream) {
        // if drawing loop is active, this function isn't the primary one
    } else {
        drawPlayhead(0, 0);
    }

    // Overlay playhead
    if (overlayAudioEl && !overlayAudioEl.paused) {
        const xTotal = overlayAudioEl.currentTime * pixelsPerSecond * overlayScaleX;
        const row = Math.floor(xTotal / canvas.width);
        const px = xTotal % canvas.width;
        if (row < plotRows) {
            drawPlayhead(px, row);
        }
    }
}

function extractTrailChunks(trail) {
    if (!trail || !trail.length) return [];
    const chunks = [];
    const h = rowHeight();
    const denom = pixelsPerSecond * (overlayScaleX || 1.0);

    const pushChunk = (points) => {
        if (!points || points.length < 2) return;
        const rowIndex = Math.floor(points[0].y / h);
        let minXTotal = Infinity;
        let maxXTotal = -Infinity;
        let minY = Infinity;
        let maxY = -Infinity;
        for (const p of points) {
            const xTotal = rowIndex * canvas.width + p.x;
            minXTotal = Math.min(minXTotal, xTotal);
            maxXTotal = Math.max(maxXTotal, xTotal);
            minY = Math.min(minY, p.y);
            maxY = Math.max(maxY, p.y);
        }
        const startT = denom > 0 ? (minXTotal / denom) : 0;
        const endT = denom > 0 ? (maxXTotal / denom) : startT;
        const midXTotal = (minXTotal + maxXTotal) * 0.5;
        const midX = midXTotal % canvas.width;
        chunks.push({
            rowIndex,
            minXTotal,
            maxXTotal,
            startT,
            endT,
            midX,
            minY,
            maxY,
        });
    };

    let cur = [];
    let curRow = null;

    for (let i = 0; i < trail.length; i++) {
        const p = trail[i];
        if (!p) continue;
        const rowIndex = Math.floor(p.y / h);

        const startsNew = (cur.length === 0) || p.gap || (curRow !== null && rowIndex !== curRow);
        if (startsNew) {
            if (cur.length) pushChunk(cur);
            cur = [p];
            curRow = rowIndex;
        } else {
            cur.push(p);
        }
    }
    if (cur.length) pushChunk(cur);
    return chunks;
}

function recomputeChunkLabels() {
    chunkLabels = null;
    if (!referenceChunks || !referenceChunks.length) return;
    if (!sttSegments || !sttSegments.length) return;

    // To prevent duplication, we assign each STT word to exactly ONE chunk (best fit).
    const chunkAssignments = new Map(); // chunkIndex -> [reading]

    for (const seg of sttSegments) {
        if (!seg || !seg.reading) continue;
        const s0 = typeof seg.start === 'number' ? seg.start : 0;
        const s1 = typeof seg.end === 'number' ? seg.end : s0;
        const sMid = (s0 + s1) * 0.5;

        let bestChunkIdx = -1;
        let maxScore = -Infinity;

        for (let i = 0; i < referenceChunks.length; i++) {
            const chunk = referenceChunks[i];
            const cMid = (chunk.startT + chunk.endT) * 0.5;
            
            // Use midpoint-to-midpoint distance for alignment.
            // This is more robust than pure overlap when trail chunks are short
            // or when Whisper word timings are slightly "wider" than the audio activity.
            const score = -Math.abs(sMid - cMid);

            if (score > maxScore) {
                maxScore = score;
                bestChunkIdx = i;
            }
        }

        if (bestChunkIdx !== -1) {
            if (!chunkAssignments.has(bestChunkIdx)) {
                chunkAssignments.set(bestChunkIdx, []);
            }
            chunkAssignments.get(bestChunkIdx).push(seg.reading.trim());
        }
    }

    const labels = [];
    console.log("--- Kana Trail Chunks ---");
    for (let i = 0; i < referenceChunks.length; i++) {
        const parts = chunkAssignments.get(i);
        const label = (parts && parts.length) ? parts.join('').trim() : "";
        console.log(`Chunk ${i} (${referenceChunks[i].startT.toFixed(2)}s - ${referenceChunks[i].endT.toFixed(2)}s): "${label}"`);
        if (label) {
            labels.push({ chunk: referenceChunks[i], label });
        }
    }
    console.log("--------------------------");
    chunkLabels = labels;
}

function drawChunkLabels() {
    if (!chunkLabels || !chunkLabels.length) return;
    if (!referenceTrail || !referenceTrail.length) return;

    const h = rowHeight();
    ctx.save();
    // Styling for "lyric" look
    ctx.font = 'bold 16px system-ui, -apple-system, sans-serif';
    ctx.textBaseline = 'alphabetic';
    ctx.textAlign = 'center';

    for (const item of chunkLabels) {
        const chunk = item.chunk;
        const text = (item.label || '').trim();
        if (!chunk || !text) continue;

        const rowIndex = chunk.rowIndex;
        if (rowIndex < 0 || rowIndex >= plotRows) continue;
        const rowTop = rowIndex * h;
        const rowBottom = rowTop + h;

        const x = chunk.midX;
        // Place above the highest point in the chunk, with a small offset
        const yAnchor = Math.max(rowTop + 20, Math.min(chunk.minY - 10, rowBottom - 8));

        // Lyric-style shadow for readability on dark background
        ctx.shadowColor = 'black';
        ctx.shadowBlur = 4;
        ctx.lineWidth = 3;
        ctx.strokeStyle = 'black';
        ctx.strokeText(text, x, yAnchor);

        ctx.fillStyle = 'white';
        ctx.fillText(text, x, yAnchor);
        
        ctx.shadowBlur = 0; // reset for next
    }

    ctx.restore();
}

function setOverlayAudio(urlOrNull) {
    overlayAudioUrl = urlOrNull;
    if (overlayAudioEl) {
        try { overlayAudioEl.pause(); } catch (e) {}
    }
    overlayAudioEl = null;
    playOverlayBtn.textContent = '▶';
    playOverlayBtn.disabled = !overlayAudioUrl;
}

async function transcribeFile(file, isUpload = false) {
    if (!file) {
        sttText.textContent = '';
        return;
    }

    sttText.textContent = 'Transcribing…';
    try {
        const form = new FormData();
        form.append('file', file);
        const resp = await fetch('/api/stt', { method: 'POST', body: form });
        const data = await resp.json();
        if (!resp.ok) {
            sttText.textContent = data && data.error ? data.error : 'STT failed';
            return;
        }
        const transcript = data.text || '';
        const reading = data.reading || '';
        sttSegments = Array.isArray(data.segments) ? data.segments : null;

        // Force gaps in the reference trail based on linguistic rules (っ, ん)
        // ONLY if the file is from an upload (not a recording overlay)
        if (sttSegments && referenceTrail && isUpload) {
            applyLinguisticGaps();
            // Re-extract chunks and labels because the trail has changed
            referenceChunks = extractTrailChunks(referenceTrail);
        }

        recomputeChunkLabels();
        sttText.innerHTML = `${transcript}<br><small style="color:rgba(255,255,255,0.6)">${reading}</small>`;
    } catch (e) {
        sttText.textContent = 'STT request failed';
    }
}

function applyLinguisticGaps() {
    if (!sttSegments || !referenceTrail || !referenceTrail.length) return;

    const denom = pixelsPerSecond * (overlayScaleX || 1.0);
    if (denom <= 0) return;

    for (const seg of sttSegments) {
        if (!seg.force_break) continue;

        // Find the points in the trail that follow this segment's end time
        // and force the next point to be a gap.
        const breakTime = seg.end;
        const breakXTotal = breakTime * denom;
        
        // Find the first point that is >= breakXTotal
        for (let i = 0; i < referenceTrail.length; i++) {
            const p = referenceTrail[i];
            const pRow = Math.floor(i / (referenceTrail.length / plotRows)); // approximate or use actual x/y
            // Correct way: points have x and we know row index implicitly or assume sorted
            // Since referenceTrail is built chronologically:
            const rowIdx = Math.floor(p.y / rowHeight());
            const pXTotal = rowIdx * canvas.width + p.x;

            if (pXTotal >= breakXTotal) {
                p.gap = true;
                break; // Found the break point for this segment
            }
        }
    }
}

function initAudio() {
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    analyser = audioContext.createAnalyser();
    analyser.fftSize = bufferLength;
    // Set to 0 to get raw unprocessed data for pitch detection
    analyser.smoothingTimeConstant = 0;
    dataArray = new Uint8Array(analyser.frequencyBinCount);
    sampleRate = audioContext.sampleRate;
}

function drawTrail(trail, style) {
    if (!trail || !trail.length) return;
    ctx.strokeStyle = style.strokeStyle;
    ctx.lineWidth = style.lineWidth;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    if (style.dash) ctx.setLineDash(style.dash);
    else ctx.setLineDash([]);
    ctx.beginPath();
    for (let i = 0; i < trail.length; i++) {
        const p = trail[i];
        if (i === 0 || p.gap) ctx.moveTo(p.x, p.y);
        else ctx.lineTo(p.x, p.y);
    }
    ctx.stroke();
    ctx.setLineDash([]);
}

function computeRms(frame) {
    let e = 0;
    for (let i = 0; i < frame.length; i++) e += frame[i] * frame[i];
    return Math.sqrt(e / frame.length);
}

function detectPitchCore(frame, sr) {
    let mean = 0;
    for (let i = 0; i < frame.length; i++) mean += frame[i];
    mean /= frame.length;

    let energy = 0;
    for (let i = 0; i < frame.length; i++) {
        const v = frame[i] - mean;
        energy += v * v;
    }
    const rms = Math.sqrt(energy / frame.length);
    if (rms <= 1e-6) return { pitch: -1, corr: 0 };

    const n = frame.length;
    const x = new Float32Array(n);
    for (let i = 0; i < n; i++) {
        const w = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (n - 1));
        x[i] = ((frame[i] - mean) / rms) * w;
    }

    let correlations = new Float32Array(n);
    let bestOffset = -1;
    let bestCorr = 0;
    const minOffset = Math.floor(sr / 500);
    const maxOffset = Math.floor(sr / 50);

    for (let offset = minOffset; offset < maxOffset; offset++) {
        let corr = 0;
        const limit = n - offset;
        for (let i = 0; i < limit; i++) corr += x[i] * x[i + offset];
        corr /= limit;
        correlations[offset] = corr;
        if (corr > bestCorr) {
            bestCorr = corr;
            bestOffset = offset;
        }
    }

    if (bestOffset > minOffset && bestOffset < maxOffset - 1) {
        // Parabolic interpolation for sub-sample precision (highly smooths the "stair-step" effect)
        const y1 = correlations[bestOffset - 1];
        const y2 = correlations[bestOffset];
        const y3 = correlations[bestOffset + 1];
        const d = (y3 - y1) / (2 * (2 * y2 - y1 - y3));
        const refinedOffset = bestOffset + d;
        return { pitch: sr / refinedOffset, corr: bestCorr };
    }

    const pitch = bestOffset > 0 ? (sr / bestOffset) : -1;
    return { pitch, corr: bestCorr };
}

function detectPitchOffline(frame, sr) {
    return detectPitchCore(frame, sr);
}

function detectPitch(timeData) {
    return detectPitchCore(timeData, sampleRate || 44100);
}

async function audioFileToReferenceTrail(file, isRecording = false) {
    if (!audioContext) initAudio();

    const arrayBuffer = await file.arrayBuffer();
    const decoded = await audioContext.decodeAudioData(arrayBuffer);

    const sr = decoded.sampleRate;
    const channelData = decoded.getChannelData(0);

    const frameSize = bufferLength;
    const hop = 256;

    const duration = decoded.duration;
    const desiredPixels = duration * pixelsPerSecond;
    const availablePixels = canvas.width * plotRows;
    const scaleX = desiredPixels > availablePixels ? (availablePixels / desiredPixels) : 1.0;

    const minFreq = parseInt(minFreqSlider.value);
    const maxFreq = parseInt(maxFreqSlider.value);

    const gapMs = 120; // Smaller threshold to catch glottal stops like small 'tsu'
    let lastPlottedMs = null;
    let lastRowIndex = null;
    let stable = 0;
    let smooth = null;

    const trail = [];

    for (let start = 0; start + frameSize <= channelData.length; start += hop) {
        const frame = channelData.subarray(start, start + frameSize);
        const frameRms = computeRms(frame);
        const tSec = start / sr;
        const xTotal = tSec * pixelsPerSecond * scaleX;
        const rowIndex = Math.floor(xTotal / canvas.width);
        const x = xTotal % canvas.width;

        if (rowIndex >= plotRows) break;

        // Low-volume gating: treat as silence
        if (frameRms < rmsSpeechThresholdOff) {
            smooth = null;
            stable = 0;
            continue;
        }

        const { pitch, corr } = detectPitchOffline(frame, sr);
        if (!(pitch > minFreq && pitch < maxFreq) || corr < 0.28) {
            stable = 0;
            continue;
        }

        stable = Math.min(stable + 1, 10);
        if (stable < 2) continue;

        const alpha = 0.35;
        smooth = smooth === null ? pitch : (alpha * pitch + (1 - alpha) * smooth);
        const yRaw = mapPitchToY(smooth, minFreq, maxFreq, rowIndex);
        const rH = rowHeight();
        const rMid = (rowIndex * rH) + (rH / 2);
        
        // Stretch and shift ONLY for user recordings, not professional uploads.
        if (isRecording) {
            const stretch = 2.0;
            const shift = rH * 0.5;
            trail.push({ 
                x, 
                y: rMid + (yRaw - rMid) * stretch - shift, 
                gap: lastPlottedMs != null ? ((tSec * 1000 - lastPlottedMs) > gapMs || rowIndex !== lastRowIndex) : false 
            });
        } else {
            trail.push({ 
                x, 
                y: yRaw, 
                gap: lastPlottedMs != null ? ((tSec * 1000 - lastPlottedMs) > gapMs || rowIndex !== lastRowIndex) : false 
            });
        }
        
        lastPlottedMs = tSec * 1000;
        lastRowIndex = rowIndex;
    }

    return { trail, scaleX };
}

async function startMic() {
    try {
        if (!audioContext) initAudio();
        // Disable browser audio processing to get raw waveform for precise pitch
        mediaStream = await navigator.mediaDevices.getUserMedia({ 
            audio: {
                echoCancellation: false,
                noiseSuppression: false,
                autoGainControl: false
            } 
        });
        microphone = audioContext.createMediaStreamSource(mediaStream);
        microphone.connect(analyser);
        
        startTime = performance.now();
        lastPitchTime = 0;
        
        draw();
        startBtn.disabled = true;
        stopBtn.disabled = false;
    } catch (err) {
        alert('Microphone access denied: ' + err.message);
    }
}

function stopMic() {
    if (microphone) {
        try { microphone.disconnect(); } catch (e) {}
        microphone = null;
    }
    if (mediaStream) {
        mediaStream.getTracks().forEach(t => t.stop());
        mediaStream = null;
    }
    if (animationId) cancelAnimationFrame(animationId);
    smoothedPitch = null;
    pitchDisplay.textContent = '-';
    startBtn.disabled = false;
    stopBtn.disabled = true;
    recording = false;
    recordBtn.textContent = RECORD_ICON;

    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        try { mediaRecorder.stop(); } catch (e) {}
    }
}

function clearTrail() {
    smoothedPitch = null;
    pitchHistory.length = 0;
    startTime = performance.now();
    stablePitchFrames = 0;
    isSpeaking = false;
    lastPlottedTime = 0;
    lastPlottedRow = 0;
    lastVoiceActivityMs = 0;
    lastValidPitchMs = 0;
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    pitchDisplay.textContent = '-';

    // If the mic isn't running, still show the overlay immediately.
    if (!mediaStream) {
        renderIdleFrame();
    }
}

function draw() {
    animationId = requestAnimationFrame(draw);

    const now = performance.now();
    const elapsed = (now - startTime) / 1000;
    const xTotal = elapsed * pixelsPerSecond;
    const currentRow = Math.floor(xTotal / canvas.width);
    const x = xTotal % canvas.width;

    // Auto-clear if end is reached
    if (currentRow >= plotRows) {
        clearTrail();
        return;
    }

    const timeData = new Float32Array(bufferLength);
    analyser.getFloatTimeDomainData(timeData);
    const rms = Math.sqrt(timeData.reduce((sum, v) => sum + v * v, 0) / timeData.length);

    const minFreq = parseInt(minFreqSlider.value);
    const maxFreq = parseInt(maxFreqSlider.value);

    // Always clear and redraw history to keep it crisp
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Visual separation between the 3 plot rows
    drawRowSeparators();

    // Draw reference trail if it exists (respects gaps)
    drawTrail(referenceTrail, { strokeStyle: 'rgba(255, 255, 255, 0.35)', lineWidth: 4, dash: null});

    // Voice activity tracking + speech gating with hysteresis + hangover.
    if (rms > rmsSpeechThresholdOff) {
        lastVoiceActivityMs = now;
    }
    if (!isSpeaking) {
        if (rms > rmsSpeechThreshold) {
            isSpeaking = true;
            smoothedPitch = null;
            stablePitchFrames = 0;
        }
    } else {
        // Don't immediately drop out on brief dips (fast speech / brief consonants).
        if (rms < rmsSpeechThresholdOff && (now - lastVoiceActivityMs) > speechHangoverMs) {
            isSpeaking = false;
            smoothedPitch = null;
            stablePitchFrames = 0;
        }
    }

    if (isSpeaking) {
        const { pitch, corr } = detectPitch(timeData);

        const inRange = pitch > minFreq && pitch < maxFreq;
        const good = inRange && corr >= minPitchCorrelation;

        if (good) {
            lastValidPitchMs = now;

            // Harmonize smoothing alpha (0.35 matches the reference file processing)
            const alpha = 0.35;

            // Clamp jump size to prevent artificial "swoops" or "drops"
            const jumpThreshold = corr > 0.45 ? 180 : 120;
            let pitchForUpdate = pitch;
            if (smoothedPitch !== null) {
                const delta = pitchForUpdate - smoothedPitch;
                if (Math.abs(delta) > jumpThreshold) {
                    pitchForUpdate = smoothedPitch + Math.sign(delta) * jumpThreshold;
                }
            }

            smoothedPitch = smoothedPitch === null ? pitchForUpdate : (alpha * pitchForUpdate + (1 - alpha) * smoothedPitch);
            stablePitchFrames = Math.min(stablePitchFrames + 1, 10);

            const yPitchRaw = mapPitchToY(smoothedPitch, minFreq, maxFreq, currentRow);
            const rH = rowHeight();
            const rMid = (currentRow * rH) + (rH / 2);
            // 2.0x vertical stretch centered on the row, shifted to prevent row overlap
            const yPitch = rMid + (yPitchRaw - rMid) * 2 - (rH * 0.5);

            // Gap detection: base this on when we last *plotted* a point.
            // Reduced to 80ms to catch Japanese small 'tsu' stops.
            const timeGap = (now - lastPlottedTime) > 80 && lastPlottedTime !== 0;
            const rowGap = currentRow !== lastPlottedRow;
            const isGap = timeGap || rowGap;

            // Require at least 2 stable frames on onset to avoid blips/drops.
            const requiredStable = 2;
            if (stablePitchFrames >= requiredStable) {
                const point = { x, y: yPitch, gap: isGap };
                pitchHistory.push(point);
                if (recording) recordedTrail.push(point);
                lastPlottedTime = now;
                lastPlottedRow = currentRow;
            }
            pitchDisplay.textContent = `${pitch.toFixed(0)} Hz`;
        } else {
            // If pitch is briefly unreliable, hold the last pitch for a short window
            // to avoid missing fast syllables.
            if (smoothedPitch !== null && (now - lastValidPitchMs) <= pitchHoldMs) {
                const yPitchRaw = mapPitchToY(smoothedPitch, minFreq, maxFreq, currentRow);
                const rH = rowHeight();
                const rMid = (currentRow * rH) + (rH / 2);
                const yPitch = rMid + (yPitchRaw - rMid) * 2 - (rH * 0.5);

                const timeGap = (now - lastPlottedTime) > 80 && lastPlottedTime !== 0;
                const rowGap = currentRow !== lastPlottedRow;
                const isGap = timeGap || rowGap;
                const point = { x, y: yPitch, gap: isGap };
                pitchHistory.push(point);
                if (recording) recordedTrail.push(point);
                lastPlottedTime = now;
                lastPlottedRow = currentRow;
            } else {
                stablePitchFrames = 0;
            }
        }

        lastPitchTime = now;
    } else {
        // Speech ended, ensure next syllable starts a new segment
        smoothedPitch = null;
        pitchDisplay.textContent = '-';
        // Keep recording state; we just don't add points while silent.
    }

    // Draw the whole history as lines
    drawTrail(pitchHistory, { strokeStyle: '#00f', lineWidth: 6, dash: null });

    // Labels aligned to contiguous overlay trail chunks
    drawChunkLabels();

    // Overlay playhead (if playing)
    if (overlayAudioEl && !overlayAudioEl.paused) {
        const oxTotal = overlayAudioEl.currentTime * pixelsPerSecond * overlayScaleX;
        const oRow = Math.floor(oxTotal / canvas.width);
        const ox = oxTotal % canvas.width;
        if (oRow < plotRows) {
            drawPlayhead(ox, oRow);
        }
    }

    // Timeline indicator for “where we are now” (mic)
    drawPlayhead(x, currentRow);
}

// Recording controls
recordBtn.onclick = async () => {
    recording = !recording;
    if (!recording) {
        recordBtn.textContent = RECORD_ICON;

        if (mediaRecorder && mediaRecorder.state !== 'inactive') {
            try { mediaRecorder.stop(); } catch (e) {}
        }
        return;
    }

    // Starting a recording auto-starts the mic if needed.
    if (!mediaStream) {
        await startMic();
    }

    if (!mediaStream) {
        // user denied or start failed
        recording = false;
        recordBtn.textContent = RECORD_ICON;
        return;
    }

    recordedTrail = [];
    recordBtn.textContent = STOP_ICON;

    recordedAudioChunks = [];
    try {
        mediaRecorder = new MediaRecorder(mediaStream);
    } catch (e) {
        alert('MediaRecorder not supported in this browser.');
        recording = false;
        recordBtn.textContent = RECORD_ICON;
        return;
    }

    mediaRecorder.ondataavailable = (evt) => {
        if (evt.data && evt.data.size > 0) recordedAudioChunks.push(evt.data);
    };
    mediaRecorder.onstop = async () => {
        try {
            const blob = new Blob(recordedAudioChunks, { type: mediaRecorder.mimeType || 'audio/webm' });
            const ext = (mediaRecorder.mimeType && mediaRecorder.mimeType.includes('ogg')) ? 'ogg' : 'webm';
            const file = new File([blob], `recording.${ext}`, { type: blob.type });

            const form = new FormData();
            form.append('file', file);
            const resp = await fetch('/api/audio/save-recording', { method: 'POST', body: form });
            const data = await resp.json();
            if (!resp.ok) throw new Error(data && data.error ? data.error : 'save failed');

            await refreshOverlayList();
        } catch (e) {
            console.error('Failed to save recording:', e);
        }
    };

    mediaRecorder.start();
};

function setExample(type) {
    setOverlayAudio(null);
    overlayScaleX = 1.0;
    if (type === 'hashi_chopsticks') {
        // はし (Chopsticks): High-Low (HL)
        setReference([
            {t: 0.1, f: 200}, {t: 0.2, f: 205}, // Ha
            {t: 0.3, f: 120}, {t: 0.4, f: 115}  // shi
        ]);
    } else if (type === 'hashi_bridge') {
        // はし (Bridge): LH (rising)
        setReference([
            {t: 0.1, f: 120}, {t: 0.2, f: 125}, // Ha
            {t: 0.3, f: 200}, {t: 0.4, f: 205}  // shi
        ]);
    }
    clearTrail();
}

startBtn.onclick = () => startMic();
stopBtn.onclick = () => stopMic();
clearBtn.onclick = () => clearTrail();

// Ensure default icon label
recordBtn.textContent = RECORD_ICON;

audioFileInput.onchange = () => {
    const hasFile = audioFileInput.files && audioFileInput.files.length > 0;
    if (!hasFile) sttText.textContent = '';

    if (hasFile) {
        // Save upload so it appears in the dropdown for future overlays
        (async () => {
            try {
                const file = audioFileInput.files[0];

                // Auto-transcribe when a file is loaded (isUpload = true)
                transcribeFile(file, true);

                const form = new FormData();
                form.append('file', file);
                const resp = await fetch('/api/audio/save-upload', { method: 'POST', body: form });
                const data = await resp.json();
                if (!resp.ok) throw new Error(data && data.error ? data.error : 'save failed');
                await refreshOverlayList();
                
                // Also preview the uploaded file immediately as an unstretched reference
                const { trail, scaleX } = await audioFileToReferenceTrail(file, false);
                referenceTrail = trail;
                overlayScaleX = scaleX;
                referenceChunks = extractTrailChunks(referenceTrail);
                recomputeChunkLabels();
                clearTrail();
                if (!mediaStream) renderIdleFrame();

            } catch (e) {
                console.error('Failed to save upload:', e);
            }
        })();
    }
};

async function refreshOverlayList() {
    try {
        const resp = await fetch('/api/audio/list');
        const data = await resp.json();
        const items = (data && data.items) ? data.items : [];

        overlaySelect.innerHTML = '';

        // // Examples first
        // const ex1 = document.createElement('option');
        // ex1.value = 'example:hashi_chopsticks';
        // ex1.textContent = 'example: はし (箸 / chopsticks)';
        // overlaySelect.appendChild(ex1);

        // const ex2 = document.createElement('option');
        // ex2.value = 'example:hashi_bridge';
        // ex2.textContent = 'example: はし (橋 / bridge)';
        // overlaySelect.appendChild(ex2);

        // const sep = document.createElement('option');
        // sep.value = '';
        // sep.textContent = '────────';
        // overlaySelect.appendChild(sep);

        const empty = document.createElement('option');
        empty.value = '';
        empty.textContent = '(select saved recording/upload)';
        overlaySelect.appendChild(empty);

        for (const item of items) {
            const opt = document.createElement('option');
            opt.value = item.url;
            const label = item.displayName
                ? item.displayName
                : (item.createdAtText ? `${item.originalName} (${item.createdAtText})` : item.originalName);
            opt.textContent = `${item.kind}: ${label}`;
            overlaySelect.appendChild(opt);
        }
    } catch (e) {
        console.error('Failed to refresh overlay list:', e);
    }
}

refreshOverlaysBtn.onclick = () => refreshOverlayList();

applyOverlayBtn.onclick = async () => {
    const url = overlaySelect.value;
    if (!url) return;

    if (url.startsWith('example:')) {
        const which = url.slice('example:'.length);
        setExample(which);
        return;
    }

    try {
        setOverlayAudio(url);
        const resp = await fetch(url);
        if (!resp.ok) throw new Error('Failed to fetch saved audio');
        const blob = await resp.blob();
        const file = new File([blob], 'overlay', { type: blob.type || 'audio/*' });

        // Check if the selected overlay is an "upload" or a "recording"
        const selectedOption = overlaySelect.options[overlaySelect.selectedIndex];
        const isUpload = selectedOption ? selectedOption.textContent.startsWith('upload:') : false;
        const isRecording = selectedOption ? selectedOption.textContent.startsWith('recording:') : false;

        // Auto-transcribe when a saved file is applied. Apply force_break only for uploads.
        transcribeFile(file, isUpload);

        const { trail, scaleX } = await audioFileToReferenceTrail(file, isRecording);
        referenceTrail = trail;
        overlayScaleX = scaleX;
        referenceChunks = extractTrailChunks(referenceTrail);
        recomputeChunkLabels();
        clearTrail();

        // Ensure the overlay is visible immediately even if mic isn't started.
        if (!mediaStream) {
            renderIdleFrame();
        }
    } catch (e) {
        alert('Failed to apply saved overlay: ' + (e && e.message ? e.message : String(e)));
    }
};

playOverlayBtn.onclick = async () => {
    if (!overlayAudioUrl) return;
    try {
        if (!overlayAudioEl) {
            overlayAudioEl = new Audio(overlayAudioUrl);
            overlayAudioEl.onended = () => {
                playOverlayBtn.textContent = '▶';
                if (overlayAnimationId) cancelAnimationFrame(overlayAnimationId);
                renderIdleFrame();
            };
        }
        if (overlayAudioEl.paused) {
            await overlayAudioEl.play();
            playOverlayBtn.textContent = '⏸';
            
            // If mic isn't running, start a dedicated loop for playhead
            if (!mediaStream) {
                const loop = () => {
                    if (overlayAudioEl && !overlayAudioEl.paused) {
                        renderIdleFrame();
                        overlayAnimationId = requestAnimationFrame(loop);
                    }
                };
                overlayAnimationId = requestAnimationFrame(loop);
            }
        } else {
            overlayAudioEl.pause();
            playOverlayBtn.textContent = '▶';
            if (overlayAnimationId) cancelAnimationFrame(overlayAnimationId);
        }
    } catch (e) {
        console.error('Failed to play overlay audio:', e);
    }
};

// initial list
refreshOverlayList();

// Initial empty render
renderIdleFrame();