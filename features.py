"""Pre-processing used by rekordbox's detect_beat models.

This is the feature chain of madmom's RNNDownBeatProcessor, re-implemented in
numpy. rekordbox re-implemented the same chain as Keras layers and embedded the
three logarithmic filterbanks in the SavedModel; this module uses those exact
matrices (extracted by extract_model_data.py).

  audio 44.1 kHz mono
    -> STFT, frame sizes 1024/2048/4096, hop 441 (100 fps), periodic Hann,
       zero padded at the end, Nyquist bin dropped
    -> logarithmic filterbank (3/6/12 bands per octave, 30..17000 Hz)
    -> log10(magnitude + 1)
    -> positive temporal difference (diff_frames 1/1/2) stacked beside the spec
    -> concatenate the three resolutions: 2 * (21 + 45 + 91) = 314 features
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

SAMPLE_RATE = 44100
FPS = 100
HOP = SAMPLE_RATE // FPS  # 441
FRAME_SIZES = (1024, 2048, 4096)
# Number of frames between the current frame and the one the difference is
# calculated against, derived from diff_ratio=0.5 and a Hann window.
DIFF_FRAMES = {1024: 1, 2048: 1, 4096: 2}


def load_audio(path: str | Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Decode any file ffmpeg understands to mono float32."""
    cmd = [
        "ffmpeg", "-v", "error", "-i", str(path), "-ac", "1",
        "-ar", str(sample_rate), "-f", "f32le", "-",
    ]
    raw = subprocess.run(cmd, check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype="<f4").astype(np.float32)


def stft_magnitude(signal: np.ndarray, frame_size: int) -> np.ndarray:
    """tf.signal.stft(frame_length=frame_size, frame_step=441, pad_end=True)
    followed by abs(); the Nyquist bin is dropped (as in madmom)."""
    # STFTLayer pads half a window at both ends so that frame 0 is centred on
    # sample 0 (madmom's default framing), then tf.signal.stft pads the end.
    half = frame_size // 2
    centred = np.zeros(len(signal) + 2 * half, dtype=np.float32)
    centred[half : half + len(signal)] = signal
    count = int(np.ceil((len(centred) - frame_size) / HOP)) + 1
    padded = np.zeros((count - 1) * HOP + frame_size, dtype=np.float32)
    padded[: len(centred)] = centred
    frames = np.lib.stride_tricks.as_strided(
        padded,
        shape=(count, frame_size),
        strides=(HOP * padded.itemsize, padded.itemsize),
        writeable=False,
    )
    # periodic Hann window, as used by tf.signal.hann_window (Keras default)
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(frame_size) / frame_size)
    spectrum = np.fft.rfft(frames * window, axis=1)
    return np.abs(spectrum[:, : frame_size // 2]).astype(np.float32)


def spec_diff(spec: np.ndarray, diff_frames: int) -> np.ndarray:
    """Log spectrogram plus its positive first difference.

    Equivalent to madmom's SpectrogramDifferenceProcessor(diff_ratio=0.5,
    positive_diffs=True, stack_diffs=np.hstack): the first `diff_frames`
    differences are 0.
    """
    diff = np.zeros_like(spec)
    diff[diff_frames:] = spec[diff_frames:] - spec[:-diff_frames]
    np.maximum(diff, 0, out=diff)
    return np.hstack((spec, diff))


def features(signal: np.ndarray, filterbanks: dict[str, np.ndarray]) -> np.ndarray:
    """Return the (num_frames, 314) input of the beat/downbeat network."""
    parts = []
    for frame_size in FRAME_SIZES:
        magnitude = stft_magnitude(signal, frame_size)
        filtered = magnitude @ filterbanks[f"fb{frame_size}"]
        log_spec = np.log10(filtered + 1.0).astype(np.float32)
        parts.append(spec_diff(log_spec, DIFF_FRAMES[frame_size]))
    return np.hstack(parts)


def load_filterbanks(path: str | Path | None = None) -> dict[str, np.ndarray]:
    path = path or Path(__file__).parent / "extracted" / "filterbanks.npz"
    data = np.load(path)
    return {key: data[key] for key in data.files}
