#!/usr/bin/env python3
"""
SongKey - drop audio files, get BPM + musical key (Camelot), see the spectrum.
Songs you choose to keep go into a SQLite library together with the file's
metadata (artist, title, mix/remix, genre, album, year, label, format, size),
and everything in the library can be edited.

Designed for old CPUs (no SSE4/AVX needed): uses only distro packages
  sudo apt install python3-tk python3-numpy python3-scipy python3-matplotlib \
                   python3-pil.imagetk ffmpeg tkdnd

Usage:
  python3 songkey.py                      # GUI
  python3 songkey.py --cli song1.mp3 ...  # analyse in terminal (add --add to store)
  python3 songkey.py --db /mnt/AI/songs.db --bpm-range 75-180
  python3 songkey.py --engine essentia    # optional: use Essentia if installed
"""

import argparse
import csv
import glob
import json
import os
import queue
import re
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
import zlib

import numpy as np
from numpy.lib.stride_tricks import as_strided
from scipy.ndimage import median_filter, uniform_filter1d

APP_NAME = "SongKey"
FROZEN = bool(getattr(sys, "frozen", False))          # running as a PyInstaller .exe
IS_WINDOWS = os.name == "nt"
APP_DIR = os.path.dirname(os.path.abspath(sys.executable if FROZEN else __file__))
# On Windows every ffmpeg/ffprobe/ffplay call would flash a console window without this
NO_WINDOW = {"creationflags": 0x08000000} if IS_WINDOWS else {}


def tool(name):
    """Full path of ffmpeg / ffprobe / ffplay: next to the app first (Windows build), then PATH."""
    exe = name + (".exe" if IS_WINDOWS else "")
    for d in (APP_DIR, os.path.join(APP_DIR, "ffmpeg"), getattr(sys, "_MEIPASS", "")):
        if d and os.path.isfile(os.path.join(d, exe)):
            return os.path.join(d, exe)
    return shutil.which(name) or name


def default_db_path():
    """songs.db next to the app if that folder is writable, else in the user's data folder."""
    if os.access(APP_DIR, os.W_OK):
        return os.path.join(APP_DIR, "songs.db")
    if IS_WINDOWS:
        base = os.environ.get("APPDATA") or os.path.expanduser("~")
    else:
        base = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    return os.path.join(base, "SongKey", "songs.db")
APP_VERSION = "1.0"
GITHUB_URL = "https://github.com/peterpt/Songkey"
SR = 22050                 # analysis sample rate (own engine)
MAX_SECONDS = 20 * 60      # analyse at most the first 20 minutes
SPEC_ROWS = 256
SPEC_FMIN = 30.0
SPEC_MAX_COLS = 1500

AUDIO_EXT = {
    ".mp3", ".wav", ".flac", ".ogg", ".oga", ".opus", ".m4a", ".aac", ".wma",
    ".aif", ".aiff", ".aifc", ".alac", ".mp4", ".webm", ".mka", ".ape", ".wv",
    ".mpc", ".ac3", ".dts", ".amr", ".3gp", ".caf", ".au", ".mp2", ".mkv", ".mov",
}

# ---------------------------------------------------------------------------
# Music theory helpers
# ---------------------------------------------------------------------------
PC_NAMES_MAJ = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
PC_NAMES_MIN = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "G#", "A", "Bb", "B"]
NOTE_TO_PC = {"C": 0, "B#": 0, "C#": 1, "DB": 1, "D": 2, "D#": 3, "EB": 3, "E": 4,
              "FB": 4, "F": 5, "E#": 5, "F#": 6, "GB": 6, "G": 7, "G#": 8, "AB": 8,
              "A": 9, "A#": 10, "BB": 10, "B": 11, "CB": 11}

# Key profiles (index 0 = tonic)
PROFILES = {
    "krumhansl": (
        [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
        [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]),
    "temperley": (
        [5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5, 2.0, 3.5, 1.5, 4.0],
        [5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.0, 4.5, 3.5, 2.0, 1.5, 4.0]),
}


def key_name(pc, minor):
    return (PC_NAMES_MIN[pc] + "m") if minor else PC_NAMES_MAJ[pc]


def camelot(pc, minor):
    n = (7 * pc + (5 if minor else 8)) % 12
    return "%d%s" % (n if n else 12, "A" if minor else "B")


def camelot_split(code):
    try:
        return int(code[:-1]), code[-1].upper()
    except (ValueError, IndexError, TypeError):
        return None, None


def camelot_compatible(code):
    """Same key, one step around the wheel, and relative major/minor."""
    n, l = camelot_split(code)
    if n is None:
        return []
    up = n % 12 + 1
    down = (n - 2) % 12 + 1
    other = "B" if l == "A" else "A"
    return ["%d%s" % (n, l), "%d%s" % (down, l), "%d%s" % (up, l), "%d%s" % (n, other)]


def camelot_sort_key(code):
    n, l = camelot_split(code or "")
    return (n or 99, l or "Z")


def camelot_color(code):
    """Colour per wheel position (like the coloured Camelot wheel)."""
    import colorsys
    n, l = camelot_split(code or "")
    if n is None:
        return "#555555"
    h = ((n - 1) / 12.0 + 0.40) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.40 if l == "B" else 0.60, 1.0 if l == "B" else 0.92)
    return "#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255))


def fmt_time(sec):
    sec = int(round(sec or 0))
    return "%d:%02d" % (sec // 60, sec % 60)


# ---------------------------------------------------------------------------
# Decoding (ffmpeg handles every format)
# ---------------------------------------------------------------------------
def probe(path):
    """Duration and tags via ffprobe. Never fails hard."""
    info = {"duration": 0.0, "title": "", "artist": "", "album": "", "channels": 2}
    try:
        p = subprocess.run(
            [tool("ffprobe"), "-v", "error", "-show_entries",
             "format=duration:format_tags:stream=channels,codec_type:stream_tags", "-of", "json", path],
            capture_output=True, timeout=60, **NO_WINDOW)
        data = json.loads(p.stdout.decode("utf-8", "replace") or "{}")
        fmt = data.get("format", {})
        info["duration"] = float(fmt.get("duration") or 0)
        tags = {}
        for s in data.get("streams", []):
            for k, v in (s.get("tags") or {}).items():
                tags.setdefault(k.lower(), v)
        for k, v in (fmt.get("tags") or {}).items():
            tags[k.lower()] = v
        for k in ("title", "artist", "album"):
            info[k] = str(tags.get(k, "")).strip()
        for st_ in data.get("streams", []):
            if st_.get("codec_type") == "audio" and st_.get("channels"):
                info["channels"] = int(st_["channels"])
                break
    except Exception:
        pass
    return info


def decode(path, sr=SR, max_seconds=MAX_SECONDS):
    cmd = [tool("ffmpeg"), "-v", "error", "-nostdin", "-i", path, "-vn", "-sn", "-dn",
           "-t", str(max_seconds), "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    try:
        p = subprocess.run(cmd, capture_output=True, **NO_WINDOW)
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found - install it: sudo apt install ffmpeg")
    if not p.stdout:
        err = p.stderr.decode("utf-8", "replace").strip().splitlines()
        raise RuntimeError("could not decode audio: " + (err[-1] if err else "unknown format"))
    x = np.frombuffer(p.stdout, dtype=np.float32).copy()
    if len(x) < sr * 2:
        raise RuntimeError("audio is shorter than 2 seconds")
    return x


# ---------------------------------------------------------------------------
# Waveform data (Audacity-style: peaks + loudness per 10 ms, raw samples when zoomed in)
# ---------------------------------------------------------------------------
WAVE_SR = 44100
PEAK_RATE = 100                      # blocks per second stored in the database
PEAK_BLOCK = WAVE_SR // PEAK_RATE    # 441 samples


def compute_peaks(path, channels=2, max_seconds=MAX_SECONDS):
    """min / max / rms per 10 ms block for each channel -> int8 array (blocks, ch, 3)."""
    ch = 1 if channels == 1 else 2
    cmd = [tool("ffmpeg"), "-v", "error", "-nostdin", "-i", path, "-vn", "-sn", "-dn",
           "-t", str(max_seconds), "-ac", str(ch), "-ar", str(WAVE_SR), "-f", "s16le", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, **NO_WINDOW)
    blk_bytes = PEAK_BLOCK * ch * 2
    out, rest = [], b""
    while True:
        data = p.stdout.read(blk_bytes * 2000)
        if not data:
            break
        data = rest + data
        nfull = len(data) // blk_bytes
        rest = data[nfull * blk_bytes:]
        if nfull:
            a = np.frombuffer(data[:nfull * blk_bytes], dtype=np.int16).reshape(nfull, PEAK_BLOCK, ch)
            a = a.astype(np.float32) / 32768.0
            out.append(np.stack([a.min(axis=1), a.max(axis=1), np.sqrt((a * a).mean(axis=1))], axis=2))
    p.wait()
    if len(rest) >= ch * 2:
        a = np.frombuffer(rest[:len(rest) // (ch * 2) * ch * 2], dtype=np.int16).reshape(-1, ch)
        a = a.astype(np.float32) / 32768.0
        out.append(np.stack([a.min(axis=0), a.max(axis=0), np.sqrt((a * a).mean(axis=0))], axis=1)[None])
    if not out:
        return None
    arr = np.concatenate(out, axis=0)
    return np.clip(np.round(arr * 127), -127, 127).astype(np.int8)


def load_segment(path, start, dur, channels=2):
    """Raw samples (n, ch) float32 for a short stretch of the file, for deep zoom."""
    ch = 1 if channels == 1 else 2
    cmd = [tool("ffmpeg"), "-v", "error", "-nostdin", "-ss", "%.4f" % max(0.0, start), "-i", path,
           "-t", "%.4f" % max(0.01, dur), "-vn", "-sn", "-dn", "-ac", str(ch), "-ar", str(WAVE_SR),
           "-f", "s16le", "-"]
    try:
        data = subprocess.run(cmd, capture_output=True, timeout=30, **NO_WINDOW).stdout
    except Exception:
        return None
    n = len(data) // (2 * ch)
    if n == 0:
        return None
    return np.frombuffer(data[:n * 2 * ch], dtype=np.int16).reshape(n, ch).astype(np.float32) / 32768.0


def fmt_ms(t):
    t = max(0.0, float(t or 0))
    return "%d:%06.3f" % (int(t // 60), t % 60)


def fmt_clock(t, span):
    if span < 8:
        return fmt_ms(t)
    if span < 90:
        t = max(0.0, t)
        return "%d:%04.1f" % (int(t // 60), t % 60)
    return fmt_time(t)


# ---------------------------------------------------------------------------
# DSP building blocks
# ---------------------------------------------------------------------------
def tri_bank(n_bins, bin_hz, centers, min_half_bins=1.0):
    """Triangular filterbank (n_bins x len(centers)), each filter sums to 1."""
    centers = np.asarray(centers, dtype=np.float64)
    bf = np.arange(n_bins) * bin_hz
    W = np.zeros((n_bins, len(centers)), np.float32)
    ratio_lo = centers[1] / centers[0]
    ratio_hi = centers[-1] / centers[-2]
    for k, c in enumerate(centers):
        lo = centers[k - 1] if k > 0 else c / ratio_lo
        hi = centers[k + 1] if k < len(centers) - 1 else c * ratio_hi
        lo = min(lo, c - min_half_bins * bin_hz)
        hi = max(hi, c + min_half_bins * bin_hz)
        w = np.minimum((bf - lo) / (c - lo), (hi - bf) / (hi - c))
        w = np.clip(w, 0.0, None)
        s = w.sum()
        if s > 0:
            W[:, k] = w / s
    return W


def stft_bank(x, n_fft, hop, W, power=False):
    """STFT magnitude (or power) projected straight onto filterbank W.
    Returns frames x bands. Processed in chunks to keep memory low."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    n = 1 + (len(x) - n_fft) // hop
    frames = as_strided(x, shape=(n, n_fft), strides=(x.strides[0] * hop, x.strides[0]))
    win = np.hanning(n_fft).astype(np.float32)
    nb = W.shape[0]
    out = np.empty((n, W.shape[1]), np.float32)
    chunk = max(16, (1 << 22) // n_fft)
    for i in range(0, n, chunk):
        m = np.abs(np.fft.rfft(frames[i:i + chunk] * win, axis=1))[:, :nb].astype(np.float32)
        if power:
            m *= m
        out[i:i + chunk] = m @ W
    return out


# ---------------------------------------------------------------------------
# Spectrum (for display)
# ---------------------------------------------------------------------------
def spectrum_freqs(sr=SR):
    return np.geomspace(SPEC_FMIN, sr / 2 * 0.98, SPEC_ROWS)


def make_spectrogram(x, sr=SR):
    n_fft, hop = 2048, 1024
    centers = spectrum_freqs(sr)
    W = tri_bank(n_fft // 2 + 1, sr / n_fft, centers, 0.5)
    P = stft_bank(x, n_fft, hop, W, power=True)            # frames x rows
    if P.shape[0] > SPEC_MAX_COLS:                        # average columns down
        g = int(np.ceil(P.shape[0] / SPEC_MAX_COLS))
        m = (P.shape[0] // g) * g
        P = P[:m].reshape(-1, g, P.shape[1]).mean(axis=1)
    ref = float(P.max()) + 1e-12
    db = np.clip(10 * np.log10(P / ref + 1e-12), -90, 0)
    img = np.round((db + 90) / 90 * 255).astype(np.uint8).T.copy()   # rows x cols
    avg = np.clip(10 * np.log10(P.mean(axis=0) / ref + 1e-12), -120, 0).astype(np.float32)
    return img, avg


def spec_to_blob(img):
    return zlib.compress(img.tobytes(), 6)


def blob_to_spec(blob, rows, cols):
    return np.frombuffer(zlib.decompress(blob), dtype=np.uint8).reshape(rows, cols)


# ---------------------------------------------------------------------------
# Key detection (own engine)
# ---------------------------------------------------------------------------
def compute_chromas(x, sr=SR):
    """12-bin pitch-class profiles (whole range + bass line only), with tuning
    correction and drum suppression. Returns (chroma, bass_chroma), each 0..1."""
    n_fft, hop = 8192, 2048
    bin_hz = sr / n_fft
    midi = np.arange(33 * 3, 100 * 3) / 3.0        # A1 (55 Hz) .. ~D#7, 1/3 semitone steps
    centers = 440.0 * 2 ** ((midi - 69) / 12.0)
    n_bins = int(centers[-1] * 1.1 / bin_hz) + 2
    W = tri_bank(n_bins, bin_hz, centers, 0.75)
    F = stft_bank(x, n_fft, hop, W)                 # frames x bands
    # Harmonic part: median along time removes short drum hits
    if F.shape[0] >= 9:
        F = median_filter(F, size=(9, 1), mode="nearest")
    # Ignore near-silent frames
    e = F.sum(axis=1)
    keep = e > (np.percentile(e, 95) * 0.02)
    if keep.sum() > 10:
        F = F[keep]
    F = np.log1p(F / (F.mean() + 1e-12) * 10.0)
    fine = np.round(midi * 3).astype(int) % 36      # fine//3 = pitch class (C = 0)

    def fold(v, idx, t=None):
        p36 = np.bincount(fine[idx], weights=v, minlength=36)
        if t is None:
            t = int(np.argmax(p36.reshape(12, 3).sum(axis=0)))   # tuning offset
        p36 = np.roll(p36, -1 if t == 1 else (1 if t == 2 else 0))
        c = np.array([p36[3 * i] + 0.5 * p36[3 * i + 1] + 0.5 * p36[(3 * i - 1) % 36] for i in range(12)])
        c -= c.min()
        return c / (c.max() + 1e-12), t
    allb = np.ones(len(midi), bool)
    chroma, t = fold(F.sum(axis=0), allb)
    bass = midi < 57                                # up to ~A3 (220 Hz): the bass line
    bass_chroma, _ = fold(F[:, bass].sum(axis=0), bass, t)
    return chroma, bass_chroma


def compute_chroma(x, sr=SR):
    return compute_chromas(x, sr)[0]


# Modal check: the same notes can have a different home note.
#   B-flat-major notes resting on C  = C Dorian      -> C minor
#   B-flat-major notes resting on F  = F Mixolydian  -> F major
# Decided from home-note evidence: what the bass leans on, how strong the note and its
# fifth are, and (for minor) a leading tone heard with it. Tuned on 84 test songs.
KEY_EV_BASS, KEY_EV_NOTE, KEY_EV_FIFTH, KEY_EV_LEADING = 0.6, 1.0, 0.6, 0.2
KEY_SWITCH_MARGIN = 0.20


def key_scores(chroma):
    """Score of all 24 keys -> array (2, 12): [0] = major, [1] = minor."""
    scores = np.zeros((2, 12))
    for maj, mnr in PROFILES.values():
        for mode, prof in ((0, maj), (1, mnr)):
            prof = np.asarray(prof, float)
            for pc in range(12):
                scores[mode, pc] += np.corrcoef(chroma, np.roll(prof, pc))[0, 1]
    return scores / len(PROFILES)


def _home_evidence(chroma, bass, t, minor):
    ev = KEY_EV_NOTE * chroma[t] + KEY_EV_FIFTH * (bass[(t + 7) % 12] + chroma[(t + 7) % 12]) / 2
    ev += KEY_EV_BASS * bass[t]
    if minor:
        ev += KEY_EV_LEADING * min(chroma[(t + 11) % 12], chroma[t])
    return ev


def key_from_chroma(chroma, bass=None):
    """Returns (pc, minor, confidence 0..1, alternative key text or '')."""
    scores = key_scores(chroma)
    flat = scores.flatten()
    order = np.argsort(flat)[::-1]
    best, second = flat[order[0]], flat[order[1]]
    mode, pc = divmod(int(order[0]), 12)
    margin = best - second
    conf = float(np.clip((best - 0.35) / 0.5, 0, 1) * 0.4 + np.clip(margin / 0.10, 0, 1) * 0.6)
    alt_key = None
    if margin < 0.05:
        m2, p2 = divmod(int(order[1]), 12)
        alt_key = (p2, m2)
        conf = min(conf, 0.30 if margin < 0.025 else 0.60)
    if bass is not None:
        # same notes, different home note? (Dorian / Mixolydian)
        if mode == 0:
            cands = [((pc + 2) % 12, 1), ((pc + 7) % 12, 0)]
        else:
            cands = [((pc + 5) % 12, 1), ((pc + 10) % 12, 0)]
        top = (pc, mode)
        top_ev = _home_evidence(chroma, bass, pc, mode == 1)
        close = None
        for t, m in cands:
            ev = _home_evidence(chroma, bass, t, m == 1)
            if ev > top_ev + KEY_SWITCH_MARGIN:
                alt_key = top                      # switched: the note-set winner becomes the alternative
                top, top_ev = (t, m), ev
                conf = min(conf, 0.60)
            elif ev > top_ev - KEY_SWITCH_MARGIN / 2 and close is None:
                close = (t, m)
        if close is not None and top == (pc, mode):
            alt_key = close                        # nearly as good a home note: say so
            conf = min(conf, 0.45)
        pc, mode = top
    alt = "%s (%s)" % (key_name(alt_key[0], alt_key[1] == 1), camelot(alt_key[0], alt_key[1] == 1)) if alt_key else ""
    return pc, bool(mode), conf, alt


def detect_key(x, sr=SR):
    chroma, bass = compute_chromas(x, sr)
    return key_from_chroma(chroma, bass)


# ---------------------------------------------------------------------------
# BPM detection (own engine)
# ---------------------------------------------------------------------------
def onset_envelope(x, sr=SR):
    o, fps, _b, _c = onset_bands(x, sr)
    return o, fps


def onset_bands(x, sr=SR):
    """Onset strength envelope + the band energies it was made from."""
    n_fft, hop = 1024, 256
    fps = sr / hop
    centers = np.geomspace(40, 8000, 36)
    W = tri_bank(n_fft // 2 + 1, sr / n_fft, centers, 1.0)
    B = stft_bank(x, n_fft, hop, W)
    L = np.log1p(B / (B.mean() + 1e-12) * 10.0)
    flux = np.maximum(0.0, np.diff(L, axis=0)).mean(axis=1)
    trend = uniform_filter1d(flux, size=max(3, int(fps * 0.5)))
    o = np.maximum(0.0, flux - trend)
    return o.astype(np.float64), fps, B, centers


def _interp(arr, pos):
    return np.interp(pos, np.arange(len(arr)), arr)


def estimate_bpm(o, fps, lo=75.0, hi=180.0):
    """Autocorrelation tempo, evaluated on a fine (sub-frame) lag grid.
    A beat period L is scored by how strongly the music repeats after 1, 2, 3 and 4
    beats; this rejects the classic 2:3 / 3:4 mistakes (e.g. 92 instead of 138)
    caused by dotted-note delays and syncopated basslines.
    Returns (bpm, confidence 0..1)."""
    o = o - o.mean()
    n = len(o)
    if n < 64:
        return 0.0, 0.0
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    ac = np.fft.irfft(np.abs(np.fft.rfft(o, nfft)) ** 2)[:n]
    if ac[0] <= 0:
        return 0.0, 0.0
    ac = ac / ac[0]
    # local maximum within +-1 frame: makes every lag robust to sub-frame offsets
    acm = np.maximum(ac, np.maximum(np.roll(ac, 1), np.roll(ac, -1)))
    acm[0] = ac[0]
    lag_lo = 60.0 * fps / hi
    lag_hi = min(60.0 * fps / lo, (n - 3) / 4.0)
    if lag_hi <= lag_lo:
        return 0.0, 0.0
    lags = np.arange(lag_lo, lag_hi, 0.2)
    weights = ((1, 1.0), (2, 0.6), (3, 0.5), (4, 0.4))
    score = sum(w * _interp(acm, k * lags) for k, w in weights) / sum(w for _, w in weights)
    bpms = 60.0 * fps / lags
    prior = np.exp(-0.5 * (np.log2(bpms / 120.0) / 1.0) ** 2)
    total = score * (0.6 + 0.4 * prior)
    L0 = float(lags[int(np.argmax(total))])
    # Refine using the peak a few beats away for sub-frame precision
    Lref = L0
    for m in (8, 4, 2, 1):
        c = m * L0
        a, b = int(c - m - 1), int(c + m + 2)
        if a < 1 or b >= n - 1:
            continue
        i = a + int(np.argmax(ac[a:b]))
        y0, y1, y2 = ac[i - 1], ac[i], ac[i + 1]
        den = y0 - 2 * y1 + y2
        off = 0.5 * (y0 - y2) / den if den != 0 else 0.0
        Lref = (i + float(np.clip(off, -0.5, 0.5))) / m
        break
    bpm = 60.0 * fps / Lref
    while bpm < lo:
        bpm *= 2
    while bpm > hi:
        bpm /= 2
    # Snap to a whole number when very close (most produced music is integer BPM)
    if abs(bpm - round(bpm)) < 0.10:
        bpm = float(round(bpm))
    conf = float(np.clip(_interp(acm, np.array([L0]))[0] / 0.35, 0, 1))
    return bpm, conf


# ---------------------------------------------------------------------------
# DJ mix points
# ---------------------------------------------------------------------------
# kind -> (label, colour, line style, plain meaning)
CUE_STYLE = {
    "in":    ("IN",    "#40c4ff", "-",  "start mixing this track in here"),
    "drop":  ("DROP",  "#40c4ff", "--", "full energy starts - the incoming track should be fully in"),
    "break": ("BREAK", "#ffab40", "--", "energy falls (breakdown) - an early place to mix out"),
    "out":   ("OUT",   "#ffab40", "-",  "start mixing the next track here (outro)"),
    "end":   ("END",   "#bdbdbd", ":",  "last beat"),
    "cue":   ("CUE",   "#ffffff", ":",  "your own marker"),
}


def find_cues(o, B, centers, fps, bpm):
    """Beat grid -> bars -> energy sections -> mix points.
    Returns (cues, first_downbeat_seconds). Each cue: {t, kind, label, auto}."""
    if not bpm or len(o) < fps * 20:
        return [], 0.0
    P = 60.0 * fps / bpm                      # frames per beat
    n = min(len(o), len(B) - 1)
    nbeats = int((n - 1) / P) - 1
    if nbeats < 64:
        return [], 0.0
    # 1) beat phase: where the onsets line up best with a grid of period P
    sm = uniform_filter1d(o[:n], 3)
    k = np.arange(nbeats) * P
    phases = np.arange(0.0, P, 0.25)
    sal = [_interp(sm, ph + k).sum() for ph in phases]
    ph = float(phases[int(np.argmax(sal))])
    beats = ph + np.arange(int((n - 1 - ph) / P)) * P
    idx = beats.astype(int)

    # 2) energy per beat in 3 bands (bass / mids / highs): average power, then log
    Bp = B[1:n + 1] ** 2
    groups = [centers < 150, (centers >= 150) & (centers < 2500), centers >= 2500]

    def per_beat(e):
        cs = np.concatenate([[0.0], np.cumsum(e)])
        b = np.append(idx[1:], len(e))
        return np.log((cs[b] - cs[idx]) / np.maximum(1, b - idx) + 1e-12)
    fbands = [per_beat(Bp[:, g].sum(axis=1)) for g in groups]
    fa = per_beat(Bp.sum(axis=1))
    fb = sum(fbands)
    nb = len(fb)

    # 3) downbeat: the beat (of 4) where 4-beat blocks change the most
    loud = fa > fa.max() - 9.2               # ignore silence before/after the music
    nov = np.zeros(nb)
    for i in range(4, nb - 4):
        if loud[i - 4:i + 4].all():
            nov[i] = abs(fb[i:i + 4].mean() - fb[i - 4:i].mean())
    r = int(np.argmax([(nov[q::4] ** 2).sum() for q in range(4)]))   # big changes decide
    bar_beats = np.arange(r, nb - 3, 4)
    nbar = len(bar_beats)
    Ea = np.array([fa[j:j + 4].mean() for j in bar_beats])
    present = np.where(Ea > Ea.max() - 9.2)[0]     # within ~40 dB of the loudest bar
    if len(present) < 16:
        return [], 0.0
    first, last = int(present[0]), int(present[-1])
    # "how full is the music": each band scaled 0..1 over the track, then averaged
    En = np.zeros(nbar)
    for fbd in fbands:
        Eb = np.array([fbd[j:j + 4].mean() for j in bar_beats])
        seg = Eb[first:last + 1]
        lo_, hi_ = np.percentile(seg, 3), np.percentile(seg, 97)
        En += np.clip((Eb - lo_) / (hi_ - lo_ + 1e-9), 0, 1)
    En /= len(fbands)

    # 4) section changes: compare the 8 bars after each bar line with the 8 before
    nov_b = np.zeros(nbar)
    for b in range(first + 4, last - 2):
        w = min(8, b - first, last + 1 - b)
        if w >= 4:
            nov_b[b] = En[b:b + w].mean() - En[b - w:b].mean()
    cand = []
    for b in range(nbar):
        v = abs(nov_b[b])
        if v >= 0.13 and v >= np.abs(nov_b[max(0, b - 4):b + 5]).max() - 1e-9:
            cand.append(b)
    keep = []
    for b in sorted(cand, key=lambda b: -abs(nov_b[b])):
        if all(abs(b - q) >= 8 for q in keep):
            keep.append(b)
    snapped = []
    for b in sorted(keep):          # sections almost always start on 4-bar lines
        rel = b - first
        s4 = int(round(rel / 4.0)) * 4
        snapped.append(first + s4 if abs(s4 - rel) <= 1 else b)

    def T(b):
        return round(float(beats[bar_beats[b]] / fps), 3)

    def after(b, w=8):
        return float(En[b:min(last + 1, b + w)].mean())
    span = max(1, last - first + 1)
    cues = [{"t": T(first), "kind": "in", "label": "IN", "auto": True}]
    falls = [b for b in snapped if after(b) < after(max(first, b - 8)) - 0.12]
    rises = [b for b in snapped if after(b) > after(max(first, b - 8)) + 0.12]
    outro = next((b for b in reversed(falls) if (b - first) / span >= 0.6), None)
    if outro is None and span > 64:
        outro = last + 1 - 32
    peak = float(np.percentile(En[first:last + 1], 90))
    for i, b in enumerate(rises):
        nxt = rises[i + 1] if i + 1 < len(rises) else None
        if nxt is not None and not any(b < f < nxt for f in falls):
            continue        # energy keeps rising after this (a build-up): the next rise is the drop
        if after(b) >= peak - 0.10 and b != first:
            cues.append({"t": T(b), "kind": "drop", "label": "DROP", "auto": True})
    for b in falls:
        if b != outro and (outro is None or b < outro) and after(b) <= 0.55:
            cues.append({"t": T(b), "kind": "break", "label": "BREAK", "auto": True})
    if outro is not None:
        cues.append({"t": T(outro), "kind": "out", "label": "OUT", "auto": True})
    end_t = round(float((beats[bar_beats[last]] + 4 * P) / fps), 3)
    cues.append({"t": end_t, "kind": "end", "label": "END", "auto": True})
    cues.sort(key=lambda c: c["t"])
    return cues, T(first)


# ---------------------------------------------------------------------------
# Optional Essentia engine
# ---------------------------------------------------------------------------
def essentia_available():
    """Probe in a subprocess so an 'Illegal instruction' crash can't kill us."""
    if FROZEN:
        return False, "Essentia is not included in this packaged version of SongKey"
    try:
        test = ("import numpy as np, essentia.standard as es\n"
                "x = (np.random.rand(44100 * 6).astype('float32') - 0.5) * 0.2\n"
                "es.KeyExtractor(profileType='bgate')(x)\n"
                "es.RhythmExtractor2013(method='multifeature')(x)\n")
        p = subprocess.run([sys.executable, "-c", test], capture_output=True, timeout=300, **NO_WINDOW)
    except Exception as e:
        return False, str(e)
    if p.returncode == 0:
        return True, "ok"
    if p.returncode == -4:
        return False, "Essentia crashed with 'Illegal instruction' (CPU too old for this build)"
    return False, (p.stderr.decode("utf-8", "replace").strip().splitlines() or ["not installed"])[-1]


def essentia_analyse(path):
    import essentia
    import essentia.standard as es
    try:
        essentia.log.infoActive = False
        essentia.log.warningActive = False
    except Exception:
        pass
    x = decode(path, sr=44100)
    k, scale, strength = es.KeyExtractor(profileType="bgate", sampleRate=44100)(x)
    bpm, _beats, bconf, _e, _i = es.RhythmExtractor2013(method="multifeature")(x)
    pc = NOTE_TO_PC.get(k.upper(), 0)
    return pc, scale == "minor", float(strength), float(bpm), float(min(1.0, bconf / 3.5))


# ---------------------------------------------------------------------------
# Full analysis of one file (does NOT save anything)
# ---------------------------------------------------------------------------
def analyse(path, engine="builtin", bpm_range=(75.0, 180.0), on_spectrum=None, on_status=None):
    def status(msg):
        if on_status:
            on_status(msg)

    t0 = time.time()
    st = os.stat(path)
    meta = probe(path)
    fa, ft = parse_filename(path)
    title = meta["title"] or ft
    artist = meta["artist"] or fa
    status("Decoding")
    x = decode(path)
    duration = meta["duration"] or len(x) / SR

    status("Drawing waveform")
    peaks = compute_peaks(path, meta.get("channels", 2))
    status("Drawing spectrum")
    img, avg = make_spectrogram(x)
    if on_spectrum:
        on_spectrum({"path": os.path.abspath(path), "spec": img, "avg": avg, "peaks": peaks,
                     "duration": duration, "title": title, "artist": artist})

    status("Finding the key")
    chroma, bass_chroma = compute_chromas(x)
    key_alt = ""
    o, fps, bands, bcenters = onset_bands(x)
    if engine == "essentia":
        status("Essentia: key + tempo")
        pc, minor, kconf, bpm, bconf = essentia_analyse(path)
        while bpm and bpm < bpm_range[0]:
            bpm *= 2
        while bpm and bpm > bpm_range[1]:
            bpm /= 2
        if bpm and abs(bpm - round(bpm)) < 0.10:
            bpm = float(round(bpm))
    else:
        pc, minor, kconf, key_alt = key_from_chroma(chroma, bass_chroma)
        status("Finding the tempo")
        bpm, bconf = estimate_bpm(o, fps, *bpm_range)
    status("Finding mix points")
    cues, beat_offset = find_cues(o, bands, bcenters, fps, bpm)

    return {
        "path": os.path.abspath(path),
        "filename": os.path.basename(path),
        "title": title,
        "artist": artist,
        "duration": float(duration),
        "size": st.st_size,
        "mtime": st.st_mtime,
        "bpm": round(float(bpm), 2),
        "bpm_conf": round(bconf, 3),
        "key_pc": pc,
        "key_minor": int(minor),
        "key_name": key_name(pc, minor),
        "camelot": camelot(pc, minor),
        "key_conf": round(kconf, 3),
        "key_alt": key_alt,
        "chroma": [round(float(c), 4) for c in chroma],
        "cues": cues,
        "beat_offset": beat_offset,
        "peaks": peaks,
        "spec": img,
        "avg": avg,
        "engine": engine,
        "analyzed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "secs": round(time.time() - t0, 1),
    }


# ---------------------------------------------------------------------------
# Metadata (read when the user adds a song to the library)
# ---------------------------------------------------------------------------
MIX_WORDS = (r"remix|rmx|mix|edit|re-?edit|rework|bootleg|flip|refix|remake|vip|dub|version|"
             r"remaster(?:ed)?|extended|radio|club|instrumental|acapella|a cappella|live|"
             r"cover|mash-?up|reprise")
STRONG_REMIX = r"remix|rmx|re-?edit|rework|bootleg|flip|refix|remake|vip|mash-?up"
WEAK_REMIX = r"edit|mix|dub"
GENERIC_WORDS = {"original", "extended", "radio", "club", "album", "single", "vocal", "instrumental",
                 "official", "version", "short", "long", "clean", "explicit", "main", "full",
                 "length", "intro", "outro", "the", "a", "an", "&", "and", "-", "new", "special",
                 "remastered", "remaster", "dirty", "uk", "us", "12\"", "7\"", "12''", "7''"}
FEAT_RE = re.compile(r"^(?:feat\.?|ft\.?|featuring)\s+", re.I)


def parse_filename(path):
    """'01 - Artist - Title (Remix).mp3' -> ('Artist', 'Title (Remix)')"""
    stem = os.path.splitext(os.path.basename(path))[0].replace("_", " ")
    stem = re.sub(r"^\s*\d{1,3}\s*[-.)]\s+", "", stem)
    if " - " in stem:
        a, t = stem.split(" - ", 1)
        return a.strip(), t.strip()
    return "", stem.strip()


def parse_title(title):
    """Split a title into (base title, mix/version text, remixer, is_remix, featuring).
    'Song (feat. X) (Armin van Buuren Remix)' -> ('Song', 'Armin van Buuren Remix', 'Armin van Buuren', True, 'X')
    """
    title = title or ""
    base, feat, mixes = title, "", []
    for m in re.finditer(r"[\(\[]([^\(\)\[\]]+)[\)\]]", title):
        inner = m.group(1).strip()
        if FEAT_RE.match(inner):
            feat = FEAT_RE.sub("", inner).strip()
            base = base.replace(m.group(0), " ")
        elif re.search(r"\b(?:%s)\b" % MIX_WORDS, inner, re.I):
            mixes.append(inner)
            base = base.replace(m.group(0), " ")
    if not mixes:   # 'Song - Extended Mix'
        m = re.search(r"\s+-\s+([^-]*\b(?:%s)\b[^-]*)$" % MIX_WORDS, base, re.I)
        if m:
            mixes.append(m.group(1).strip())
            base = base[:m.start()]
    m = re.search(r"\s+(?:feat\.?|ft\.?|featuring)\s+(.+)$", base, re.I)
    if m and not feat:
        feat = m.group(1).strip()
        base = base[:m.start()]
    base = re.sub(r"\s{2,}", " ", base).strip(" -")

    remixer, is_remix = "", False
    for mx in mixes:
        strong = re.search(r"\b(?:%s)\b" % STRONG_REMIX, mx, re.I)
        weak = re.search(r"\b(?:%s)\b" % WEAK_REMIX, mx, re.I)
        hit = strong or weak
        if not hit:
            continue
        name_part = mx[:hit.start()]
        words = [w for w in name_part.split()
                 if w.lower().strip(".,") not in GENERIC_WORDS and not re.fullmatch(r"'?\d{2,4}", w)]
        name = " ".join(words).strip(" -'")
        name = re.sub(r"'s$", "", name)
        if strong or name:
            is_remix = True
            remixer = name
            break
    return base or title.strip(), " / ".join(mixes), remixer, is_remix, feat


def read_metadata(path):
    """Everything we can learn from the file itself: tags + format."""
    st = os.stat(path)
    md = {"path": os.path.abspath(path), "filename": os.path.basename(path),
          "size": st.st_size, "mtime": st.st_mtime}
    data = {}
    try:
        p = subprocess.run([tool("ffprobe"), "-v", "error", "-show_format", "-show_streams",
                            "-of", "json", path], capture_output=True, timeout=60, **NO_WINDOW)
        data = json.loads(p.stdout.decode("utf-8", "replace") or "{}")
    except Exception:
        pass
    fmt = data.get("format", {}) or {}
    streams = data.get("streams", []) or []
    a = next((s for s in streams if s.get("codec_type") == "audio"), {})
    tags = {}
    for s in streams:
        for k, v in (s.get("tags") or {}).items():
            tags.setdefault(k.lower(), v)
    for k, v in (fmt.get("tags") or {}).items():
        tags[k.lower()] = v

    def tag(*names):
        for n in names:
            v = tags.get(n)
            if v not in (None, ""):
                return str(v).strip()
        return ""

    fa, ft = parse_filename(path)
    raw_title = tag("title") or ft
    base, mix, remixer, is_remix, feat = parse_title(raw_title)
    year = tag("date", "year", "tdrc", "originaldate", "tory", "tyer")
    m = re.search(r"\d{4}", year)
    ext = os.path.splitext(path)[1].lstrip(".").upper()
    if not ext:
        ext = (fmt.get("format_name") or "").split(",")[0].upper()
    try:
        kbps = int(a.get("bit_rate") or fmt.get("bit_rate") or 0) // 1000
    except ValueError:
        kbps = 0
    try:
        dur = float(fmt.get("duration") or a.get("duration") or 0)
    except ValueError:
        dur = 0.0
    md.update({
        "orig_title": raw_title,
        "title": base,
        "artist": tag("artist", "album_artist", "albumartist", "performer") or fa,
        "featuring": feat,
        "mix_name": mix,
        "remixer": remixer or tag("remixer", "mixartist", "tpe4"),
        "is_remix": int(is_remix or bool(tag("remixer", "mixartist", "tpe4"))),
        "genre": tag("genre"),
        "album": tag("album"),
        "year": m.group(0) if m else "",
        "label": tag("label", "publisher", "organization", "tpub", "copyright"),
        "comment": tag("comment", "description", "comm"),
        "tag_bpm": tag("tbpm", "bpm"),
        "tag_key": tag("initialkey", "tkey", "key"),
        "file_format": ext,
        "codec": a.get("codec_name", "") or "",
        "bitrate_kbps": kbps,
        "sample_rate": int(a.get("sample_rate") or 0),
        "channels": int(a.get("channels") or 0),
    })
    if dur:
        md["duration"] = dur
    return md


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
DB_COLUMNS = [
    ("path", "TEXT UNIQUE NOT NULL"), ("filename", "TEXT"),
    ("artist", "TEXT"), ("title", "TEXT"), ("orig_title", "TEXT"), ("featuring", "TEXT"),
    ("mix_name", "TEXT"), ("remixer", "TEXT"), ("is_remix", "INTEGER DEFAULT 0"),
    ("genre", "TEXT"), ("album", "TEXT"), ("year", "TEXT"), ("label", "TEXT"),
    ("comment", "TEXT"), ("notes", "TEXT"),
    ("file_format", "TEXT"), ("codec", "TEXT"), ("bitrate_kbps", "INTEGER"),
    ("sample_rate", "INTEGER"), ("channels", "INTEGER"), ("size", "INTEGER"), ("mtime", "REAL"),
    ("duration", "REAL"), ("tag_bpm", "TEXT"), ("tag_key", "TEXT"),
    ("bpm", "REAL"), ("bpm_conf", "REAL"), ("key_pc", "INTEGER"), ("key_minor", "INTEGER"),
    ("key_name", "TEXT"), ("camelot", "TEXT"), ("key_conf", "REAL"), ("key_alt", "TEXT"), ("chroma", "TEXT"),
    ("beat_offset", "REAL"), ("cues", "TEXT"), ("peaks", "BLOB"), ("peaks_ch", "INTEGER"),
    ("spec", "BLOB"), ("spec_rows", "INTEGER"), ("spec_cols", "INTEGER"), ("avg_spec", "BLOB"),
    ("engine", "TEXT"), ("analyzed_at", "TEXT"), ("added_at", "TEXT"), ("edited_at", "TEXT"),
]
BLOB_COLS = {"spec", "avg_spec", "peaks"}
LIST_COLS = ["id"] + [c for c, _ in DB_COLUMNS if c not in BLOB_COLS]
EDITABLE = {"artist", "title", "featuring", "mix_name", "remixer", "is_remix", "genre", "album",
            "year", "label", "comment", "notes", "bpm", "bpm_conf", "camelot", "key_name",
            "key_pc", "key_minor", "key_conf", "key_alt", "cues"}
FILE_INFO_COLS = ["filename", "orig_title", "file_format", "codec", "bitrate_kbps", "sample_rate",
                  "channels", "size", "mtime", "duration", "tag_bpm", "tag_key"]
ANALYSIS_COLS = ["bpm", "bpm_conf", "key_pc", "key_minor", "key_name", "camelot", "key_conf", "key_alt",
                 "chroma", "beat_offset", "cues", "peaks", "peaks_ch", "spec", "spec_rows",
                 "spec_cols", "avg_spec",
                 "engine", "analyzed_at",
                 "size", "mtime", "duration"]


def now_str():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def to_db_values(rec):
    """Convert an in-memory record (numpy arrays, lists) into DB column values."""
    d = {}
    for c, _ in DB_COLUMNS:
        if c in rec and c not in BLOB_COLS and c not in ("chroma", "cues"):
            d[c] = rec[c]
    if "chroma" in rec:
        d["chroma"] = json.dumps(rec["chroma"])
    if "cues" in rec:
        d["cues"] = json.dumps(rec["cues"] or [])
    if rec.get("spec") is not None:
        img = rec["spec"]
        d["spec"] = spec_to_blob(img)
        d["spec_rows"], d["spec_cols"] = int(img.shape[0]), int(img.shape[1])
    if rec.get("peaks") is not None:
        pk = np.ascontiguousarray(rec["peaks"], dtype=np.int8)
        d["peaks"] = zlib.compress(pk.tobytes(), 6)
        d["peaks_ch"] = int(pk.shape[1])
    if rec.get("avg") is not None:
        d["avg_spec"] = np.asarray(rec["avg"], np.float32).tobytes()
    return d


class SongDB:
    def __init__(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.path = path
        self.con = sqlite3.connect(path)
        self.con.row_factory = sqlite3.Row
        try:
            self.con.execute("CREATE TABLE IF NOT EXISTS songs(id INTEGER PRIMARY KEY, %s)" %
                             ", ".join("%s %s" % c for c in DB_COLUMNS))
        except sqlite3.DatabaseError as e:
            raise RuntimeError("%s is not a SongKey database (%s)" % (path, e))
        have = {r[1] for r in self.con.execute("PRAGMA table_info(songs)")}
        for c, t in DB_COLUMNS:       # migrate older databases
            if c not in have:
                t = t.replace("UNIQUE", "").replace("NOT NULL", "")
                self.con.execute("ALTER TABLE songs ADD COLUMN %s %s" % (c, t))
        for c in ("camelot", "bpm", "artist", "genre"):
            self.con.execute("CREATE INDEX IF NOT EXISTS idx_%s ON songs(%s)" % (c, c))
        self.con.commit()

    def insert(self, rec):
        d = to_db_values(rec)
        d["added_at"] = now_str()
        cols = list(d)
        sql = ("INSERT INTO songs(%s) VALUES (%s) ON CONFLICT(path) DO UPDATE SET %s" % (
            ",".join(cols), ",".join("?" * len(cols)),
            ",".join("%s=excluded.%s" % (c, c) for c in cols if c != "path")))
        self.con.execute(sql, [d[c] for c in cols])
        self.con.commit()
        return self.con.execute("SELECT id FROM songs WHERE path=?", (d["path"],)).fetchone()[0]

    def update_fields(self, ids, changes):
        d = {k: v for k, v in changes.items() if k in EDITABLE}
        if not d:
            return
        d["edited_at"] = now_str()
        cols = list(d)
        self.con.executemany("UPDATE songs SET %s WHERE id=?" % ",".join("%s=?" % c for c in cols),
                             [[d[c] for c in cols] + [i] for i in ids])
        self.con.commit()

    def update_file_info(self, song_id, meta):
        cols = [c for c in FILE_INFO_COLS if c in meta]
        self.con.execute("UPDATE songs SET %s WHERE id=?" % ",".join("%s=?" % c for c in cols),
                         [meta[c] for c in cols] + [song_id])
        self.con.commit()

    def update_peaks(self, song_id, peaks):
        self.con.execute("UPDATE songs SET peaks=?, peaks_ch=? WHERE id=?",
                         (zlib.compress(np.ascontiguousarray(peaks).tobytes(), 6), int(peaks.shape[1]),
                          song_id))
        self.con.commit()

    def update_analysis(self, song_id, rec):
        d = to_db_values(rec)
        cols = [c for c in ANALYSIS_COLS if c in d]
        self.con.execute("UPDATE songs SET %s WHERE id=?" % ",".join("%s=?" % c for c in cols),
                         [d[c] for c in cols] + [song_id])
        self.con.commit()

    def all(self):
        return [dict(r) for r in self.con.execute(
            "SELECT %s FROM songs ORDER BY artist COLLATE NOCASE, title COLLATE NOCASE" %
            ",".join(LIST_COLS))]

    def get(self, song_id):
        r = self.con.execute("SELECT * FROM songs WHERE id=?", (song_id,)).fetchone()
        return dict(r) if r else None

    def by_path(self, path):
        r = self.con.execute("SELECT id, size, mtime FROM songs WHERE path=?",
                             (os.path.abspath(path),)).fetchone()
        return dict(r) if r else None

    def genres(self):
        return [r[0] for r in self.con.execute(
            "SELECT DISTINCT genre FROM songs WHERE genre<>'' ORDER BY genre COLLATE NOCASE")]

    def count(self):
        return self.con.execute("SELECT COUNT(*) FROM songs").fetchone()[0]

    def delete(self, ids):
        self.con.executemany("DELETE FROM songs WHERE id=?", [(i,) for i in ids])
        self.con.commit()

    def export_csv(self, out):
        cols = ["artist", "title", "featuring", "mix_name", "remixer", "is_remix", "genre", "album",
                "year", "label", "bpm", "key_name", "camelot", "duration", "file_format",
                "bitrate_kbps", "sample_rate", "channels", "size", "notes", "cues", "path"]
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(cols)
            for r in self.all():
                w.writerow([fmt_time(r["duration"]) if c == "duration" else
                            cues_short(parse_cues(r.get("cues"))) if c == "cues" else r.get(c)
                            for c in cols])


def row_to_display(r):
    """DB row (dict) -> record usable by the display code."""
    d = dict(r)
    if d.get("spec") is not None and d.get("spec_rows"):
        d["spec"] = blob_to_spec(d["spec"], d["spec_rows"], d["spec_cols"])
    else:
        d["spec"] = None
    d["avg"] = np.frombuffer(d["avg_spec"], dtype=np.float32) if d.get("avg_spec") else None
    d["chroma"] = json.loads(d["chroma"]) if d.get("chroma") else []
    d["cues"] = parse_cues(d.get("cues"))
    if d.get("peaks"):
        try:
            d["peaks"] = np.frombuffer(zlib.decompress(d["peaks"]), dtype=np.int8).reshape(
                -1, d.get("peaks_ch") or 2, 3)
        except (zlib.error, ValueError):
            d["peaks"] = None
    else:
        d["peaks"] = None
    return d


def library_id_if_unchanged(db, path):
    row = db.by_path(path)
    if not row:
        return None
    try:
        st = os.stat(path)
    except OSError:
        return None
    if row["size"] == st.st_size and abs((row["mtime"] or 0) - st.st_mtime) < 1.0:
        return row["id"]
    return None


def expand_paths(paths):
    out = []
    for p in paths:
        p = os.path.expanduser(p)
        if os.path.isdir(p):
            for root, _dirs, files in os.walk(p):
                for f in sorted(files):
                    if os.path.splitext(f)[1].lower() in AUDIO_EXT:
                        out.append(os.path.join(root, f))
        elif os.path.isfile(p):
            out.append(p)
    return out


def apply_key_bpm_edits(changes):
    """Turn edited 'camelot' / 'bpm' into all the DB fields they affect."""
    d = dict(changes)
    if "camelot" in d:
        code = d["camelot"]
        if code in CAMELOT_TO_KEY:
            pc, minor = CAMELOT_TO_KEY[code]
            d.update(key_pc=pc, key_minor=int(minor), key_name=key_name(pc, minor), key_conf=-1.0, key_alt="")
        else:
            d.pop("camelot")
    if "bpm" in d:
        if d["bpm"] is None:
            d.pop("bpm")
        else:
            d["bpm_conf"] = -1.0
    return d


CAMELOT_TO_KEY = {camelot(pc, mi): (pc, mi) for pc in range(12) for mi in (False, True)}
KEY_CHOICES = ["%s   %s" % (c, key_name(*CAMELOT_TO_KEY[c]))
               for c in sorted(CAMELOT_TO_KEY, key=camelot_sort_key)]
COMMON_GENRES = ["House", "Deep House", "Tech House", "Progressive House", "Techno", "Trance",
                 "Drum & Bass", "Dubstep", "Breaks", "Electro", "EDM", "Hip-Hop", "R&B", "Pop",
                 "Dance", "Disco", "Funk", "Soul", "Rock", "Reggae", "Reggaeton", "Latin",
                 "Afro House", "Kizomba", "Ambient", "Downtempo", "Jazz", "Classical"]


# ---------------------------------------------------------------------------
# Plain-language descriptions
# ---------------------------------------------------------------------------
def confidence_word(c):
    if c is None:
        return ""
    if c < 0:
        return "set by you"
    return "sure" if c >= 0.66 else ("fairly sure" if c >= 0.33 else "not sure")


def fmt_bpm(bpm):
    return ("%.1f" % bpm).rstrip("0").rstrip(".") if bpm else ""


def fmt_size(n):
    if not n:
        return ""
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return ("%.0f %s" if unit in ("B", "KB") else "%.1f %s") % (n, unit)
        n /= 1024.0


def fmt_format(r):
    f = r.get("file_format") or ""
    if r.get("bitrate_kbps"):
        f += " %dk" % r["bitrate_kbps"]
    return f


def describe_key(name, code, conf, alt=""):
    if not name:
        return "unknown"
    mode = "minor (darker mood)" if name.endswith("m") else "major (brighter mood)"
    extra = ("  -  could also be %s" % alt) if alt and (conf is None or conf >= 0) else ""
    return "%s %s - Camelot %s  (%s)%s" % (name[:-1] if name.endswith("m") else name, mode, code,
                                           confidence_word(conf), extra)


def describe_bpm(bpm, conf):
    if not bpm:
        return "no steady beat found"
    if bpm < 90:
        feel = "slow"
    elif bpm < 115:
        feel = "medium"
    elif bpm < 135:
        feel = "dance tempo"
    elif bpm < 160:
        feel = "fast"
    else:
        feel = "very fast"
    return "%s BPM, %s  (%s)" % (fmt_bpm(bpm), feel, confidence_word(conf))


def describe_file(r):
    parts = []
    if r.get("file_format"):
        s = r["file_format"]
        if r.get("bitrate_kbps"):
            s += " %d kbps" % r["bitrate_kbps"]
        if r.get("sample_rate"):
            s += ", %.1f kHz" % (r["sample_rate"] / 1000.0)
        if r.get("channels"):
            s += ", " + {1: "mono", 2: "stereo"}.get(r["channels"], "%d ch" % r["channels"])
        parts.append(s)
    if r.get("size"):
        parts.append(fmt_size(r["size"]))
    return "  |  ".join(parts)


def parse_cues(v):
    if isinstance(v, list):
        return v
    try:
        return json.loads(v) if v else []
    except ValueError:
        return []


def visible_cues(cues):
    return sorted([c for c in (cues or []) if not c.get("hidden")], key=lambda c: c["t"])


def cues_short(cues):
    return "; ".join("%s %s" % (c.get("label") or c["kind"].upper(), fmt_time(c["t"]))
                     for c in visible_cues(cues))


def merge_marks(old, new_auto):
    """After re-analysis: keep everything the user did, refresh the automatic marks."""
    mine = [c for c in old if not c.get("auto")]
    moved = [{"kind": c["kind"], "t": c["orig_t"]} for c in mine if "orig_t" in c]
    removed = [c for c in old if c.get("auto") and c.get("hidden")]
    out = []
    for c in new_auto:
        if any(g["kind"] == c["kind"] and abs(g["t"] - c["t"]) < 2.0 for g in moved):
            continue                       # you adjusted this one - keep your position
        if any(g["kind"] == c["kind"] and abs(g["t"] - c["t"]) < 2.0 for g in removed):
            c = dict(c, hidden=True)       # you removed this one - stays removed
        out.append(c)
    return sorted(out + mine, key=lambda c: c["t"])


def tap_tempo(times):
    """BPM from tap times (seconds). Fits a straight line through the taps instead of
    averaging gaps, ignores double taps and fills in missed beats.
    Returns (bpm, plus_minus_bpm, taps_used) or None."""
    t = np.asarray(times, dtype=float)
    if len(t) < 4:
        return None
    iv = np.diff(t)
    med = float(np.median(iv))
    if med <= 0:
        return None
    keep = np.concatenate([[True], iv > 0.45 * med])        # drop accidental double taps
    t = t[keep]
    if len(t) < 4:
        return None
    iv = np.diff(t)
    med = float(np.median(iv))
    beats = np.concatenate([[0.0], np.cumsum(np.maximum(1, np.round(iv / med)))])   # missed beats
    slope, icpt = np.polyfit(beats, t, 1)
    resid = t - (icpt + slope * beats)
    good = np.abs(resid) < 0.3 * slope
    if good.sum() >= 4 and not good.all():
        beats, t = beats[good], t[good]
        slope, icpt = np.polyfit(beats, t, 1)
        resid = t - (icpt + slope * beats)
    if slope <= 0:
        return None
    n = len(t)
    spread = np.sqrt(((beats - beats.mean()) ** 2).sum())
    se = np.sqrt((resid ** 2).sum() / max(1, n - 2)) / max(spread, 1e-9)
    bpm = 60.0 / slope
    return bpm, 60.0 * se / slope ** 2, n


def tap_relation(tapped, measured):
    """Plain words for how the tapped tempo relates to the measured one."""
    if not tapped or not measured:
        return ""
    r = tapped / measured
    for f, words in ((1.0, "matches what the app measured"),
                     (0.5, "is half of the app's value (the app counted double time)"),
                     (2.0, "is double the app's value (the app counted half time)"),
                     (0.75, "is 3/4 of the app's value - the app made a 4:3 mistake"),
                     (4 / 3.0, "is 4/3 of the app's value - the app made a 3:4 mistake"),
                     (2 / 3.0, "is 2/3 of the app's value - the app made a 3:2 mistake"),
                     (1.5, "is 1.5x the app's value - the app made a 2:3 mistake")):
        if abs(r / f - 1) < 0.015:
            return "Your tapping " + words
    return "Your tapping differs from the app (%s BPM)" % fmt_bpm(measured)


def describe_cues(cues):
    vis = visible_cues(cues)
    if not vis:
        return "Mix points: none yet  (press Re-analyse, or right-click the spectrum to add your own)"

    def pick(kind, last=False):
        c = [x for x in vis if x["kind"] == kind]
        return (c[-1] if last else c[0]) if c else None
    parts = []
    for kind, text, last in (("in", "Mix in %s", False), ("drop", "full energy %s", False),
                             ("out", "mix out %s", False), ("end", "end %s", True)):
        c = pick(kind, last)
        if c:
            parts.append(text % fmt_time(c["t"]))
    return "Mix points:  " + "  |  ".join(parts)


class Player:
    """Plays audio with ffplay (part of the ffmpeg package).
    The position shown comes from ffplay's own audio clock."""

    def __init__(self):
        self.proc = None
        self.path = None
        self.volume = 80
        self.base = 0.0
        self.reported = None
        self.paused_at = None

    @staticmethod
    def available():
        t = tool("ffplay")
        return os.path.isfile(t) or shutil.which(t) is not None

    def play(self, path, pos=0.0):
        self.stop()
        pos = max(0.0, float(pos))
        cmd = [tool("ffplay"), "-nodisp", "-autoexit", "-hide_banner", "-loglevel", "info", "-stats",
               "-volume", str(int(self.volume)), "-ss", "%.3f" % pos, path]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.PIPE, **NO_WINDOW)
        self.path, self.base, self.reported, self.paused_at = path, pos, None, None
        threading.Thread(target=self._read, args=(self.proc,), daemon=True).start()

    def _read(self, proc):
        rx = re.compile(rb"^\s*(\d+\.\d+)\s+M-A")
        buf = b""
        try:
            fd = proc.stderr.fileno()
        except (ValueError, AttributeError):
            return
        while True:
            try:
                data = os.read(fd, 4096)
            except OSError:
                break
            if not data:
                break
            buf += data
            parts = re.split(rb"[\r\n]", buf)
            buf = parts[-1][-512:]
            for line in parts[:-1]:
                m = rx.match(line)
                if m and proc is self.proc:
                    self.reported = float(m.group(1))

    def playing(self):
        return self.proc is not None and self.proc.poll() is None

    def position(self):
        if self.paused_at is not None:
            return self.paused_at
        if not self.playing():
            return None
        return self.reported if self.reported is not None else self.base

    def pause(self):
        pos = self.position()
        self.stop(keep=True)
        self.paused_at = pos

    def resume(self):
        if self.path and self.paused_at is not None:
            self.play(self.path, self.paused_at)

    def stop(self, keep=False):
        p, self.proc = self.proc, None
        if p is not None:
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=1.5)
                except subprocess.TimeoutExpired:
                    p.kill()
            try:
                p.stderr.close()
            except Exception:
                pass
        if not keep:
            self.paused_at = None
            self.reported = None


# ---------------------------------------------------------------------------
# Button icons (drawn with Pillow at 4x size, then scaled down for smooth edges)
# ---------------------------------------------------------------------------
def make_icon_images(size=18):
    from PIL import Image, ImageDraw
    S = 72
    FG = (226, 229, 233, 255)
    GREEN = (102, 214, 120, 255)
    RED = (239, 100, 100, 255)
    BLUE = (79, 195, 247, 255)
    ORANGE = (255, 171, 64, 255)
    YELLOW = (255, 210, 80, 255)
    DARK = (40, 42, 48, 255)
    W = 7
    icons = {}
    try:
        lanczos = Image.Resampling.LANCZOS
    except AttributeError:
        lanczos = Image.LANCZOS

    def canvas():
        im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
        return im, ImageDraw.Draw(im)

    def done(name, im):
        icons[name] = im.resize((size, size), lanczos)

    def outline(d, pts, col, w=W):
        d.line(pts + [pts[0]], fill=col, width=w, joint="curve")

    def doc(d, col=FG):
        pts = [(16, 6), (42, 6), (56, 20), (56, 66), (16, 66)]
        d.polygon(pts, fill=DARK)
        outline(d, pts, col)
        d.line([(42, 6), (42, 20), (56, 20)], fill=col, width=5, joint="curve")

    def magnifier(d):
        d.ellipse((6, 6, 48, 48), outline=FG, width=W)
        d.line((42, 42, 64, 64), fill=FG, width=11)

    def tri(d, pts, col):
        d.polygon(pts, fill=col)

    im, d = canvas(); doc(d); d.line((48, 40, 48, 70), fill=GREEN, width=8); d.line((33, 55, 63, 55), fill=GREEN, width=8)
    done("open_file", im)

    im, d = canvas()
    d.polygon([(6, 16), (26, 16), (33, 24), (66, 24), (66, 60), (6, 60)], fill=YELLOW)
    d.polygon([(6, 30), (66, 30), (66, 60), (6, 60)], fill=(255, 190, 50, 255))
    done("folder", im)

    im, d = canvas()
    d.line((18, 54, 52, 20), fill=YELLOW, width=15)
    d.line((48, 24, 58, 14), fill=RED, width=15)
    tri(d, [(8, 64), (11, 46), (26, 61)], FG)
    done("edit", im)

    im, d = canvas()
    d.arc((12, 12, 60, 60), start=30, end=300, fill=BLUE, width=W + 1)
    tri(d, [(58, 21), (52, 4), (41, 25)], BLUE)
    done("refresh", im)

    im, d = canvas()
    d.line((12, 18, 60, 18), fill=RED, width=W)
    d.line([(27, 17), (29, 8), (43, 8), (45, 17)], fill=RED, width=5, joint="curve")
    outline(d, [(18, 24), (54, 24), (50, 64), (22, 64)], RED)
    d.line((31, 32, 31, 56), fill=RED, width=5)
    d.line((41, 32, 41, 56), fill=RED, width=5)
    done("trash", im)

    im, d = canvas(); doc(d)
    d.line((36, 24, 36, 50), fill=GREEN, width=8)
    tri(d, [(22, 44), (50, 44), (36, 62)], GREEN)
    done("export", im)

    im, d = canvas(); tri(d, [(18, 10), (18, 62), (62, 36)], GREEN); done("play", im)
    im, d = canvas(); d.rectangle((16, 10, 30, 62), fill=FG); d.rectangle((42, 10, 56, 62), fill=FG); done("pause", im)
    im, d = canvas(); d.rounded_rectangle((14, 14, 58, 58), radius=6, fill=RED); done("stop", im)

    im, d = canvas(); d.rectangle((10, 12, 19, 60), fill=BLUE); tri(d, [(62, 12), (62, 60), (24, 36)], FG)
    done("prev_mark", im)
    im, d = canvas(); tri(d, [(10, 12), (10, 60), (48, 36)], FG); d.rectangle((53, 12, 62, 60), fill=BLUE)
    done("next_mark", im)

    im, d = canvas(); magnifier(d)
    d.line((15, 27, 39, 27), fill=GREEN, width=7); d.line((27, 15, 27, 39), fill=GREEN, width=7)
    done("zoom_in", im)
    im, d = canvas(); magnifier(d); d.line((15, 27, 39, 27), fill=ORANGE, width=7)
    done("zoom_out", im)

    im, d = canvas()
    d.line((4, 14, 4, 58), fill=FG, width=6); d.line((68, 14, 68, 58), fill=FG, width=6)
    d.line((20, 36, 52, 36), fill=FG, width=W)
    tri(d, [(10, 36), (24, 24), (24, 48)], FG); tri(d, [(62, 36), (48, 24), (48, 48)], FG)
    done("fit", im)

    im, d = canvas(); d.line([(46, 12), (24, 36), (46, 60)], fill=FG, width=9, joint="curve"); done("left", im)
    im, d = canvas(); d.line([(26, 12), (48, 36), (26, 60)], fill=FG, width=9, joint="curve"); done("right", im)
    im, d = canvas()
    d.line([(36, 12), (14, 36), (36, 60)], fill=BLUE, width=9, joint="curve")
    d.line([(60, 12), (38, 36), (60, 60)], fill=BLUE, width=9, joint="curve")
    done("dleft", im)
    im, d = canvas()
    d.line([(12, 12), (34, 36), (12, 60)], fill=BLUE, width=9, joint="curve")
    d.line([(36, 12), (58, 36), (36, 60)], fill=BLUE, width=9, joint="curve")
    done("dright", im)

    im, d = canvas(); d.line((16, 16, 56, 56), fill=RED, width=10); d.line((56, 16, 16, 56), fill=RED, width=10)
    done("delete", im)

    im, d = canvas(); d.ellipse((6, 6, 66, 66), fill=GREEN)
    d.line((36, 18, 36, 54), fill="white", width=9); d.line((18, 36, 54, 36), fill="white", width=9)
    done("add", im)

    im, d = canvas()
    outline(d, [(6, 18), (50, 18), (50, 64), (6, 64)], FG)
    d.rectangle((6, 18, 50, 28), fill=FG)
    d.line((32, 40, 62, 10), fill=BLUE, width=8)
    tri(d, [(66, 6), (44, 8), (64, 28)], BLUE)
    done("window", im)

    im, d = canvas()
    for i, h in enumerate((20, 44, 30, 56, 24, 38, 16)):
        x = 8 + i * 9
        d.line((x, 36 - h / 2, x, 36 + h / 2), fill=GREEN, width=6)
    done("wave", im)

    im, d = canvas()
    outline(d, [(24, 8), (48, 8), (60, 64), (12, 64)], FG)
    d.line((14, 52, 58, 52), fill=FG, width=5)
    d.line((36, 52, 54, 12), fill=BLUE, width=6)
    d.rectangle((42, 26, 52, 34), fill=ORANGE)
    done("tap", im)

    im, d = canvas()
    d.ellipse((6, 6, 66, 66), outline=BLUE, width=W)
    d.ellipse((31, 15, 41, 25), fill=BLUE)
    d.line((36, 32, 36, 56), fill=BLUE, width=9)
    done("info", im)

    return icons


def make_app_icon(size=256):
    """SongKey's own icon (window + .exe): green waveform bars on a dark rounded square."""
    from PIL import Image, ImageDraw
    S = 512
    im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((16, 16, S - 16, S - 16), radius=96, fill=(30, 31, 36, 255),
                        outline=(63, 191, 95, 255), width=14)
    heights = (120, 230, 170, 330, 260, 380, 210, 290, 150, 230, 110)
    n = len(heights)
    bw = 26
    gap = (S - 120 - n * bw) / (n - 1)
    for i, h in enumerate(heights):
        x = 60 + i * (bw + gap)
        g = 150 + int(90 * h / 380)
        d.rounded_rectangle((x, S / 2 - h / 2, x + bw, S / 2 + h / 2), radius=13, fill=(80, g, 110, 255))
    try:
        lanczos = Image.Resampling.LANCZOS
    except AttributeError:
        lanczos = Image.LANCZOS
    return im.resize((size, size), lanczos)


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------
EDIT_FIELDS = [   # key, label, kind
    ("artist", "Artist", "entry"),
    ("title", "Title", "entry"),
    ("featuring", "Featuring", "entry"),
    ("mix_name", "Mix / version", "entry"),
    ("remixer", "Remixed by", "entry"),
    ("is_remix", "This is a remix", "check"),
    ("genre", "Style / genre", "genre"),
    ("album", "Album", "entry"),
    ("year", "Year", "entry"),
    ("label", "Label", "entry"),
    ("bpm", "BPM", "bpm"),
    ("camelot", "Key", "key"),
    ("comment", "Comment (from file)", "entry"),
    ("notes", "My notes", "text"),
]
MULTI_FIELDS = {"artist", "genre", "album", "year", "label", "remixer", "notes"}


def run_gui(args):
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox

    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.ticker import FuncFormatter
    from matplotlib.colors import LinearSegmentedColormap

    BG, PANEL, FG, DIM, ACCENT = "#1e1f24", "#26282e", "#e6e6e6", "#9a9ca3", "#4fc3f7"
    GREEN = "#3fbf5f"
    GREEN_CMAP = LinearSegmentedColormap.from_list(
        "songkey_green", ["#060907", "#0b2614", "#0f4a24", "#1c7f3c", "#3fbf5f", "#9cf0a9", "#effff1"])

    # ------------------------------------------------------------------ worker
    class Worker(threading.Thread):
        def __init__(self, out_q, engine, bpm_range):
            super().__init__(daemon=True)
            self.jobs = queue.Queue()
            self.out = out_q
            self.engine = engine
            self.bpm_range = bpm_range

        def run(self):
            while True:
                path, song_id = self.jobs.get()
                name = os.path.basename(path)
                try:
                    rec = analyse(
                        path, self.engine, self.bpm_range,
                        on_spectrum=lambda d, s=song_id: self.out.put(("spectrum", (d, s))),
                        on_status=lambda m, n=name: self.out.put(("status", "%s: %s..." % (n, m))))
                    self.out.put(("done", (rec, song_id)))
                except Exception as e:
                    self.out.put(("error", (path, song_id, str(e))))
                self.out.put(("idle", self.jobs.qsize()))

    # ------------------------------------------------------------------ dialog
    class EditDialog(tk.Toplevel):
        """mode: 'add' (new song, prefilled from file metadata), 'edit' (one song),
        'multi' (several songs: only changed fields are applied)."""

        def __init__(self, app, mode, records, on_save, on_close=None):
            super().__init__(app.root)
            self.app, self.mode, self.on_save, self.on_close = app, mode, on_save, on_close
            self.records = records
            n = len(records)
            self.title({"add": "Add to library", "edit": "Edit song",
                        "multi": "Edit %d songs" % n}[mode])
            self.configure(bg=PANEL)
            self.transient(app.root)
            self.resizable(True, False)
            self.protocol("WM_DELETE_WINDOW", self.cancel)

            frm = ttk.Frame(self, style="Panel.TFrame", padding=14)
            frm.pack(fill="both", expand=True)
            frm.columnconfigure(1, weight=1)
            row = 0
            intro = {
                "add": "Check the details read from the file, fix anything wrong, then press "
                       "'Add to library'.",
                "edit": "Change anything you like, then press 'Save'.",
                "multi": "Only the boxes you change are applied to all %d songs. "
                         "An empty box means the songs have different values." % n,
            }[mode]
            ttk.Label(frm, text=intro, style="Dim.TLabel", wraplength=520).grid(
                row=row, column=0, columnspan=3, sticky="w", pady=(0, 10))
            row += 1

            fields = [f for f in EDIT_FIELDS if mode != "multi" or f[0] in MULTI_FIELDS]
            self.vars, self.widgets, self.initial = {}, {}, {}
            genres = sorted(set(app.db.genres()) | set(COMMON_GENRES), key=str.lower)
            for key, label, kind in fields:
                vals = [self._value(r, key) for r in records]
                same = all(v == vals[0] for v in vals)
                init = vals[0] if same else None
                self.initial[key] = init
                ttk.Label(frm, text=label, style="Panel.TLabel").grid(
                    row=row, column=0, sticky="nw", padx=(0, 10), pady=3)
                if kind == "check":
                    var = tk.BooleanVar(value=bool(init))
                    w = ttk.Checkbutton(frm, variable=var, style="Panel.TCheckbutton")
                    w.grid(row=row, column=1, sticky="w", pady=3)
                elif kind == "text":
                    var = None
                    w = tk.Text(frm, height=3, width=50, bg=BG, fg=FG, insertbackground=FG,
                                relief="flat", wrap="word")
                    w.insert("1.0", init or "")
                    w.grid(row=row, column=1, columnspan=2, sticky="ew", pady=3)
                elif kind == "genre":
                    var = tk.StringVar(value=init or "")
                    w = ttk.Combobox(frm, textvariable=var, values=genres)
                    w.grid(row=row, column=1, columnspan=2, sticky="ew", pady=3)
                elif kind == "key":
                    var = tk.StringVar(value=next((k for k in KEY_CHOICES if k.split()[0] == init), ""))
                    w = ttk.Combobox(frm, textvariable=var, values=KEY_CHOICES, state="readonly",
                                     width=14)
                    w.grid(row=row, column=1, sticky="w", pady=3)
                    det = records[0].get("tag_key") if mode != "multi" else ""
                    if det:
                        ttk.Label(frm, text="file tag says: %s" % det, style="Dim.TLabel").grid(
                            row=row, column=2, sticky="w")
                elif kind == "bpm":
                    var = tk.StringVar(value=fmt_bpm(init) if init else "")
                    box = ttk.Frame(frm, style="Panel.TFrame")
                    box.grid(row=row, column=1, columnspan=2, sticky="w", pady=3)
                    w = ttk.Entry(box, textvariable=var, width=8)
                    w.pack(side="left")
                    ttk.Button(box, text="half", width=5, command=lambda: self._scale_bpm(0.5)).pack(
                        side="left", padx=(6, 2))
                    ttk.Button(box, text="double", width=6, command=lambda: self._scale_bpm(2)).pack(
                        side="left", padx=(0, 2))
                    ttk.Button(box, text="x1.5", width=5, command=lambda: self._scale_bpm(1.5)).pack(
                        side="left", padx=(0, 2))
                    ttk.Button(box, text="x2/3", width=5, command=lambda: self._scale_bpm(2 / 3.0)).pack(
                        side="left")
                    det = records[0].get("tag_bpm") if mode != "multi" else ""
                    if det:
                        ttk.Label(box, text="   file tag says: %s" % det, style="Dim.TLabel").pack(
                            side="left")
                else:
                    var = tk.StringVar(value=init or "")
                    w = ttk.Entry(frm, textvariable=var, width=50)
                    w.grid(row=row, column=1, columnspan=2, sticky="ew", pady=3)
                self.vars[key], self.widgets[key] = var, w
                row += 1

            if mode != "multi":
                r = records[0]
                ttk.Separator(frm).grid(row=row, column=0, columnspan=3, sticky="ew", pady=8)
                row += 1
                info = [("File", r.get("path", "")),
                        ("Format", describe_file(r) or "-"),
                        ("Length", fmt_time(r.get("duration"))),
                        ("Title in file", r.get("orig_title") or "-")]
                for lab, val in info:
                    ttk.Label(frm, text=lab, style="Dim.TLabel").grid(row=row, column=0, sticky="nw")
                    ttk.Label(frm, text=val, style="Panel.TLabel", wraplength=520).grid(
                        row=row, column=1, columnspan=2, sticky="w")
                    row += 1

            self.file_info = None
            btns = ttk.Frame(frm, style="Panel.TFrame")
            btns.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(14, 0))
            if mode == "edit":
                ttk.Button(btns, text="Re-read from file", command=self.reread).pack(side="left")
            ok_text = {"add": "Add to library", "edit": "Save", "multi": "Apply to %d songs" % n}[mode]
            ttk.Button(btns, text="Not now" if mode == "add" else "Cancel",
                       command=self.cancel).pack(side="right", padx=(6, 0))
            ttk.Button(btns, text=ok_text, style="Accent.TButton", command=self.save).pack(side="right")
            self.bind("<Escape>", lambda e: self.cancel())
            self.update_idletasks()
            x = app.root.winfo_rootx() + 80
            y = app.root.winfo_rooty() + 40
            self.geometry("+%d+%d" % (x, y))
            self.grab_set()
            self.widgets.get("artist", self).focus_set()

        def reread(self):
            path = self.records[0].get("path")
            try:
                meta = read_metadata(path)
            except OSError as e:
                messagebox.showerror(APP_NAME, "Can't read the file:\n%s" % e, parent=self)
                return
            for key in ("artist", "title", "featuring", "mix_name", "remixer", "genre", "album",
                        "year", "label", "comment"):
                if key in self.vars:
                    self.vars[key].set(meta.get(key) or "")
            self.vars["is_remix"].set(bool(meta.get("is_remix")))
            self.file_info = meta
            self.app.set_status("Details re-read from the file - press Save to keep them")

        @staticmethod
        def _value(r, key):
            v = r.get(key)
            if key == "is_remix":
                return int(bool(v))
            if key == "bpm":
                return round(float(v), 2) if v else None
            return "" if v is None else str(v)

        def _scale_bpm(self, f):
            try:
                v = round(float(self.vars["bpm"].get().replace(",", ".")) * f, 2)
                if abs(v - round(v)) < 0.10:
                    v = float(round(v))
                self.vars["bpm"].set(fmt_bpm(v))
            except ValueError:
                pass

        def collect(self):
            out = {}
            for key, var in self.vars.items():
                w = self.widgets[key]
                if key == "notes":
                    v = w.get("1.0", "end").strip()
                elif key == "is_remix":
                    v = int(var.get())
                elif key == "bpm":
                    s = var.get().strip().replace(",", ".")
                    if s:
                        v = float(s)    # ValueError handled by caller
                        if not 20 <= v <= 400:
                            raise ValueError("BPM must be between 20 and 400")
                        v = round(v, 2)
                    else:
                        v = None
                elif key == "camelot":
                    s = var.get().strip()
                    v = s.split()[0] if s else ""
                else:
                    v = var.get().strip()
                init = self.initial[key]
                if key == "bpm" and v is not None and init is not None and fmt_bpm(v) == fmt_bpm(init):
                    v = init           # the box shows 1 decimal: same text means unchanged
                if self.mode == "multi" and init is None and v in ("", None):
                    continue
                if self.mode == "add" or v != init:
                    out[key] = v
            return out

        def save(self):
            try:
                changes = self.collect()
            except ValueError as e:
                messagebox.showerror(APP_NAME, "BPM: %s" % e, parent=self)
                return
            if self.file_info:
                changes["_file_info"] = self.file_info
            self.grab_release()
            self.destroy()
            self.on_save(changes)
            if self.on_close:
                self.on_close()

        def cancel(self):
            self.grab_release()
            self.destroy()
            if self.on_close:
                self.on_close()

    # ------------------------------------------------------------------ small helpers
    class Tooltip:
        """Little yellow hint that appears when the mouse rests on a widget."""

        def __init__(self, widget, text):
            self.w, self.text, self.tip, self.job = widget, text, None, None
            widget.bind("<Enter>", self.schedule, add="+")
            widget.bind("<Leave>", self.hide, add="+")
            widget.bind("<ButtonPress>", self.hide, add="+")

        def schedule(self, _e=None):
            self.hide()
            self.job = self.w.after(550, self.show)

        def show(self):
            if self.tip is not None or not self.text:
                return
            try:
                x = self.w.winfo_rootx() + 12
                y = self.w.winfo_rooty() + self.w.winfo_height() + 4
                self.tip = tk.Toplevel(self.w)
                self.tip.wm_overrideredirect(True)
                self.tip.geometry("+%d+%d" % (x, y))
                tk.Label(self.tip, text=self.text, bg="#fff6c2", fg="#111111", relief="solid", bd=1,
                         padx=6, pady=2, font=("Helvetica", 9)).pack()
            except tk.TclError:
                self.tip = None

        def hide(self, _e=None):
            if self.job:
                try:
                    self.w.after_cancel(self.job)
                except tk.TclError:
                    pass
                self.job = None
            if self.tip is not None:
                try:
                    self.tip.destroy()
                except tk.TclError:
                    pass
                self.tip = None

    class Compactor:
        """When a toolbar is too narrow for its buttons, show icons only (the hover hints
        still explain them); go back to icon + text when there is room again."""

        def __init__(self, frame):
            self.frame, self.items, self.compact, self.full_w = frame, [], False, None
            frame.bind("<Configure>", self.check, add="+")

        def add_button(self, b):
            if b.cget("image") and b.cget("text"):
                self.items.append((b, {"text": b.cget("text"), "compound": b.cget("compound"),
                                       "width": b.cget("width")}, {"text": "", "compound": "image", "width": 0}))
            return b

        def add_text(self, w, short):
            self.items.append((w, {"text": w.cget("text")}, {"text": short}))
            return w

        def check(self, _e=None):
            w = self.frame.winfo_width()
            if w <= 1:
                return
            if not self.compact:
                need = self.frame.winfo_reqwidth()
                if need > w + 2:
                    self.full_w = need
                    self.apply(True)
            elif self.full_w and self.full_w <= w - 8:
                self.apply(False)

        def apply(self, on):
            self.compact = on
            for wdg, full, small in self.items:
                try:
                    wdg.configure(**(small if on else full))
                except tk.TclError:
                    pass

    def icon_button(app, parent, text, icon, cmd, tip=None, style="TButton", compound="left", width=None):
        img = app.icons.get(icon, "") if icon else ""
        if not img and not text:      # no Pillow: fall back to words
            text = {"zoom_in": "Zoom +", "zoom_out": "Zoom -"}.get(icon, icon)
        kw = {"text": text, "command": cmd, "style": style}
        if img:
            kw.update(image=img, compound=compound if text else "image")
        if width:
            kw["width"] = width
        elif text:
            kw["width"] = len(text)           # don't let the theme stretch every button
        else:
            kw["width"] = 0
        b = ttk.Button(parent, **kw)
        if tip:
            Tooltip(b, tip)
        return b

    # ------------------------------------------------------------------ LED VU meter
    class VUMeter:
        """Stereo LED meter: green -> yellow -> red, fast rise, smooth fall, peak-hold LED.
        Levels are RMS (+3 dB, so a full-scale sine reads 0 dB) from the stored waveform data."""
        SEGS = 18
        LO, HI = -40.0, 0.0
        FALL_DB_PER_S = 24.0
        HOLD_S = 1.2

        def __init__(self, parent, width=196, height=58):
            self.c = tk.Canvas(parent, width=width, height=height, bg=PANEL, highlightthickness=0)
            self.thr = np.linspace(self.LO, self.HI, self.SEGS + 1)[1:]     # dB needed to light each LED
            x0, x1 = 18, width - 4
            seg_w = (x1 - x0) / float(self.SEGS)
            self.on_col, self.off_col = [], []
            for t in self.thr:
                if t > -4:
                    on, off = "#ff4b3e", "#3a1b19"
                elif t > -12:
                    on, off = "#f5d33f", "#3a3519"
                else:
                    on, off = "#3fdc5a", "#163020"
                self.on_col.append(on)
                self.off_col.append(off)
            self.rects = []
            for ch, y in enumerate((5, 24)):
                self.c.create_text(8, y + 7, text="LR"[ch], fill=DIM, font=("Helvetica", 9, "bold"))
                row = []
                for k in range(self.SEGS):
                    xa = x0 + k * seg_w + 1
                    row.append(self.c.create_rectangle(xa, y, xa + seg_w - 2, y + 14, outline="",
                                                       fill=self.off_col[k]))
                self.rects.append(row)
            for db_mark in (-40, -20, -10, -4, 0):
                x = x0 + (db_mark - self.LO) / (self.HI - self.LO) * (x1 - x0)
                self.c.create_text(min(max(x, x0 + 6), x1 - 6), height - 6, text=str(db_mark), fill=DIM,
                                   font=("Helvetica", 7))
            self.level = [self.LO - 10] * 2
            self.peak = [self.LO - 10] * 2
            self.peak_t = [0.0, 0.0]
            self.state = [[False] * self.SEGS for _ in range(2)]
            self.peak_seg = [-1, -1]
            self.last_t = time.monotonic()

        def update(self, levels_db):
            """levels_db: [left, right] in dB, or None when nothing is playing."""
            now = time.monotonic()
            dt = min(0.25, now - self.last_t)
            self.last_t = now
            for ch in range(2):
                target = levels_db[ch] if levels_db else self.LO - 10
                cur = self.level[ch]
                cur = target if target > cur else max(target, cur - self.FALL_DB_PER_S * dt)
                self.level[ch] = cur
                if cur >= self.peak[ch] or now - self.peak_t[ch] > self.HOLD_S:
                    if cur >= self.peak[ch]:
                        self.peak_t[ch] = now
                    self.peak[ch] = cur if cur >= self.peak[ch] else max(cur, self.peak[ch] - self.FALL_DB_PER_S * dt)
                lit = int(np.searchsorted(self.thr, cur, side="right"))
                pk = int(np.searchsorted(self.thr, self.peak[ch], side="right")) - 1
                for k in range(self.SEGS):
                    on = k < lit or k == pk
                    if on != self.state[ch][k]:
                        self.state[ch][k] = on
                        self.c.itemconfigure(self.rects[ch][k], fill=self.on_col[k] if on else self.off_col[k])

    # ------------------------------------------------------------------ waveform
    WAVE_PEAK, WAVE_RMS, WAVE_BG = "#2e9e4f", "#8de89b", "#0e1210"

    class WaveView:
        """Audacity-style green waveform with an overview strip, zoom down to single
        samples, a beat grid, and marks you can drag or nudge."""
        MIN_SPAN = 0.05

        def __init__(self, app, parent, big=False, snap_var=None, follow_var=None):
            self.app = app
            self.big = big
            self.frame = ttk.Frame(parent, style="Panel.TFrame")
            zb = ttk.Frame(self.frame, style="Panel.TFrame")
            zb.pack(fill="x", pady=(4, 4))

            self.compactor = Compactor(zb)

            def btn(text, icon, cmd, tip, padx=1, compound="left"):
                b = icon_button(app, zb, text, icon, cmd, tip=tip, style="Small.TButton", compound=compound)
                b.pack(side="left", padx=padx)
                return self.compactor.add_button(b)
            btn("", "zoom_in", lambda: self.zoom(0.5), "Zoom in  (mouse wheel up, or +)")
            btn("", "zoom_out", lambda: self.zoom(2.0), "Zoom out  (mouse wheel down, or -)")
            btn("All", "fit", self.zoom_all, "Show the whole track", padx=(1, 4 if not big else 12))
            if not big:
                btn("Big editor", "window", app.open_editor,
                    "Open the waveform in a big separate window to fine-tune the marks", padx=(1, 12))
            ttk.Label(zb, text="Mark:", style="Panel.TLabel").pack(side="left")
            self.mark_var = tk.StringVar()
            self.mark_box = ttk.Combobox(zb, textvariable=self.mark_var, state="readonly", width=15)
            self.mark_box.pack(side="left", padx=4)
            self.mark_box.bind("<<ComboboxSelected>>", self.on_mark_chosen)
            Tooltip(self.mark_box, "Pick a mark: the view zooms to it so you can adjust it")
            btn("beat", "dleft", lambda: self.nudge("beat", -1), "Move the selected mark 1 beat earlier  (Shift+Left)")
            btn("10ms", "left", lambda: self.nudge("ms", -1), "Move the selected mark 10 ms earlier  (Left)")
            btn("10ms", "right", lambda: self.nudge("ms", 1), "Move the selected mark 10 ms later  (Right)",
                compound="right")
            btn("beat", "dright", lambda: self.nudge("beat", 1), "Move the selected mark 1 beat later  (Shift+Right)",
                compound="right")
            btn("Delete", "delete", self.delete_selected, "Delete the selected mark  (Delete key)", padx=(6, 1))
            self.snap_var = snap_var or tk.BooleanVar(value=True)
            cb = ttk.Checkbutton(zb, text="Snap to beat", variable=self.snap_var, style="Panel.TCheckbutton")
            cb.pack(side="left", padx=(10, 4))
            self.compactor.add_text(cb, "Snap")
            Tooltip(cb, "Moved and new marks land exactly on a beat")
            self.follow_var = follow_var or tk.BooleanVar(value=True)
            fb = ttk.Checkbutton(zb, text="Follow", variable=self.follow_var, style="Panel.TCheckbutton")
            fb.pack(side="left")
            Tooltip(fb, "Turn the page when the playhead runs off the right edge")

            self.fig = Figure(figsize=(10, 6.0 if big else 3.4), dpi=96, facecolor=PANEL)
            self.ax_sp = None
            self.spec_img = None
            self._sp_cache = None
            self._banks = {}
            self._build_axes()
            self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
            self.hbar = ttk.Scrollbar(self.frame, orient="horizontal", command=self.on_scrollbar)
            self.hbar.pack(side="bottom", fill="x")
            self.widget = self.canvas.get_tk_widget()
            self.widget.pack(fill="both", expand=True)
            for ev, fn in (("button_press_event", self.on_press), ("motion_notify_event", self.on_motion),
                           ("button_release_event", self.on_release), ("scroll_event", self.on_scroll),
                           ("draw_event", self.on_draw)):
                self.canvas.mpl_connect(ev, fn)
            w = self.widget
            w.bind("<Left>", lambda e: self._k(self.nudge, "ms", -1))
            w.bind("<Right>", lambda e: self._k(self.nudge, "ms", 1))
            w.bind("<Shift-Left>", lambda e: self._k(self.nudge, "beat", -1))
            w.bind("<Shift-Right>", lambda e: self._k(self.nudge, "beat", 1))
            w.bind("<Delete>", lambda e: self._k(self.delete_selected))
            for k in ("<plus>", "<KP_Add>", "<equal>"):
                w.bind(k, lambda e: self._k(self.zoom, 0.5))
            for k in ("<minus>", "<KP_Subtract>"):
                w.bind(k, lambda e: self._k(self.zoom, 2.0))

            self.peaks, self.ch, self.dur, self.path = None, 2, 0.0, None
            self.t0, self.t1 = 0.0, 1.0
            self.sel = self.drag = None
            self.drag_t, self.drag_moved, self.ov_drag = 0.0, False, False
            self.raw = None
            self.ov_cache = None
            self.marks_drawn = []
            self._mark_list = []
            self.cur = self.cur_ov = None
            self.curs = []
            self.bgs = None
            self._last = "x"
            self.clear()

        def _build_axes(self):
            self.fig.clear()
            show_sp = self.big and self.app.ed_spec_var.get()
            left = 0.04 if self.big else 0.045
            if show_sp:
                gs = self.fig.add_gridspec(3, 1, height_ratios=[1, 5, 4], hspace=0.08, left=left,
                                           right=0.995, top=0.985, bottom=0.07)
            else:
                gs = self.fig.add_gridspec(2, 1, height_ratios=[1, 5], hspace=0.12, left=left,
                                           right=0.995, top=0.98, bottom=0.07 if self.big else 0.11)
            self.ax_ov = self.fig.add_subplot(gs[0])
            self.ax = self.fig.add_subplot(gs[1])
            self.ax_sp = self.fig.add_subplot(gs[2]) if show_sp else None

        def set_spectrogram(self):
            self._build_axes()
            self.curs, self.bgs = [], None
            if self.peaks is not None:
                self.redraw()
            else:
                self.clear()

        @staticmethod
        def _k(fn, *a):
            fn(*a)
            return "break"

        # ----- data -----
        def load(self, path, duration, peaks, keep_view=False, spec=None):
            same = (path == self.path)
            self.path, self.dur = path, float(duration or 0)
            self.peaks = peaks
            self.spec_img = spec
            self._sp_cache = None
            self.ch = int(peaks.shape[1]) if peaks is not None else 2
            self.ov_cache = None
            if not (same and keep_view):
                self.t0, self.t1 = 0.0, max(self.dur, 1.0)
                self.sel, self.raw = None, None
            self.redraw()

        def clear(self, msg="Drop audio files here"):
            self.path, self.peaks, self.dur = None, None, 0.0
            self.spec_img, self._sp_cache = None, None
            for a in (self.ax, self.ax_ov, self.ax_sp):
                if a is not None:
                    a.clear()
            for a in [x for x in (self.ax, self.ax_ov, self.ax_sp) if x is not None]:
                self.app._style(a)
                a.set_facecolor(WAVE_BG)
                a.set_xticks([])
                a.set_yticks([])
            self.ax.text(0.5, 0.5, msg, ha="center", va="center", color=DIM, fontsize=13,
                         transform=self.ax.transAxes)
            self.cur = self.cur_ov = None
            self.curs, self.bgs = [], None
            self.marks_drawn, self._mark_list = [], []
            self.mark_box["values"] = []
            self.mark_var.set("")
            self.canvas.draw_idle()

        def columns(self):
            return max(200, int(self.ax.bbox.width))

        @staticmethod
        def _reduce(mn, mx, sq, n_out):
            """Reduce per-sample/per-block arrays (n, ch) to n_out columns."""
            n = len(mn)
            edges = np.unique(np.linspace(0, n, n_out + 1).astype(int)[:-1])
            cnt = np.diff(np.append(edges, n))[:, None]
            return (edges, np.minimum.reduceat(mn, edges, axis=0), np.maximum.reduceat(mx, edges, axis=0),
                    np.sqrt(np.add.reduceat(sq, edges, axis=0) / np.maximum(cnt, 1)))

        def raw_samples(self, t0, t1):
            if not self.path or not os.path.exists(self.path):
                return None
            r = self.raw
            if not (r and r[0] <= t0 + 1e-6 and r[1] >= t1 - 1e-6):
                span = t1 - t0
                s0, s1 = max(0.0, t0 - span), min(self.dur, t1 + span)
                a = load_segment(self.path, s0, s1 - s0, self.ch)
                if a is None:
                    return None
                self.raw = r = (s0, s0 + len(a) / float(WAVE_SR), a)
            i0 = max(0, int((t0 - r[0]) * WAVE_SR))
            i1 = min(len(r[2]), int(np.ceil((t1 - r[0]) * WAVE_SR)) + 1)
            return r[0] + i0 / float(WAVE_SR), r[2][i0:i1]

        def envelope(self, t0, t1, nc):
            if self.peaks is not None and (t1 - t0) * PEAK_RATE >= nc * 0.9:
                i0 = max(0, int(t0 * PEAK_RATE))
                i1 = min(len(self.peaks), int(np.ceil(t1 * PEAK_RATE)) + 1)
                blk = self.peaks[i0:i1].astype(np.float32) / 127.0
                if len(blk) == 0:
                    return None
                e, mn, mx, rms = self._reduce(blk[:, :, 0], blk[:, :, 1], blk[:, :, 2] ** 2, nc)
                return "env", ((i0 + e) / float(PEAK_RATE), mn, mx, rms)
            got = self.raw_samples(t0, t1)
            if got is None:
                if self.peaks is None:
                    return None
                i0 = max(0, int(t0 * PEAK_RATE))
                i1 = min(len(self.peaks), int(np.ceil(t1 * PEAK_RATE)) + 1)
                blk = self.peaks[i0:i1].astype(np.float32) / 127.0
                return "env", ((i0 + np.arange(len(blk))) / float(PEAK_RATE), blk[:, :, 0], blk[:, :, 1],
                               blk[:, :, 2])
            start, a = got
            if len(a) > 3 * nc:
                e, mn, mx, rms = self._reduce(a, a, a * a, nc)
                if len(a) < 24 * nc:          # few samples per pixel: loudness band is meaningless
                    rms = None
                return "env", (start + e / float(WAVE_SR), mn, mx, rms)
            return "line", (start + np.arange(len(a)) / float(WAVE_SR), a)

        def spectro(self, t0, t1, nc):
            """Spectrogram for the visible window. Far out: slice of the stored image.
            Zoomed in: recalculated from the audio so it stays sharp."""
            key = (round(t0, 4), round(t1, 4), nc)
            if self._sp_cache and self._sp_cache[0] == key:
                return self._sp_cache[1]
            res = None
            span = t1 - t0
            img = self.spec_img
            if img is not None and self.dur > 0 and span / self.dur * img.shape[1] >= 350:
                C = img.shape[1]
                c0 = max(0, int(t0 / self.dur * C))
                c1 = min(C, int(np.ceil(t1 / self.dur * C)) + 1)
                part = img[:, c0:c1].astype(np.float32) / 255.0 * 90.0 - 90.0
                res = (part, c0 / float(C) * self.dur, c1 / float(C) * self.dur)
            else:
                n_fft = 2048 if span > 3 else (1024 if span > 0.6 else 512)
                pad = n_fft / float(WAVE_SR)
                got = self.raw_samples(max(0.0, t0 - pad), min(self.dur, t1 + pad))
                if got is not None and len(got[1]) >= 64:
                    start, a = got
                    mono = a.mean(axis=1) if a.ndim == 2 else a
                    hop = max(16, int((len(mono) - n_fft) / max(1, nc)))
                    if n_fft not in self._banks:
                        self._banks[n_fft] = tri_bank(n_fft // 2 + 1, WAVE_SR / float(n_fft),
                                                      spectrum_freqs(), 0.5)
                    P = stft_bank(mono, n_fft, hop, self._banks[n_fft], power=True)
                    ref = (np.hanning(n_fft).sum() / 2.0) ** 2
                    db = np.clip(10 * np.log10(P / ref + 1e-12), -90, 0).T
                    x0 = start + n_fft / 2.0 / WAVE_SR
                    res = (db, x0, x0 + P.shape[0] * hop / float(WAVE_SR))
            self._sp_cache = (key, res)
            return res

        # ----- drawing -----
        def marks(self):
            return visible_cues((self.app.view or {}).get("cues"))

        def redraw(self):
            if self.peaks is None or self.dur <= 0:
                if self.path:
                    self.clear("No waveform for this song yet - press Re-analyse")
                return
            ax, ov, sp = self.ax, self.ax_ov, self.ax_sp
            for a in [x for x in (ax, ov, sp) if x is not None]:
                a.clear()
                self.app._style(a)
                a.set_facecolor(WAVE_BG)
            t0, t1, span = self.t0, self.t1, self.t1 - self.t0
            nc = self.columns()
            # overview strip (whole track)
            if self.ov_cache is None:
                blk = self.peaks.astype(np.float32) / 127.0
                e, mn, mx, _r = self._reduce(blk[:, :, 0], blk[:, :, 1], blk[:, :, 2] ** 2, nc)
                amp = np.maximum(np.abs(mn), np.abs(mx)).max(axis=1)
                self.ov_cache = (e / float(PEAK_RATE), amp)
            x, amp = self.ov_cache
            ov.fill_between(x, -amp, amp, color=WAVE_PEAK, lw=0)
            ov.axvspan(t0, t1, color="#ffffff", alpha=0.14, lw=0)
            ov.set_xlim(0, self.dur)
            ov.set_ylim(-1.05, 1.05)
            ov.set_xticks([])
            ov.set_yticks([])
            for c in self.marks():
                ov.axvline(c["t"], color=CUE_STYLE.get(c["kind"], CUE_STYLE["cue"])[1], lw=1.0)
            # main view
            centers = [1.0, -1.0] if self.ch == 2 else [0.0]
            ylim = (-2.05, 2.05) if self.ch == 2 else (-1.05, 1.05)
            sc = 0.95
            env = self.envelope(t0, t1, nc)
            bpm, off = (self.app.view or {}).get("bpm"), (self.app.view or {}).get("beat_offset")
            if bpm and off is not None and span <= 45:
                beat = 60.0 / bpm
                ks = np.arange(int(np.ceil((t0 - off) / beat)), int(np.floor((t1 - off) / beat)) + 1)
                xs = off + ks * beat
                bars = (ks % 4 == 0)
                ax.vlines(xs[~bars], ylim[0], ylim[1], color="#22332a", lw=0.7, zorder=0)
                ax.vlines(xs[bars], ylim[0], ylim[1], color="#46664f", lw=1.1, zorder=0)
            if env is not None:
                mode, data = env
                for k, cy in enumerate(centers):
                    if mode == "env":
                        xx, mn, mx, rms = data
                        ax.fill_between(xx, cy + mn[:, k] * sc, cy + mx[:, k] * sc, color=WAVE_PEAK, lw=0,
                                        zorder=2)
                        if rms is not None:
                            ax.fill_between(xx, cy - rms[:, k] * sc, cy + rms[:, k] * sc, color=WAVE_RMS,
                                            lw=0, zorder=3)
                    else:
                        xx, a = data
                        ax.plot(xx, cy + a[:, k] * sc, color=WAVE_RMS, lw=1.1, zorder=3)
                        if len(xx) < nc / 6:
                            ax.plot(xx, cy + a[:, k] * sc, "o", ms=2.5, color=WAVE_RMS, zorder=4)
            for cy in centers:
                ax.axhline(cy, color="#2c3d31", lw=0.6, zorder=1)
            if self.ch == 2:
                ax.axhline(0, color="#3a3f44", lw=1.0, zorder=1)
            # marks (labels that would overlap step down to the next row)
            ax.set_xlim(t0, t1)
            ax.set_ylim(*ylim)
            self.marks_drawn = []
            fs = 9 if self.big else 7.5
            char_px = fs * 0.72
            rows = []                                   # occupied pixel ranges per row
            for c in self.marks():
                if not (t0 - span * 0.02 <= c["t"] <= t1 + span * 0.02):
                    continue
                lab, col, ls, _m = CUE_STYLE.get(c["kind"], CUE_STYLE["cue"])
                is_sel = c is self.sel
                line = ax.axvline(c["t"], color=col, ls=ls, lw=3.0 if is_sel else 1.6, zorder=5)
                right = c["t"] > t0 + span * 0.85
                text = " %s  %s " % (c.get("label") or lab, fmt_clock(c["t"], span))
                x_px = ax.transData.transform((c["t"], 0))[0]
                w_px = len(text) * char_px + 6
                lo, hi = (x_px - w_px, x_px) if right else (x_px, x_px + w_px)
                row = 0
                while row < len(rows) and any(not (hi < a or lo > b) for a, b in rows[row]):
                    row += 1
                if row == len(rows):
                    rows.append([])
                rows[row].append((lo, hi))
                txt = ax.text(c["t"], 0.985 - min(row, 4) * 0.095, text, transform=ax.get_xaxis_transform(),
                              color="#101010", fontsize=fs, fontweight="bold", va="top",
                              ha="right" if right else "left", zorder=6,
                              bbox=dict(boxstyle="square,pad=0.15", fc=col, alpha=0.95,
                                        ec="white" if is_sel else "none", lw=1.2))
                self.marks_drawn.append((c, line, txt))
            ax.set_ylim(*ylim)
            ax.set_yticks([])
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p, s=span: fmt_clock(v, s)))
            self.cur = ax.axvline(t0, color="#ff5252", lw=1.7, animated=True, zorder=9)
            self.cur_ov = ov.axvline(0, color="#ff5252", lw=1.2, animated=True)
            self.curs = [(ax, self.cur, True), (ov, self.cur_ov, False)]
            if sp is not None:
                got = self.spectro(t0, t1, nc)
                if got is not None:
                    img, x0, x1 = got
                    sp.imshow(img, aspect="auto", origin="lower", cmap=GREEN_CMAP, vmin=-80, vmax=0,
                              extent=[x0, x1, 0, img.shape[0]], interpolation="nearest")
                    freqs = spectrum_freqs()
                    rows = img.shape[0]
                    ticks = [f for f in (50, 100, 200, 500, 1000, 2000, 5000, 10000) if freqs[0] <= f <= freqs[-1]]
                    sp.set_yticks([np.log(f / freqs[0]) / np.log(freqs[-1] / freqs[0]) * (rows - 1) for f in ticks])
                    sp.set_yticklabels([("%dk" % (f // 1000)) if f >= 1000 else str(f) for f in ticks])
                    sp.set_ylim(0, rows)
                else:
                    sp.set_yticks([])
                for c in self.marks():
                    lab, col, ls, _m = CUE_STYLE.get(c["kind"], CUE_STYLE["cue"])
                    sp.axvline(c["t"], color=col, ls=ls, lw=3.0 if c is self.sel else 1.4, zorder=5)
                sp.set_xlim(t0, t1)
                sp.xaxis.set_major_formatter(FuncFormatter(lambda v, p, s=span: fmt_clock(v, s)))
                ax.tick_params(labelbottom=False)
                cur_sp = sp.axvline(t0, color="#ff5252", lw=1.7, animated=True, zorder=9)
                self.curs.append((sp, cur_sp, True))
            self.bgs = None
            self.canvas.draw_idle()
            if self.dur > 0:
                self.hbar.set(t0 / self.dur, t1 / self.dur)
            self.update_mark_list()

        def update_mark_list(self):
            ms = self.marks()
            self._mark_list = ms
            self.mark_box["values"] = ["%s   %s" % (c.get("label"), fmt_ms(c["t"])) for c in ms]
            if self.sel is not None and self.sel in ms:
                self.mark_box.current(ms.index(self.sel))
            else:
                self.mark_var.set("")

        def on_draw(self, _e=None):
            if self.curs:
                self.bgs = [(a, self.canvas.copy_from_bbox(a.bbox)) for a, _l, _z in self.curs]
                self._last = "x"

        def update_cursor(self, pos):
            if not self.bgs or not self.curs:
                return
            key = None if pos is None else round(pos, 3)
            if key == self._last:
                return
            try:
                for _a, bg in self.bgs:
                    self.canvas.restore_region(bg)
                if pos is not None:
                    for a, line, zoomed in self.curs:
                        if not zoomed or self.t0 <= pos <= self.t1:
                            line.set_xdata([pos, pos])
                            a.draw_artist(line)
                for a, _bg in self.bgs:
                    self.canvas.blit(a.bbox)
                self._last = key
            except Exception:
                pass

        def follow(self, pos, playing):
            """Turn the page when the playhead runs off the right edge of the view.
            Never pulls the view away from a place you zoomed to yourself."""
            span = self.t1 - self.t0
            prev, self._prev_pos = getattr(self, "_prev_pos", None), pos
            if (playing and pos is not None and prev is not None and self.follow_var.get()
                    and self.drag is None and span < self.dur * 0.98
                    and self.t0 <= prev <= self.t1 and pos > self.t1 and pos - prev < 2.0):
                self.set_window(self.t1 - 0.02 * span, self.t1 + 0.98 * span)

        # ----- navigation -----
        def set_window(self, t0, t1):
            if self.dur <= 0:
                return
            span = min(max(t1 - t0, self.MIN_SPAN), self.dur)
            t0 = max(0.0, t0)
            if t0 + span > self.dur:
                t0 = max(0.0, self.dur - span)
            self.t0, self.t1 = t0, t0 + span
            self.redraw()

        def zoom(self, f, center=None):
            span = self.t1 - self.t0
            if center is None:
                pos = self.app.player.position() if self.app.player.path == self.path else None
                if self.sel is not None and self.t0 <= self.sel["t"] <= self.t1:
                    center = self.sel["t"]
                elif pos is not None and self.t0 <= pos <= self.t1:
                    center = pos
                else:
                    center = (self.t0 + self.t1) / 2
            new = min(max(span * f, self.MIN_SPAN), self.dur)
            t0 = center - (center - self.t0) * (new / span)
            self.set_window(t0, t0 + new)

        def zoom_all(self):
            self.set_window(0.0, self.dur)

        def zoom_to(self, t, span=4.0):
            self.set_window(t - span / 2, t + span / 2)

        def on_scrollbar(self, *args):
            span = self.t1 - self.t0
            if not args or self.dur <= 0:
                return
            if args[0] == "moveto":
                t0 = float(args[1]) * self.dur
            else:
                step = span * (0.9 if args[2] == "pages" else 0.1)
                t0 = self.t0 + int(args[1]) * step
            self.set_window(t0, t0 + span)

        def on_mark_chosen(self, _e=None):
            i = self.mark_box.current()
            if 0 <= i < len(self._mark_list):
                self.sel = self._mark_list[i]
                self.zoom_to(self.sel["t"], 4.0)
                self.widget.focus_set()

        # ----- mouse -----
        def mark_near(self, px):
            best, bd = None, 7.0
            for c, _l, _t in self.marks_drawn:
                d = abs(self.ax.transData.transform((c["t"], 0))[0] - px)
                if d <= bd:
                    best, bd = c, d
            return best

        def on_press(self, e):
            self.widget.focus_set()
            if self.peaks is None:
                return
            if e.inaxes is self.ax_ov and e.button == 1 and e.xdata is not None:
                self.ov_drag = True
                span = self.t1 - self.t0
                self.set_window(e.xdata - span / 2, e.xdata + span / 2)
                return
            if e.inaxes not in (self.ax, self.ax_sp) or e.inaxes is None or e.xdata is None:
                return
            t = min(max(float(e.xdata), 0.0), self.dur)
            if e.button == 1:
                m = self.mark_near(e.x)
                if m is not None:
                    self.sel, self.drag, self.drag_moved, self.drag_t = m, m, False, m["t"]
                    self.redraw()
                    return
                self.app.play_from(t)
            elif e.button == 3:
                self.app.mark_menu(t, span=self.t1 - self.t0, view=self)

        def on_motion(self, e):
            if self.ov_drag and e.inaxes is self.ax_ov and e.xdata is not None:
                span = self.t1 - self.t0
                self.set_window(e.xdata - span / 2, e.xdata + span / 2)
                return
            if self.drag is not None and e.xdata is not None:
                t = min(max(float(e.xdata), 0.0), self.dur)
                if self.snap_var.get():
                    t = self.app.snap(t)
                self.drag_t, self.drag_moved = t, True
                for c, line, txt in self.marks_drawn:
                    if c is self.drag:
                        line.set_xdata([t, t])
                        txt.set_x(t)
                        txt.set_text(" %s  %s " % (c.get("label"), fmt_clock(t, self.t1 - self.t0)))
                self.canvas.draw_idle()
                return
            if e.inaxes is not None and e.inaxes in (self.ax, self.ax_sp) and e.x is not None and self.marks_drawn:
                self.widget.configure(cursor="sb_h_double_arrow" if self.mark_near(e.x) else "")

        def on_release(self, e):
            self.ov_drag = False
            if self.drag is not None:
                m, self.drag = self.drag, None
                if self.drag_moved:
                    self.app.move_mark(m, self.drag_t, snap=False)

        def on_scroll(self, e):
            if e.inaxes is None or e.inaxes not in (self.ax, self.ax_ov, self.ax_sp) or e.xdata is None \
                    or self.peaks is None:
                return
            ge = getattr(e, "guiEvent", None)
            shift = bool(ge is not None and getattr(ge, "state", 0) & 0x1)
            span = self.t1 - self.t0
            if shift or e.inaxes is self.ax_ov:
                d = -e.step * 0.15 * span
                self.set_window(self.t0 + d, self.t1 + d)
            else:
                self.zoom(0.8 if e.step > 0 else 1.25, center=float(e.xdata))

        # ----- mark editing -----
        def nudge(self, kind, d):
            m = self.sel
            if m is None or m not in self.marks():
                self.app.set_status("Pick a mark first: click its line, or choose it in the Mark list")
                return
            if kind == "ms":
                dt = 0.010 * d
            else:
                bpm = (self.app.view or {}).get("bpm")
                if not bpm:
                    self.app.set_status("No tempo known for this song")
                    return
                dt = d * 60.0 / bpm
            self.app.move_mark(m, m["t"] + dt, snap=False)
            if not (self.t0 <= m["t"] <= self.t1):
                span = self.t1 - self.t0
                self.set_window(m["t"] - span / 2, m["t"] + span / 2)

        def delete_selected(self):
            if self.sel is None or self.sel not in self.marks():
                self.app.set_status("Pick a mark first: click its line, or choose it in the Mark list")
                return
            m, self.sel = self.sel, None
            self.app.remove_mark(m)

    class EditorWindow(tk.Toplevel):
        """Big separate window for fine-tuning the marks of the current song."""

        def __init__(self, app):
            super().__init__(app.root)
            self.app = app
            self.title("SongKey - mark editor")
            self.configure(bg=PANEL)
            sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
            self.geometry("%dx%d+10+10" % (max(900, int(sw * 0.96)), max(600, int(sh * 0.88))))
            self.protocol("WM_DELETE_WINDOW", self.close)
            top = ttk.Frame(self, style="Panel.TFrame", padding=8)
            top.pack(fill="both", expand=True)

            hdr = ttk.Frame(top, style="Panel.TFrame")
            hdr.pack(fill="x")
            self.lbl = ttk.Label(hdr, text="", style="Title.TLabel")
            self.lbl.pack(side="left")
            self.lbl_info = ttk.Label(hdr, text="", style="Dim.TLabel")
            self.lbl_info.pack(side="left", padx=14)
            ttk.Checkbutton(hdr, text="Show spectrogram", variable=app.ed_spec_var,
                            command=lambda: self.wave.set_spectrogram(),
                            style="Panel.TCheckbutton").pack(side="right")

            tp = ttk.Frame(top, style="Panel.TFrame")
            tp.pack(side="bottom", fill="x", pady=(6, 0))
            self.btn_play = icon_button(app, tp, "Play", "play", app.play_pause, tip="Play / pause  (space)",
                                        width=7)
            self.btn_play.pack(side="left")
            icon_button(app, tp, "Stop", "stop", app.stop_play, tip="Stop").pack(side="left", padx=4)
            icon_button(app, tp, "Mark", "prev_mark", lambda: app.jump_mark(-1),
                        tip="Play from the previous mark").pack(side="left")
            icon_button(app, tp, "Mark", "next_mark", lambda: app.jump_mark(1), tip="Play from the next mark",
                        compound="right").pack(side="left", padx=4)
            self.lbl_time = ttk.Label(tp, text="0:00 / 0:00", style="Panel.TLabel", width=22)
            self.lbl_time.pack(side="left", padx=10)
            ttk.Label(tp, text=("click: play   drag mark: move   wheel: zoom   "
                                "arrows: 10 ms   Shift+arrows: 1 beat   I/O: mark at playhead")[0:200],
                      style="Dim.TLabel").pack(side="left")

            self.wave = WaveView(app, top, big=True, snap_var=app.wave.snap_var, follow_var=app.wave.follow_var)
            self.wave.frame.pack(fill="both", expand=True, pady=(6, 0))
            self.bind("<space>", app._space)
            for k, kind in (("<Key-i>", "in"), ("<Key-I>", "in"), ("<Key-o>", "out"), ("<Key-O>", "out")):
                self.bind(k, lambda e, kind=kind: app._key_mark(kind, e))
            self.bind("<Escape>", lambda e: self.close())
            self.refresh()

        def refresh(self, keep_view=True):
            a = self.app
            self.lbl.configure(text=a.lbl_title.cget("text"))
            v = a.view or {}
            info = []
            if v.get("bpm"):
                info.append("%s BPM" % fmt_bpm(v["bpm"]))
            if a._key_text:
                info.append(a._key_text)
            self.lbl_info.configure(text="   |   ".join(info))
            if a._wave_args:
                path, dur, peaks, spec = a._wave_args
                self.wave.load(path, dur, peaks, keep_view=keep_view, spec=spec)
            else:
                self.wave.clear()

        def close(self):
            self.app.editor = None
            self.destroy()

    class TapWindow(tk.Toplevel):
        """Tap the tempo by hand while the song plays."""

        def __init__(self, app):
            super().__init__(app.root)
            self.app = app
            v = app.view
            self.ref, self.path = v["ref"], v["path"]
            self.dur = v.get("duration") or 0
            self.app_bpm = v.get("bpm")
            self.taps, self.mult, self.result = [], 1.0, None
            self.title("SongKey - tap the tempo")
            self.configure(bg=PANEL)
            self.transient(app.root)
            self.resizable(False, False)
            self.protocol("WM_DELETE_WINDOW", self.close)

            frm = ttk.Frame(self, style="Panel.TFrame", padding=16)
            frm.pack(fill="both", expand=True)
            ttk.Label(frm, text=app.lbl_title.cget("text"), style="Title.TLabel", wraplength=560).pack(anchor="w")
            ttk.Label(frm, text="The app measured:  %s BPM" % (fmt_bpm(self.app_bpm) or "?"),
                      style="Dim.TLabel").pack(anchor="w", pady=(2, 10))

            tb = ttk.Frame(frm, style="Panel.TFrame")
            tb.pack(fill="x")
            self.btn_play = icon_button(app, tb, "Play", "play", self.play_pause, tip="Play / pause  (Enter)",
                                        width=7)
            self.btn_play.pack(side="left")
            icon_button(app, tb, "Stop", "stop", app.player.stop, tip="Stop").pack(side="left", padx=4)
            self.lbl_time = ttk.Label(tb, text="0:00 / %s" % fmt_time(self.dur), style="Panel.TLabel")
            self.lbl_time.pack(side="left", padx=10)

            self.pad = tk.Label(frm, text="TAP\n\nspace bar on every beat  -  or click here",
                                font=("Helvetica", 15, "bold"), bg="#22302a", fg="#9cf0a9",
                                height=5, relief="ridge", bd=2, cursor="hand2")
            self.pad.pack(fill="x", pady=12)
            self.pad.bind("<ButtonPress-1>", self.tap)

            self.lbl_bpm = ttk.Label(frm, text="--", style="Big.TLabel", anchor="center")
            self.lbl_bpm.pack(fill="x")
            self.lbl_info = ttk.Label(frm, text="Press Play, then tap the space bar on every beat (follow the kick drum).",
                                      style="Panel.TLabel", wraplength=560, justify="center", anchor="center")
            self.lbl_info.pack(fill="x", pady=(4, 2))
            self.lbl_rel = ttk.Label(frm, text="", style="Cue.TLabel", wraplength=560, justify="center",
                                     anchor="center")
            self.lbl_rel.pack(fill="x", pady=(0, 8))

            row = ttk.Frame(frm, style="Panel.TFrame")
            row.pack()
            ttk.Button(row, text="half", width=6, command=lambda: self.scale(0.5)).pack(side="left", padx=2)
            ttk.Button(row, text="double", width=7, command=lambda: self.scale(2.0)).pack(side="left", padx=2)
            ttk.Button(row, text="Start over", width=10, command=self.reset).pack(side="left", padx=(12, 2))
            self.round_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(row, text="Round to a whole number", variable=self.round_var, command=self.show,
                            style="Panel.TCheckbutton").pack(side="left", padx=(14, 0))

            bot = ttk.Frame(frm, style="Panel.TFrame")
            bot.pack(fill="x", pady=(14, 0))
            ttk.Button(bot, text="Close", command=self.close).pack(side="right")
            self.btn_use = ttk.Button(bot, text="Use this BPM", style="Accent.TButton", command=self.use)
            self.btn_use.pack(side="right", padx=6)
            self.btn_use.state(["disabled"])

            self.bind("<space>", self.tap)
            self.bind("<Return>", lambda e: self.play_pause())
            self.bind("<Escape>", lambda e: self.close())
            self.update_idletasks()
            self.geometry("+%d+%d" % (app.root.winfo_rootx() + 160, app.root.winfo_rooty() + 80))
            self.focus_force()
            self._alive = True
            self.tick()

        def play_pause(self):
            pl = self.app.player
            if not self.app.player_ok or not self.path:
                return
            if pl.path == self.path and pl.playing():
                pl.pause()
            elif pl.path == self.path and pl.paused_at is not None:
                pl.resume()
            elif os.path.exists(self.path):
                pl.volume = self.app.vol.get()
                pl.play(self.path, 0.0)
            self.focus_force()

        def tick(self):
            if not self._alive:
                return
            pl = self.app.player
            mine = pl.path == self.path
            playing = mine and pl.playing()
            pos = pl.position() if mine else None
            self.btn_play.configure(text="Pause" if playing else "Play",
                                    image=self.app.icons.get("pause" if playing else "play", ""))
            self.lbl_time.configure(text="%s / %s" % (fmt_time(pos or 0), fmt_time(self.dur)))
            self.after(150, self.tick)

        def tap(self, e=None):
            t = (e.time / 1000.0) if (e is not None and getattr(e, "time", 0)) else time.monotonic()
            if self.taps and (t - self.taps[-1] > 2.5 or t < self.taps[-1]):
                self.taps, self.mult = [], 1.0          # long pause: start a new count
            if self.taps and t - self.taps[-1] < 0.12:
                return "break"                          # key bounce
            self.taps.append(t)
            self.pad.configure(bg="#3fbf5f", fg="#0e1210")
            self.after(70, lambda: self.pad.configure(bg="#22302a", fg="#9cf0a9"))
            self.show()
            return "break"

        def scale(self, f):
            if self.result:
                self.mult *= f
                self.show()

        def reset(self):
            self.taps, self.mult, self.result = [], 1.0, None
            self.show()

        def value(self):
            if not self.result:
                return None
            v = self.result * self.mult
            return float(round(v)) if self.round_var.get() else round(v, 1)

        def show(self):
            n = len(self.taps)
            r = tap_tempo(self.taps)
            if r is None:
                self.result = None
                self.lbl_bpm.configure(text="--")
                self.lbl_info.configure(text=("Press Play, then tap the space bar on every beat (follow the kick drum)."
                                              if n == 0 else "%d tap%s - keep going..." % (n, "" if n == 1 else "s")))
                self.lbl_rel.configure(text="")
                self.btn_use.state(["disabled"])
                return
            bpm, err, used = r
            self.result = bpm
            shown = bpm * self.mult
            err *= self.mult
            if err < 0.15:
                q = "very precise"
            elif err < 0.5:
                q = "good - a few more taps make it more precise"
            elif err < 1.5:
                q = "rough - keep tapping"
            else:
                q = "uneven taps - try to follow the kick drum"
            self.lbl_bpm.configure(text="%.1f" % shown)
            extra = ""
            if self.round_var.get() and abs(shown - round(shown)) >= 0.05:
                extra = "   (will be saved as %d)" % round(shown)
            self.lbl_info.configure(text="%d taps   |   about +/- %.1f BPM   |   %s%s" % (used, err, q, extra))
            self.lbl_rel.configure(text=tap_relation(shown, self.app_bpm))
            self.btn_use.state(["!disabled"])

        def use(self):
            v = self.value()
            if not v:
                return
            if not 20 <= v <= 400:
                self.lbl_info.configure(text="That tempo is out of range (20-400 BPM)")
                return
            self.app.set_bpm(self.ref, v)
            self.close()

        def close(self):
            self._alive = False
            self.app.tapwin = None
            self.destroy()

    class AboutWindow(tk.Toplevel):
        def __init__(self, app):
            super().__init__(app.root)
            self.title("About SongKey")
            self.configure(bg=PANEL)
            self.transient(app.root)
            self.resizable(False, False)
            frm = ttk.Frame(self, style="Panel.TFrame", padding=22)
            frm.pack(fill="both", expand=True)
            head = ttk.Frame(frm, style="Panel.TFrame")
            head.pack(anchor="w")
            if app.icons.get("wave"):
                tk.Label(head, image=app.icons["wave"], bg=PANEL).pack(side="left", padx=(0, 8))
            ttk.Label(head, text="SongKey", style="Big.TLabel").pack(side="left")
            ttk.Label(head, text="  version %s" % APP_VERSION, style="Dim.TLabel").pack(side="left", anchor="s",
                                                                                     pady=(0, 8))
            ttk.Label(frm, text=("Finds the tempo (BPM) and musical key (Camelot) of your songs, shows the "
                                 "waveform and spectrum, and marks where to mix in and out."),
                      style="Panel.TLabel", wraplength=470, justify="left").pack(anchor="w", pady=(6, 14))

            def row(label, value, style="Panel.TLabel"):
                r = ttk.Frame(frm, style="Panel.TFrame")
                r.pack(anchor="w", pady=2)
                ttk.Label(r, text=label, style="Dim.TLabel", width=12).pack(side="left")
                w = ttk.Label(r, text=value, style=style)
                w.pack(side="left")
                return w
            row("Idea:", "Pedro (peterpt)")
            row("Built by:", "Claude Opus 5.5, an AI model from Anthropic")
            link = row("Project:", GITHUB_URL, style="Link.TLabel")
            link.configure(cursor="hand2")
            link.bind("<Button-1>", lambda e: self.open_link())
            Tooltip(link, "Open the project page in your web browser")
            eng = "Essentia" if app._essentia_ok else None
            row("Built with:", "Python, Tk, NumPy, SciPy, Matplotlib, FFmpeg" + (", " + eng if eng else ""))
            ttk.Label(frm, text="Pedro had the idea - Claude wrote the code.", style="Cue.TLabel").pack(
                anchor="w", pady=(14, 0))
            ttk.Button(frm, text="Close", command=self.destroy).pack(anchor="e", pady=(16, 0))
            self.bind("<Escape>", lambda e: self.destroy())
            self.update_idletasks()
            self.geometry("+%d+%d" % (app.root.winfo_rootx() + 200, app.root.winfo_rooty() + 120))
            self.focus_force()

        @staticmethod
        def open_link():
            import webbrowser
            try:
                webbrowser.open(GITHUB_URL)
            except Exception:
                pass

    # ------------------------------------------------------------------ app
    class App:
        LIB_COLS = ("artist", "title", "mix", "genre", "bpm", "key", "camelot", "duration",
                    "format", "size")
        NEW_COLS = ("title", "artist", "bpm", "key", "camelot", "duration", "status", "file")

        def __init__(self, root):
            self.root = root
            self.db = SongDB(args.db)
            self.q = queue.Queue()
            self.engine = args.engine
            self.worker = Worker(self.q, self.engine, args.bpm_range)
            self.worker.start()
            self.pending = 0
            self.failed = []
            self.last_done = ""
            self.session = {}            # path -> analysis record (not in the library yet)
            self.current = None          # ("new", path) or ("lib", id)
            self.ask_queue = []
            self.dialog_open = False
            self.compat_only = tk.BooleanVar(value=False)
            self.ask_each = tk.BooleanVar(value=False)
            self.search = tk.StringVar()
            self.sort = {"lib": (None, False), "new": (None, False)}
            self.player = Player()
            self.player_ok = Player.available()
            self.icons = {}
            try:
                from PIL import ImageTk
                for k, im in make_icon_images(18).items():
                    self.icons[k] = ImageTk.PhotoImage(im, master=root)
            except Exception:
                self.icons = {}          # no Pillow: plain text buttons
            self.editor = None
            self.tapwin = None
            self._essentia_ok = True if self.engine == "essentia" else None
            self.ed_spec_var = tk.BooleanVar(value=True)
            self._wave_args = None
            self._key_text = ""
            self.view = None              # what the chart shows: path, duration, cues, bpm...
            self.ax_spec = self.cursor = self.bg = None
            self._last_cursor = "x"
            self._spec_args = None
            self._spec_dirty = False
            self._build()
            self.refresh_tables()
            self.draw_empty()
            self.canvas.mpl_connect("draw_event", self._on_draw)
            self.canvas.mpl_connect("button_press_event", self.on_click)
            self.root.bind("<space>", self._space)
            for k, kind in (("<Key-i>", "in"), ("<Key-I>", "in"), ("<Key-o>", "out"), ("<Key-O>", "out")):
                self.root.bind(k, lambda e, kind=kind: self._key_mark(kind, e))
            self.root.protocol("WM_DELETE_WINDOW", self.on_close)
            self.root.after(100, self.poll)
            self.root.after(120, self.tick)
            self.root.after(150, self.vu_tick)

        # ---------------------------------------------------------- layout
        def _build(self):
            r = self.root
            r.title(APP_NAME)
            r.geometry("1250x860")
            r.configure(bg=BG)
            st = ttk.Style(r)
            st.theme_use("clam")
            st.configure(".", background=BG, foreground=FG, fieldbackground=PANEL)
            st.configure("TFrame", background=BG)
            st.configure("Panel.TFrame", background=PANEL)
            st.configure("TLabel", background=BG, foreground=FG)
            st.configure("Panel.TLabel", background=PANEL, foreground=FG)
            st.configure("Dim.TLabel", background=PANEL, foreground=DIM)
            st.configure("Cue.TLabel", background=PANEL, foreground="#9cf0a9")
            st.configure("Link.TLabel", background=PANEL, foreground="#4fc3f7", font=("Helvetica", 10, "underline"))
            st.configure("Horizontal.TScale", background=PANEL, troughcolor=BG)
            st.configure("Big.TLabel", background=PANEL, foreground=FG, font=("Helvetica", 30, "bold"))
            st.configure("Title.TLabel", background=PANEL, foreground=FG, font=("Helvetica", 14, "bold"))
            st.configure("TButton", background="#33363d", foreground=FG, padding=(10, 5))
            st.map("TButton", background=[("active", "#3f434b"), ("disabled", "#2a2c31")],
                   foreground=[("disabled", "#666")])
            st.configure("Accent.TButton", background="#2e7d32", foreground="white")
            st.configure("Small.TButton", background="#33363d", foreground=FG, padding=(6, 2))
            st.map("Small.TButton", background=[("active", "#3f434b")])
            st.map("Accent.TButton", background=[("active", "#388e3c"), ("disabled", "#2a2c31")])
            st.configure("TCheckbutton", background=BG, foreground=FG)
            st.map("TCheckbutton", background=[("active", BG)])
            st.configure("Panel.TCheckbutton", background=PANEL, foreground=FG)
            st.map("Panel.TCheckbutton", background=[("active", PANEL)])
            st.configure("Treeview", background=PANEL, fieldbackground=PANEL, foreground=FG,
                         rowheight=24, borderwidth=0)
            st.configure("Treeview.Heading", background="#33363d", foreground=FG)
            st.map("Treeview", background=[("selected", "#3d5a80")])
            st.configure("TEntry", fieldbackground=BG, foreground=FG, insertcolor=FG)
            st.configure("TCombobox", fieldbackground=BG, foreground=FG, arrowcolor=FG)
            st.map("TCombobox", fieldbackground=[("readonly", BG)], foreground=[("readonly", FG)])
            st.configure("TNotebook", background=BG, borderwidth=0)
            st.configure("TNotebook.Tab", background="#33363d", foreground=FG, padding=(12, 4))
            st.map("TNotebook.Tab", background=[("selected", "#3d5a80")])
            r.option_add("*TCombobox*Listbox.background", PANEL)
            r.option_add("*TCombobox*Listbox.foreground", FG)

            bar = ttk.Frame(r, padding=(8, 8, 8, 4))
            bar.pack(fill="x")
            cmp_ = Compactor(bar)
            for text, icon, cmd, tip, padx in (
                    ("Open files", "open_file", self.open_files,
                     "Analyse audio files (you can also drag & drop them)", 0),
                    ("Open folder", "folder", self.open_folder, "Analyse every audio file in a folder", 4),
                    ("Edit", "edit", self.edit_selected, "Edit the details of the selected song(s)", (12, 0)),
                    ("Re-analyse", "refresh", self.reanalyse,
                     "Analyse the selected songs again (your names and marks are kept)", 4),
                    ("Remove", "trash", self.remove,
                     "Remove the selected songs from the list (audio files are not touched)", 0),
                    ("Export CSV", "export", self.export, "Save the whole library as a spreadsheet file", 4)):
                b = icon_button(self, bar, text, icon, cmd, tip=tip)
                b.pack(side="left", padx=padx)
                cmp_.add_button(b)
            cmp_.add_button(icon_button(self, bar, "About", "info", lambda: AboutWindow(self), tip="About SongKey"))
            cmp_.items[-1][0].pack(side="right", padx=(8, 0))
            ttk.Checkbutton(bar, text="Only compatible", variable=self.compat_only,
                            command=self.refresh_tables).pack(side="right")
            ent = ttk.Entry(bar, textvariable=self.search, width=16)
            ent.pack(side="right", padx=8)
            ttk.Label(bar, text="Search:").pack(side="right")
            self.search.trace_add("write", lambda *a: self.refresh_tables())

            sbar = ttk.Frame(r, padding=(8, 2, 8, 6))
            sbar.pack(side="bottom", fill="x")
            self.status = ttk.Label(sbar, text="")
            self.status.pack(side="left")
            self.prog = ttk.Progressbar(sbar, mode="indeterminate", length=160)
            self.prog.pack(side="right")
            ttk.Checkbutton(sbar, text="Ask me to add each song after analysis",
                            variable=self.ask_each).pack(side="right", padx=12)
            self.engine_var = tk.StringVar(value="Essentia" if self.engine == "essentia" else "Built-in")
            ecb = ttk.Combobox(sbar, textvariable=self.engine_var, values=["Built-in", "Essentia"],
                               state="readonly", width=9)
            ecb.pack(side="right")
            ecb.bind("<<ComboboxSelected>>", self.on_engine)
            Tooltip(ecb, "Which analysis engine measures BPM and key for new analyses and Re-analyse.\n"
                         "Built-in always works. Essentia must be installed separately.")
            ttk.Label(sbar, text="Engine:").pack(side="right", padx=(0, 4))
            self.status.pack_forget()          # long messages get cut, not the controls on the right
            self.status.pack(side="left", fill="x", expand=True)

            pw = ttk.PanedWindow(r, orient="vertical")
            pw.pack(fill="both", expand=True, padx=8, pady=4)

            top = ttk.Frame(pw, style="Panel.TFrame", padding=8)
            info = ttk.Frame(top, style="Panel.TFrame")
            info.pack(fill="x")
            # right column is packed FIRST so a long title can never push it off the window
            right = ttk.Frame(info, style="Panel.TFrame")
            right.pack(side="right", anchor="n")
            big = ttk.Frame(right, style="Panel.TFrame")
            big.pack(anchor="e")
            self.lbl_bpm = ttk.Label(big, text="--", style="Big.TLabel")
            self.lbl_bpm.pack(side="left", padx=12)
            ttk.Label(big, text="BPM", style="Dim.TLabel").pack(side="left")
            self.badge = tk.Label(big, text="--", font=("Helvetica", 26, "bold"), width=4,
                                  bg="#555555", fg="#111111", padx=6, pady=2)
            self.badge.pack(side="left", padx=(18, 4))
            row2 = ttk.Frame(right, style="Panel.TFrame")
            row2.pack(anchor="e", pady=(8, 0))
            btns = ttk.Frame(row2, style="Panel.TFrame")
            btns.pack(side="right")
            self.btn_main = icon_button(self, btns, "Add to library", "add", self.main_button,
                                        style="Accent.TButton")
            self.btn_main.pack(anchor="e")
            self.btn_main.state(["disabled"])
            self.btn_tap = icon_button(self, btns, "Tap BPM", "tap", self.open_tap,
                                       tip="Tap the tempo yourself while the song plays")
            self.btn_tap.pack(anchor="e", pady=(4, 0))
            self.btn_tap.state(["disabled"])
            self.vu = VUMeter(row2)
            self.vu.c.pack(side="right", padx=(0, 12))
            Tooltip(self.vu.c, "Volume level while the song plays (L / R)")

            left = ttk.Frame(info, style="Panel.TFrame")
            left.pack(side="left", fill="x", expand=True)
            self.lbl_title = ttk.Label(left, text="Drop audio files anywhere in this window",
                                       style="Title.TLabel")
            self.lbl_title.pack(anchor="w")
            self._title_tip = Tooltip(self.lbl_title, "")
            self.lbl_sub = ttk.Label(left, text="", style="Dim.TLabel")
            self.lbl_sub.pack(anchor="w")
            self.lbl_bpm_txt = ttk.Label(left, text="", style="Panel.TLabel")
            self.lbl_bpm_txt.pack(anchor="w", pady=(4, 0))
            self.lbl_key_txt = ttk.Label(left, text="", style="Panel.TLabel")
            self.lbl_key_txt.pack(anchor="w")
            self.lbl_mix = ttk.Label(left, text="", style="Dim.TLabel")
            self.lbl_mix.pack(anchor="w")
            self.lbl_cues = ttk.Label(left, text="", style="Cue.TLabel")
            self.lbl_cues.pack(anchor="w", pady=(2, 0))

            tp = ttk.Frame(top, style="Panel.TFrame")
            tp.pack(side="bottom", fill="x", pady=(4, 0))
            self.btn_play = icon_button(self, tp, "Play", "play", self.play_pause, tip="Play / pause  (space)",
                                        width=7)
            self.btn_play.pack(side="left")
            icon_button(self, tp, "Stop", "stop", self.stop_play, tip="Stop").pack(side="left", padx=4)
            icon_button(self, tp, "Mark", "prev_mark", lambda: self.jump_mark(-1),
                        tip="Play from the previous mark").pack(side="left")
            icon_button(self, tp, "Mark", "next_mark", lambda: self.jump_mark(1), tip="Play from the next mark",
                        compound="right").pack(side="left", padx=4)
            self.lbl_time = ttk.Label(tp, text="0:00 / 0:00", style="Panel.TLabel", width=18)
            self.lbl_time.pack(side="left", padx=8)
            hint = ttk.Label(tp, text=("click: play   drag mark: move   wheel: zoom   "
                                       "right-click: marks") if self.player_ok else
                             "Playback needs ffplay:  sudo apt install ffmpeg", style="Dim.TLabel")
            self.vol = tk.IntVar(value=80)
            vs = ttk.Scale(tp, from_=0, to=100, variable=self.vol, orient="horizontal", length=110)
            vs.pack(side="right")
            vs.bind("<ButtonRelease-1>", self.set_volume)
            Tooltip(vs, "Volume")
            ttk.Label(tp, text="Volume", style="Dim.TLabel").pack(side="right", padx=4)
            hint.pack(side="left")             # packed last: it gets cut on small screens, not the slider
            if not self.player_ok:
                self.btn_play.state(["disabled"])

            self.viewnb = ttk.Notebook(top)
            self.viewnb.pack(fill="both", expand=True, pady=(6, 0))
            self.wave = WaveView(self, self.viewnb)
            self.viewnb.add(self.wave.frame, text="Waveform")
            specf = ttk.Frame(self.viewnb, style="Panel.TFrame")
            self.viewnb.add(specf, text="Spectrum & notes")
            self.fig = Figure(figsize=(10, 4.0), dpi=96, facecolor=PANEL)
            self.canvas = FigureCanvasTkAgg(self.fig, master=specf)
            self.canvas.get_tk_widget().pack(fill="both", expand=True)
            self.viewnb.bind("<<NotebookTabChanged>>", self.on_view_tab)
            pw.add(top, weight=3)

            self.nb = ttk.Notebook(pw)
            self.tree_new = self._make_tree(self.nb, self.NEW_COLS, "new", {
                "title": 260, "artist": 180, "bpm": 70, "key": 60, "camelot": 75,
                "duration": 70, "status": 160, "file": 280})
            self.tree_lib = self._make_tree(self.nb, self.LIB_COLS, "lib", {
                "artist": 170, "title": 220, "mix": 170, "genre": 110, "bpm": 60, "key": 55,
                "camelot": 70, "duration": 60, "format": 80, "size": 75})
            self.nb.add(self.tree_new.master, text="Just analysed")
            self.nb.add(self.tree_lib.master, text="Library")
            self.nb.bind("<<NotebookTabChanged>>", self.on_tab)
            pw.add(self.nb, weight=2)

            self.dnd = self._setup_dnd([r, top, self.canvas.get_tk_widget(), self.wave.widget, self.tree_new,
                                        self.tree_lib, info, self.nb])
            eng = "Essentia" if self.engine == "essentia" else "built-in"
            self.set_status("Ready - %d songs in library  |  engine: %s  |  drag & drop: %s  |  DB: %s" % (
                self.db.count(), eng, "on" if self.dnd else "off (apt install tkdnd)", self.db.path))

        def _make_tree(self, parent, cols, which, widths):
            box = ttk.Frame(parent)
            tree = ttk.Treeview(box, columns=cols, show="headings", selectmode="extended")
            for c in cols:
                tree.heading(c, text=c.capitalize(), command=lambda c=c, w=which: self.sort_by(w, c))
                tree.column(c, width=widths[c], anchor="center" if c in
                            ("bpm", "key", "camelot", "duration", "format", "size") else "w")
            sb = ttk.Scrollbar(box, orient="vertical", command=tree.yview)
            tree.configure(yscrollcommand=sb.set)
            tree.pack(side="left", fill="both", expand=True)
            sb.pack(side="right", fill="y")
            tree.bind("<<TreeviewSelect>>", lambda e, w=which: self.on_select(w))
            tree.bind("<Double-1>", lambda e, w=which: self.on_double(w, e))
            tree.bind("<Delete>", lambda e: self.remove())
            tree.bind("<Return>", lambda e, w=which: self.edit_selected())
            return tree

        def _setup_dnd(self, widgets):
            tkr = self.root.tk
            cands = []
            try:
                import struct
                import tkinterdnd2
                base = os.path.join(os.path.dirname(tkinterdnd2.__file__), "tkdnd")
                bits = struct.calcsize("P") * 8
                mach = getattr(os, "uname", lambda: None)()
                mach = mach.machine if mach else ""
                if IS_WINDOWS:
                    plat = "win-x64" if bits == 64 else "win-x86"
                elif sys.platform == "darwin":
                    plat = "osx-arm64" if mach == "arm64" else "osx-x64"
                else:
                    plat = "linux-arm64" if mach in ("aarch64", "arm64") else "linux-x64"
                if os.path.isdir(os.path.join(base, plat)):
                    cands.append(os.path.join(base, plat))      # only the variant for this platform
                else:
                    cands += [base] + glob.glob(os.path.join(base, "*"))
            except Exception:
                pass
            for pat in ("/usr/lib/tkdnd*", "/usr/lib/tcltk/*/tkdnd*", "/usr/lib/tcltk/tkdnd*",
                        "/usr/lib/*/tkdnd*", "/usr/share/tcltk/tkdnd*", "/usr/local/lib/tkdnd*"):
                for d in glob.glob(pat):
                    cands += [d, os.path.dirname(d)]
            for d in cands:
                try:
                    tkr.call("lappend", "auto_path", d)
                except tk.TclError:
                    pass
            try:
                tkr.call("package", "require", "tkdnd")
            except tk.TclError:
                return False
            cmd = self.root.register(self._on_drop)
            for w in widgets:
                try:
                    tkr.call("tkdnd::drop_target", "register", w._w, ("DND_Files",))
                    tkr.call("bind", w._w, "<<Drop:DND_Files>>", cmd + " %D")
                except tk.TclError:
                    pass
            return True

        def _on_drop(self, data):
            files = list(self.root.tk.splitlist(data))
            files = [f[7:] if f.startswith("file://") else f for f in files]
            self.enqueue(files)
            return "copy"

        # ---------------------------------------------------------- jobs
        def enqueue(self, paths, force=False):
            files = expand_paths(paths)
            if not files:
                self.set_status("No audio files found in what you dropped")
                return
            added, known = 0, None
            for f in files:
                f = os.path.abspath(f)
                lib_id = library_id_if_unchanged(self.db, f)
                if lib_id and not force:
                    known = ("lib", lib_id)
                    continue
                if f in self.session and not force:
                    known = ("new", f)
                    continue
                self.worker.jobs.put((f, None))
                added += 1
            self.pending += added
            if added:
                self.prog.start(12)
                self.set_status("Analysing %d file(s)..." % added)
            elif known:
                self.select(known)
                self.set_status("Already analysed - showing it (use Re-analyse to redo it)")

        def poll(self):
            try:
                while True:
                    kind, data = self.q.get_nowait()
                    if kind == "status":
                        self.set_status("[%d left] %s" % (self.pending, data))
                    elif kind == "spectrum":
                        d, song_id = data
                        busy = self.player.playing()
                        if song_id is None:
                            self.session[d["path"]] = dict(d, status="analysing...")
                            self.refresh_new()
                            if not busy:
                                self.select(("new", d["path"]), show=False)
                                self.current = ("new", d["path"])
                        if not busy:
                            self.view = {"ref": self.current, "path": d["path"], "duration": d["duration"],
                                         "cues": [], "bpm": None, "beat_offset": None}
                            self.show_header(d, partial=True)
                            self._wave_args = (d["path"], d["duration"], d.get("peaks"), d.get("spec"))
                            for wv in self.waves():
                                wv.load(d["path"], d["duration"], d.get("peaks"), spec=d.get("spec"))
                            if self.editor is not None:
                                self.editor.refresh()
                            self.draw(d["spec"], d["avg"], d["duration"], None, None, None)
                    elif kind == "done":
                        rec, song_id = data
                        self.pending = max(0, self.pending - 1)
                        busy = self.player.playing()
                        if song_id is None:
                            rec["status"] = "ready - not in library"
                            self.session[rec["path"]] = rec
                            self.refresh_new()
                            # while music plays, don't let ANOTHER song take over the screen -
                            # but the song already on screen always gets its results
                            if not busy or self.current == ("new", rec["path"]):
                                self.select(("new", rec["path"]))
                            if self.ask_each.get():
                                self.ask_queue.append(rec["path"])
                                self.process_ask_queue()
                        else:
                            old = self.db.get(song_id) or {}
                            rec["cues"] = merge_marks(parse_cues(old.get("cues")), rec.get("cues", []))
                            self.db.update_analysis(song_id, rec)
                            self.refresh_lib()
                            if not busy or self.current == ("lib", song_id):
                                self.select(("lib", song_id))
                        self.last_done = "Done: %s  ->  %s BPM, %s (%s) in %ss" % (
                            rec["filename"], fmt_bpm(rec["bpm"]), rec["key_name"], rec["camelot"],
                            rec["secs"])
                        self.set_status(self.last_done + (
                            "" if song_id else "   - press '+ Add to library' to keep it"))
                    elif kind == "error":
                        path, song_id, err = data
                        self.pending = max(0, self.pending - 1)
                        if song_id is None:
                            self.session.pop(path, None)
                            self.refresh_new()
                        self.failed.append("%s (%s)" % (os.path.basename(path), err))
                        self.set_status("Failed: %s - %s" % (os.path.basename(path), err))
                    elif kind == "essentia":
                        ok, why = data
                        self._essentia_ok = ok
                        if ok:
                            self.set_engine("essentia")
                        else:
                            self.engine_var.set("Built-in")
                            self.set_status("Essentia can't be used - staying with the built-in engine")
                            messagebox.showwarning(APP_NAME, "Essentia can't be used on this computer:\n\n%s\n\n"
                                                   "The built-in engine stays active." % why)
                    elif kind == "idle" and self.pending == 0:
                        self.prog.stop()
                        if self.failed:
                            self.set_status("Finished. %s  |  %d failed: %s" % (
                                self.last_done, len(self.failed), "; ".join(self.failed)[:200]))
                            self.failed = []
            except queue.Empty:
                pass
            self.root.after(100, self.poll)

        # ---------------------------------------------------------- adding
        def process_ask_queue(self):
            if self.dialog_open:
                return
            while self.ask_queue:
                p = self.ask_queue.pop(0)
                if p in self.session and "bpm" in self.session[p]:
                    self.add_paths([p], on_close=self.process_ask_queue)
                    return

        def add_paths(self, paths, on_close=None):
            ready = [p for p in paths if p in self.session and "bpm" in self.session[p]]
            if not ready:
                self.set_status("Nothing to add - pick a finished song in 'Just analysed'")
                return
            if len(ready) == 1:
                p = ready[0]
                try:
                    meta = read_metadata(p)
                except OSError as e:
                    messagebox.showerror(APP_NAME, "Can't read the file:\n%s" % e)
                    return
                rec = dict(self.session[p])
                rec.update(meta)
                self.dialog_open = True

                def closed():
                    self.dialog_open = False
                    if on_close:
                        on_close()
                EditDialog(self, "add", [rec], lambda ch, rec=rec: self._save_new(rec, ch),
                           on_close=closed)
            else:
                ids = []
                for p in ready:
                    try:
                        rec = dict(self.session[p])
                        rec.update(read_metadata(p))
                        ids.append(self._save_new(rec, {}, select=False))
                    except OSError as e:
                        self.failed.append("%s (%s)" % (os.path.basename(p), e))
                self.refresh_tables()
                self.nb.select(self.tree_lib.master)
                if ids:
                    self.tree_lib.selection_set([str(i) for i in ids if self.tree_lib.exists(str(i))])
                self.set_status("Added %d songs to the library. Double-click one to edit it." % len(ids))

        def _save_new(self, rec, changes, select=True):
            rec = dict(rec)
            changes = dict(changes)
            if changes.get("bpm") is not None and fmt_bpm(changes["bpm"]) == fmt_bpm(rec.get("bpm")):
                changes.pop("bpm")
            if changes.get("camelot") == rec.get("camelot"):
                changes.pop("camelot")
            rec.update(apply_key_bpm_edits(changes))
            rec.pop("status", None)
            sid = self.db.insert(rec)
            self.session.pop(rec["path"], None)
            if select:
                self.refresh_tables()
                self.nb.select(self.tree_lib.master)
                self.select(("lib", sid))
                self.set_status("Added to library: %s" % (rec.get("title") or rec["filename"]))
            return sid

        def main_button(self):
            if not self.current:
                return
            kind, ref = self.current
            if kind == "new":
                self.add_paths([ref])
            else:
                self.edit_ids([ref])

        # ---------------------------------------------------------- editing
        def edit_selected(self):
            which = self.active_tab()
            if which == "new":
                self.add_paths(self.selected_paths())
            else:
                ids = self.selected_ids()
                if not ids and self.current and self.current[0] == "lib":
                    ids = [self.current[1]]
                self.edit_ids(ids)

        def edit_ids(self, ids):
            recs = [self.db.get(i) for i in ids]
            recs = [r for r in recs if r]
            if not recs:
                self.set_status("Select one or more songs in the Library to edit")
                return
            mode = "edit" if len(recs) == 1 else "multi"
            EditDialog(self, mode, recs, lambda ch, ids=[r["id"] for r in recs]: self._save_edit(ids, ch))

        def _save_edit(self, ids, changes):
            info = changes.pop("_file_info", None)
            if info and len(ids) == 1:
                self.db.update_file_info(ids[0], info)
            if not changes and not info:
                self.set_status("Nothing changed")
                return
            if changes:
                self.db.update_fields(ids, apply_key_bpm_edits(changes))
            self.refresh_lib()
            if len(ids) == 1:
                self.select(("lib", ids[0]))
            elif self.current and self.current[0] == "lib":
                self.show(self.current)
            self.set_status("Saved changes to %d song(s)" % len(ids))

        def on_double(self, which, event):
            tree = self.tree_new if which == "new" else self.tree_lib
            if tree.identify_region(event.x, event.y) != "cell":
                return
            self.edit_selected()

        # ---------------------------------------------------------- other actions
        def open_files(self):
            fs = filedialog.askopenfilenames(title="Choose audio files")
            if fs:
                self.enqueue(list(fs))

        def open_folder(self):
            d = filedialog.askdirectory(title="Choose a music folder")
            if d:
                self.enqueue([d])

        def active_tab(self):
            return "new" if self.nb.select() == str(self.tree_new.master) else "lib"

        def selected_ids(self):
            return [int(i) for i in self.tree_lib.selection()]

        def selected_paths(self):
            return list(self.tree_new.selection())

        def reanalyse(self):
            if self.active_tab() == "new":
                paths = self.selected_paths()
                if paths:
                    self.enqueue(paths, force=True)
                return
            todo = []
            for i in self.selected_ids():
                r = self.db.get(i)
                if r and os.path.exists(r["path"]):
                    todo.append((r["path"], i))
                elif r:
                    self.failed.append("%s (file not found)" % r["filename"])
            for p, i in todo:
                self.worker.jobs.put((p, i))
            self.pending += len(todo)
            if todo:
                self.prog.start(12)
                self.set_status("Re-analysing %d song(s) - your edited names/genre are kept" % len(todo))
            elif self.failed:
                self.set_status("Can't re-analyse: " + "; ".join(self.failed))
                self.failed = []

        def remove(self):
            if self.active_tab() == "new":
                for p in self.selected_paths():
                    self.session.pop(p, None)
                self.refresh_new()
                self.draw_empty()
                return
            ids = self.selected_ids()
            if ids and messagebox.askyesno(APP_NAME, "Remove %d song(s) from the library?\n"
                                           "(The audio files are not touched.)" % len(ids)):
                self.db.delete(ids)
                self.refresh_lib()
                self.current = None
                self.draw_empty()

        def export(self):
            out = filedialog.asksaveasfilename(defaultextension=".csv", initialfile="songs.csv")
            if out:
                self.db.export_csv(out)
                self.set_status("Exported to " + out)

        # ---------------------------------------------------------- tables
        def refresh_tables(self):
            self.refresh_new()
            self.refresh_lib()

        def _filter(self, rows, fields):
            s = self.search.get().strip().lower()
            if s:
                rows = [r for r in rows if s in " ".join(str(r.get(f) or "") for f in fields).lower()]
            if self.compat_only.get() and self.current:
                cur = self.current_record()
                if cur and cur.get("camelot"):
                    comp = set(camelot_compatible(cur["camelot"]))
                    b = cur.get("bpm") or 0

                    def bpm_ok(x):
                        return bool(b and x) and any(abs(x * f - b) / b <= 0.06 for f in (0.5, 1, 2))
                    rows = [r for r in rows if r.get("camelot") in comp and bpm_ok(r.get("bpm"))]
            return rows

        def _sorted(self, which, rows):
            col, rev = self.sort[which]
            if not col:
                return rows
            keyf = {
                "bpm": lambda r: r.get("bpm") or 0,
                "camelot": lambda r: camelot_sort_key(r.get("camelot")),
                "duration": lambda r: r.get("duration") or 0,
                "size": lambda r: r.get("size") or 0,
                "key": lambda r: r.get("key_name") or "",
                "mix": lambda r: (r.get("mix_name") or "").lower(),
                "format": lambda r: fmt_format(r),
                "file": lambda r: (r.get("filename") or "").lower(),
            }.get(col, lambda r: str(r.get(col) or "").lower())
            return sorted(rows, key=keyf, reverse=rev)

        def refresh_new(self):
            t = self.tree_new
            sel = t.selection()
            t.delete(*t.get_children())
            rows = [dict(v, path=k) for k, v in self.session.items()]
            rows = self._sorted("new", self._filter(rows, ("title", "artist", "camelot", "key_name",
                                                           "filename")))
            for r in rows:
                tag = "c_" + (r.get("camelot") or "x")
                t.insert("", "end", iid=r["path"], tags=(tag,), values=(
                    r.get("title", ""), r.get("artist", ""), fmt_bpm(r.get("bpm")),
                    r.get("key_name", ""), r.get("camelot", ""), fmt_time(r.get("duration")),
                    r.get("status", ""), os.path.basename(r["path"])))
                t.tag_configure(tag, foreground=camelot_color(r.get("camelot")))
            keep = [i for i in sel if t.exists(i)]
            if keep:
                t.selection_set(keep)
            self.nb.tab(t.master, text="Just analysed (%d)" % len(self.session))

        def refresh_lib(self):
            t = self.tree_lib
            sel = t.selection()
            t.delete(*t.get_children())
            allrows = self.db.all()
            rows = self._sorted("lib", self._filter(allrows, (
                "artist", "title", "mix_name", "remixer", "featuring", "genre", "album", "label",
                "camelot", "key_name", "filename", "notes")))
            for r in rows:
                tag = "c_" + (r.get("camelot") or "x")
                title = r.get("title") or ""
                if r.get("featuring"):
                    title += " (feat. %s)" % r["featuring"]
                t.insert("", "end", iid=str(r["id"]), tags=(tag,), values=(
                    r.get("artist") or "", title, r.get("mix_name") or "", r.get("genre") or "",
                    fmt_bpm(r.get("bpm")), r.get("key_name") or "", r.get("camelot") or "",
                    fmt_time(r.get("duration")), fmt_format(r), fmt_size(r.get("size"))))
                t.tag_configure(tag, foreground=camelot_color(r.get("camelot")))
            keep = [i for i in sel if t.exists(i)]
            if keep:
                t.selection_set(keep)
            self.nb.tab(t.master, text="Library (%d)" % len(allrows))

        def sort_by(self, which, col):
            c, rev = self.sort[which]
            self.sort[which] = (col, (not rev) if c == col else False)
            self.refresh_new() if which == "new" else self.refresh_lib()

        def select(self, ref, show=True):
            kind, key = ref
            tree = self.tree_new if kind == "new" else self.tree_lib
            iid = str(key)
            if not tree.exists(iid):
                self.search.set("")
                self.compat_only.set(False)
                self.refresh_tables()
            self.nb.select(tree.master)
            if tree.exists(iid):
                tree.selection_set(iid)
                tree.see(iid)
            if show:
                self.show(ref)

        def on_select(self, which):
            tree = self.tree_new if which == "new" else self.tree_lib
            sel = tree.selection()
            if len(sel) != 1:
                return
            ref = ("new", sel[0]) if which == "new" else ("lib", int(sel[0]))
            if ref != self.current or not (self.view or {}).get("bpm"):
                self.show(ref)                 # also refreshes a song still showing "working on it"

        def on_tab(self, _e=None):
            which = self.active_tab()
            tree = self.tree_new if which == "new" else self.tree_lib
            if len(tree.selection()) == 1:
                self.on_select(which)

        # ---------------------------------------------------------- display
        def set_status(self, msg):
            self.status.configure(text=msg)

        def current_record(self):
            if not self.current:
                return None
            kind, key = self.current
            if kind == "new":
                return self.session.get(key)
            return self.db.get(key)

        def show(self, ref):
            kind, key = ref
            if kind == "new":
                rec = self.session.get(key)
                if not rec:
                    return
            else:
                row = self.db.get(key)
                if not row:
                    return
                rec = row_to_display(row)
            self.current = ref
            done = "bpm" in rec
            path = rec.get("path") or (key if kind == "new" else None)
            if self.player.path and self.player.path != path:
                self.player.stop()
                self.player.path = None
            self.view = {"ref": ref, "path": path, "duration": rec.get("duration"),
                         "cues": parse_cues(rec.get("cues")), "bpm": rec.get("bpm"),
                         "beat_offset": rec.get("beat_offset")}
            self.show_header(rec, partial=not done, in_library=(kind == "lib"))
            peaks = rec.get("peaks")
            if peaks is None and kind == "lib" and path and os.path.exists(path):
                self.set_status("Building the waveform for this song (only the first time)...")
                self.root.config(cursor="watch")
                self.root.update_idletasks()
                try:
                    peaks = compute_peaks(path, rec.get("channels") or 2)
                    if peaks is not None:
                        self.db.update_peaks(key, peaks)
                finally:
                    self.root.config(cursor="")
                self.set_status("Waveform ready")
            self._wave_args = (path, rec.get("duration"), peaks, rec.get("spec"))
            self._key_text = ("%s (%s)" % (rec.get("key_name"), rec.get("camelot"))) if done and rec.get("key_name") else ""
            self.wave.load(path, rec.get("duration"), peaks, keep_view=True, spec=rec.get("spec"))
            if self.editor is not None:
                self.editor.refresh()
            if rec.get("spec") is not None:
                self._spec_args = (rec["spec"], rec["avg"], rec.get("duration"),
                                   rec.get("chroma") if done else None, rec.get("key_pc"),
                                   bool(rec.get("key_minor")))
                self._spec_dirty = False
                self.draw(*self._spec_args, cues=self.view["cues"] if done else None)
            else:
                self._spec_args = None
                self.draw_empty("No spectrum stored for this song - press Re-analyse", wave=False)
            if self.compat_only.get():
                self.refresh_tables()

        def show_header(self, r, partial=False, in_library=False):
            title = r.get("title") or r.get("filename") or ""
            if r.get("featuring"):
                title += " (feat. %s)" % r["featuring"]
            if r.get("artist"):
                title = "%s  -  %s" % (r["artist"], title)
            self.lbl_title.configure(text=title)
            self._title_tip.text = title
            sub = []
            if r.get("mix_name"):
                s = r["mix_name"]
                if r.get("is_remix") and r.get("remixer"):
                    s += "  (remix by %s)" % r["remixer"]
                elif r.get("is_remix"):
                    s += "  (remix)"
                sub.append(s)
            if r.get("genre"):
                sub.append(r["genre"])
            if in_library:
                f = describe_file(r)
                if f:
                    sub.append(f)
            else:
                sub.append("not in your library yet")
            self.lbl_sub.configure(text="   |   ".join(sub))
            if partial:
                self.lbl_bpm.configure(text="...")
                self.badge.configure(text="...", bg="#555555")
                self.lbl_bpm_txt.configure(text="Tempo: working on it...")
                self.lbl_key_txt.configure(text="Key: working on it...")
                self.lbl_mix.configure(text="")
                self.lbl_cues.configure(text="Mix points: working on it...")
                self.btn_main.configure(text="Add to library", image=self.icons.get("add", ""))
                self.btn_main.state(["disabled"])
                self.btn_tap.state(["disabled"])
                return
            self.lbl_bpm.configure(text=fmt_bpm(r.get("bpm")) or "--")
            self.badge.configure(text=r.get("camelot") or "--", bg=camelot_color(r.get("camelot")))
            self.lbl_bpm_txt.configure(text="Tempo:  " + describe_bpm(r.get("bpm"), r.get("bpm_conf")) +
                                       ("     - measured by Essentia" if r.get("engine") == "essentia" else ""))
            self.lbl_key_txt.configure(text="Key:  " + describe_key(r.get("key_name"), r.get("camelot"),
                                                                   r.get("key_conf"), r.get("key_alt") or ""))
            comp = camelot_compatible(r.get("camelot"))
            self.lbl_mix.configure(text=("Mixes well with keys:  %s   (same, one step down/up, "
                                         "major/minor swap)" % "  ".join(comp)) if comp else "")
            self.lbl_cues.configure(text=describe_cues(r.get("cues")))
            self.btn_main.configure(text="Edit details..." if in_library else "Add to library",
                                    image=self.icons.get("edit" if in_library else "add", ""))
            self.btn_main.state(["!disabled"])
            self.btn_tap.state(["!disabled"])

        # ---------------------------------------------------------- views
        def waves(self):
            return [self.wave] + ([self.editor.wave] if self.editor is not None else [])

        def on_engine(self, _e=None):
            want = "essentia" if self.engine_var.get() == "Essentia" else "builtin"
            if want == self.engine:
                return
            if want == "builtin":
                self.set_engine("builtin")
                return
            if self._essentia_ok:
                self.set_engine("essentia")
                return
            if self._essentia_ok is False:
                self._essentia_ok = None          # try again: maybe it was installed meanwhile
            self.set_status("Checking whether Essentia works on this computer (can take a little while)...")
            self.root.config(cursor="watch")

            def probe():
                res = essentia_available()
                self.q.put(("essentia", res))
            threading.Thread(target=probe, daemon=True).start()
            self.root.after(300, lambda: self.root.config(cursor=""))

        def set_engine(self, engine):
            self.engine = engine
            self.worker.engine = engine
            self.engine_var.set("Essentia" if engine == "essentia" else "Built-in")
            self.set_status("Engine: %s - new analyses and Re-analyse will use it" %
                            ("Essentia" if engine == "essentia" else "built-in"))

        def open_tap(self):
            if not self.view or not self.view.get("ref") or not self.view.get("path"):
                self.set_status("Pick a song first")
                return
            if self.tapwin is not None:
                self.tapwin.close()
            self.tapwin = TapWindow(self)

        def set_bpm(self, ref, bpm):
            """Save a BPM the user chose (tapped or typed)."""
            if ref[0] == "lib":
                self.db.update_fields([ref[1]], {"bpm": bpm, "bpm_conf": -1.0})
                self.refresh_lib()
            elif ref[1] in self.session:
                self.session[ref[1]]["bpm"] = bpm
                self.session[ref[1]]["bpm_conf"] = -1.0
                self.refresh_new()
            if self.view and self.view.get("ref") == ref:
                self.show(ref)
            self.set_status("Tempo set to %s BPM%s" % (fmt_bpm(bpm), "  - saved" if ref[0] == "lib" else
                                                        "  (kept when you add the song to the library)"))

        def open_editor(self):
            if not self.view or not self._wave_args:
                self.set_status("Pick a song first")
                return
            if self.editor is not None:
                self.editor.lift()
                self.editor.focus_set()
                return
            self.editor = EditorWindow(self)
            sel = self.wave.sel
            if sel is not None:
                self.editor.wave.sel = sel
                self.editor.wave.zoom_to(sel["t"], 8.0)

        # ---------------------------------------------------------- playback
        def view_path(self):
            return self.view.get("path") if self.view else None

        def play_pause(self):
            p = self.view_path()
            if not self.player_ok or not p:
                return
            if self.player.path == p and self.player.playing():
                self.player.pause()
            elif self.player.path == p and self.player.paused_at is not None:
                self.player.resume()
            else:
                self.play_from(0.0)

        def play_from(self, t):
            p = self.view_path()
            if not self.player_ok or not p:
                return
            if not os.path.exists(p):
                self.set_status("Can't play - file not found: " + p)
                return
            self.player.volume = self.vol.get()
            self.player.play(p, t)

        def stop_play(self):
            self.player.stop()

        def set_volume(self, _e=None):
            self.player.volume = self.vol.get()
            if self.player.playing():
                self.player.play(self.player.path, self.player.position() or 0.0)

        def jump_mark(self, direction):
            cues = visible_cues((self.view or {}).get("cues"))
            if not cues:
                self.set_status("This song has no mix marks yet")
                return
            pos = self.player.position() if self.player.path == self.view_path() else None
            if direction > 0:
                ref = -1.0 if pos is None else pos + 0.3
                c = next((c for c in cues if c["t"] > ref), None)
            else:
                ref = 1e9 if pos is None else pos - 1.5
                c = next((c for c in reversed(cues) if c["t"] < ref), cues[0])
            if c:
                self.play_from(c["t"])
                self.set_status("Playing from %s (%s)" % (c.get("label"), fmt_time(c["t"])))

        def tick(self):
            p = self.view_path()
            mine = p is not None and self.player.path == p
            pos = self.player.position() if mine else None
            playing = mine and self.player.playing()
            dur = (self.view or {}).get("duration") or 0
            self.btn_play.configure(text="Pause" if playing else "Play",
                                    image=self.icons.get("pause" if playing else "play", ""))
            paused = mine and self.player.paused_at is not None
            self.lbl_time.configure(text="%s / %s%s" % (fmt_time(pos or 0), fmt_time(dur),
                                                         "  paused" if paused else ""))
            if self.editor is not None:
                ed = self.editor
                ed.wave.follow(pos, playing)
                ed.wave.update_cursor(pos)
                ed.btn_play.configure(text="Pause" if playing else "Play",
                                      image=self.icons.get("pause" if playing else "play", ""))
                ed.lbl_time.configure(text="%s / %s%s" % (fmt_ms(pos or 0), fmt_time(dur),
                                                           "  paused" if paused else ""))
            if self.viewnb.index(self.viewnb.select()) == 0:
                self.wave.follow(pos, playing)
                self.wave.update_cursor(pos)
            else:
                key = None if pos is None else round(pos, 2)
                if key != self._last_cursor:
                    self._blit_cursor(pos)
            self.root.after(80, self.tick)

        def vu_tick(self):
            levels = None
            p = self.view_path()
            pk = self.wave.peaks
            if pk is not None and p and self.player.path == p and self.player.playing():
                pos = self.player.position()
                if pos is not None:
                    i = int(pos * PEAK_RATE)
                    a, b = max(0, i - 4), min(len(pk), i + 1)       # the last ~50 ms
                    if b > a:
                        rms = np.sqrt(((pk[a:b, :, 2].astype(np.float32) / 127.0) ** 2).mean(axis=0))
                        db = 20 * np.log10(np.maximum(rms * 1.4142, 1e-5))
                        levels = [float(db[0]), float(db[-1])]          # mono: both bars the same
            try:
                self.vu.update(levels)
            except tk.TclError:
                return
            self.root.after(40, self.vu_tick)

        def _on_draw(self, _e=None):
            if self.ax_spec is not None:
                self.bg = self.canvas.copy_from_bbox(self.ax_spec.bbox)
                self._last_cursor = "x"

        def _blit_cursor(self, pos):
            if self.ax_spec is None or self.cursor is None or self.bg is None:
                return
            try:
                self.canvas.restore_region(self.bg)
                if pos is not None:
                    self.cursor.set_xdata([pos, pos])
                    self.ax_spec.draw_artist(self.cursor)
                self.canvas.blit(self.ax_spec.bbox)
                self._last_cursor = None if pos is None else round(pos, 2)
            except Exception:
                pass

        def _space(self, e):
            if e.widget.winfo_class() in ("Entry", "TEntry", "Text", "TCombobox", "Spinbox"):
                return None
            self.play_pause()
            return "break"

        def on_close(self):
            self.player.stop()
            if self.editor is not None:
                self.editor.destroy()
                self.editor = None
            self.root.destroy()

        # ---------------------------------------------------------- mix marks
        def on_click(self, e):
            if self.ax_spec is None or e.inaxes is not self.ax_spec or e.xdata is None:
                return
            dur = (self.view or {}).get("duration") or 0
            t = float(min(max(e.xdata, 0.0), dur))
            if e.button == 1:
                self.play_from(t)
            elif e.button == 3:
                self.mark_menu(t)

        def snap(self, t):
            v = self.view or {}
            bpm, off = v.get("bpm"), v.get("beat_offset")
            if not bpm or off is None:
                return round(t, 3)
            beat = 60.0 / bpm
            return round(max(0.0, off + round((t - off) / beat) * beat), 3)

        def nearest_mark(self, t, span=None):
            cues = visible_cues((self.view or {}).get("cues"))
            if not cues:
                return None
            c = min(cues, key=lambda c: abs(c["t"] - t))
            dur = (self.view or {}).get("duration") or 1
            lim = span * 0.05 if span else max(8.0, dur * 0.04)
            return c if abs(c["t"] - t) <= lim else None

        def mark_menu(self, t, span=None, view=None):
            if not self.view or not self.view.get("ref"):
                return
            m = tk.Menu(self.root, tearoff=0, bg=PANEL, fg=FG, activebackground="#3d5a80",
                        activeforeground="white")
            st = self.snap(t) if self.wave.snap_var.get() else round(t, 3)
            m.add_command(label="Play from %s" % fmt_ms(t), command=lambda: self.play_from(t))
            m.add_separator()
            m.add_command(label="Add IN mark here (%s)" % fmt_ms(st), command=lambda: self.add_mark(st, "in"))
            m.add_command(label="Add OUT mark here (%s)" % fmt_ms(st), command=lambda: self.add_mark(st, "out"))
            m.add_command(label="Add CUE mark here (%s)" % fmt_ms(st), command=lambda: self.add_mark(st, "cue"))
            near = self.nearest_mark(t, span)
            if near:
                m.add_command(label="Move mark %s here (%s)" % (near.get("label"), fmt_ms(st)),
                              command=lambda: self.move_mark(near, st, snap=False))
                m.add_command(label="Remove mark %s (%s)" % (near.get("label"), fmt_ms(near["t"])),
                              command=lambda: self.remove_mark(near))
            m.add_separator()
            v = view or self.wave
            if v is self.wave:
                m.add_command(label="Zoom in here", command=lambda: (self.viewnb.select(0), v.zoom_to(t, 8.0)))
            else:
                m.add_command(label="Zoom in here", command=lambda: v.zoom_to(t, 8.0))
            m.add_command(label="Reset marks to the automatic ones", command=self.reset_marks)
            try:
                m.tk_popup(self.root.winfo_pointerx(), self.root.winfo_pointery())
            finally:
                m.grab_release()

        def _key_mark(self, kind, e):
            if e.widget.winfo_class() in ("Entry", "TEntry", "Text", "TCombobox", "Spinbox"):
                return None
            if not self.view or not self.view.get("ref"):
                return "break"
            pos = self.player.position() if self.player.path == self.view_path() else None
            if pos is None:
                self.set_status("Start playback first - I / O put a mark where the song is playing")
                return "break"
            self.add_mark(self.snap(pos) if self.wave.snap_var.get() else round(pos, 3), kind)
            return "break"

        def add_mark(self, t, kind):
            cues = (self.view or {}).get("cues")
            if cues is None or not self.view.get("ref"):
                return
            label = "CUE" if kind == "cue" else CUE_STYLE[kind][0] + "*"
            m = {"t": round(t, 3), "kind": kind, "label": label, "auto": False}
            cues.append(m)
            for wv in self.waves():
                wv.sel = m
            self.save_marks("Added %s at %s" % (label, fmt_ms(t)))

        def move_mark(self, m, t, snap=None):
            if snap is None:
                snap = self.wave.snap_var.get()
            dur = (self.view or {}).get("duration") or 0
            t = self.snap(t) if snap else t
            t = round(min(max(0.0, t), dur or t), 3)
            if m.get("auto"):          # first manual change: remember where the app had put it
                m["orig_t"] = m["t"]
                m["auto"] = False
                if not str(m.get("label", "")).endswith("*"):
                    m["label"] = (m.get("label") or CUE_STYLE[m["kind"]][0]) + "*"
            m["t"] = t
            for wv in self.waves():
                wv.sel = m
            self.save_marks("Moved %s to %s" % (m.get("label"), fmt_ms(t)))

        def remove_mark(self, mark):
            cues = (self.view or {}).get("cues") or []
            out = []
            for c in cues:
                if c is mark:
                    if c.get("auto"):
                        out.append(dict(c, hidden=True))       # kept so 'reset' can bring it back
                    elif "orig_t" in c:
                        out.append({"t": c["orig_t"], "kind": c["kind"],
                                    "label": str(c.get("label", "")).rstrip("*"), "auto": True, "hidden": True})
                    continue
                out.append(c)
            cues[:] = out
            for wv in self.waves():
                if wv.sel is mark:
                    wv.sel = None
            self.save_marks("Removed %s" % mark.get("label"))

        def reset_marks(self):
            cues = (self.view or {}).get("cues")
            if cues is None:
                return
            out = []
            for c in cues:
                if c.get("auto"):
                    out.append({k: v for k, v in c.items() if k != "hidden"})
                elif "orig_t" in c:
                    out.append({"t": c["orig_t"], "kind": c["kind"],
                                "label": str(c.get("label", "")).rstrip("*"), "auto": True})
            cues[:] = out
            for wv in self.waves():
                wv.sel = None
            self.save_marks("Marks reset to the automatic ones")

        def save_marks(self, msg):
            ref = self.view["ref"]
            cues = self.view["cues"]
            cues.sort(key=lambda c: c["t"])
            if ref[0] == "new":
                if ref[1] in self.session:
                    self.session[ref[1]]["cues"] = cues
            else:
                self.db.update_fields([ref[1]], {"cues": json.dumps(cues)})
            self.lbl_cues.configure(text=describe_cues(cues))
            for wv in self.waves():
                wv.redraw()
            self._spec_dirty = True
            if self.viewnb.index(self.viewnb.select()) == 1:
                self.redraw_spectrum()
            self.set_status(msg + ("  - saved" if ref[0] == "lib" else
                                   "  (kept when you add the song to the library)"))

        def redraw_spectrum(self):
            if self._spec_args:
                self.draw(*self._spec_args, cues=(self.view or {}).get("cues"))
            self._spec_dirty = False

        def on_view_tab(self, _e=None):
            if self.viewnb.index(self.viewnb.select()) == 1:
                if self._spec_dirty:
                    self.redraw_spectrum()
                else:
                    self.canvas.draw_idle()
            else:
                self.wave.canvas.draw_idle()

        def _style(self, ax):
            ax.set_facecolor(BG)
            for s in ax.spines.values():
                s.set_color("#44464d")
            ax.tick_params(colors=DIM, labelsize=8)
            ax.xaxis.label.set_color(DIM)
            ax.yaxis.label.set_color(DIM)
            ax.title.set_color(FG)

        def draw_empty(self, msg="Drop audio files here\n(any format ffmpeg can read)", wave=True):
            if wave and hasattr(self, "wave"):
                self._wave_args = None
                for wv in self.waves():
                    wv.clear(msg.replace("\n", "  "))
            self.ax_spec = self.cursor = self.bg = None
            self.fig.clear()
            ax = self.fig.add_subplot(111)
            self._style(ax)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.text(0.5, 0.5, msg, ha="center", va="center", color=DIM, fontsize=14,
                    transform=ax.transAxes)
            self.canvas.draw_idle()

        def draw(self, spec, avg, duration, chroma, key_pc, minor, cues=None):
            self.ax_spec = self.cursor = self.bg = None
            self.fig.clear()
            gs = self.fig.add_gridspec(2, 2, height_ratios=[2.6, 1], width_ratios=[1.6, 1],
                                       hspace=0.45, wspace=0.15, left=0.06, right=0.99,
                                       top=0.93, bottom=0.1)
            freqs = spectrum_freqs()
            rows = spec.shape[0]
            db = spec.astype(np.float32) / 255.0 * 90.0 - 90.0

            ax = self.fig.add_subplot(gs[0, :])
            self._style(ax)
            ax.imshow(db, aspect="auto", origin="lower", cmap=GREEN_CMAP, vmin=-80, vmax=0,
                      extent=[0, duration or 1, 0, rows], interpolation="nearest")
            ticks = [f for f in (50, 100, 200, 500, 1000, 2000, 5000, 10000)
                     if freqs[0] <= f <= freqs[-1]]
            labels = [("%dk" % (f // 1000)) if f >= 1000 else str(f) for f in ticks]
            ax.set_yticks([np.log(f / freqs[0]) / np.log(freqs[-1] / freqs[0]) * (rows - 1)
                           for f in ticks])
            ax.set_yticklabels(labels)
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: fmt_time(v)))
            ax.set_title("Spectrum over time  (brighter = louder;  bottom = bass, top = treble)", fontsize=10)
            ax.set_ylabel("Hz")
            ax.set_xlim(0, duration or 1)
            ax.set_ylim(0, rows)
            for i, c in enumerate(visible_cues(cues)):
                lab, col, ls, _m = CUE_STYLE.get(c["kind"], CUE_STYLE["cue"])
                ax.axvline(c["t"], color=col, ls=ls, lw=1.5, alpha=0.95)
                right = c["t"] > (duration or 1) * 0.93
                ax.text(c["t"], rows * (0.975 if i % 2 == 0 else 0.86), " %s " % (c.get("label") or lab),
                        color="#101010", fontsize=7.5, fontweight="bold", va="top",
                        ha="right" if right else "left",
                        bbox=dict(boxstyle="square,pad=0.15", fc=col, ec="none", alpha=0.92))
            self.cursor = ax.axvline(0, color="#ff5252", lw=1.8, animated=True)
            self.ax_spec = ax
            self._last_cursor = "x"

            ax2 = self.fig.add_subplot(gs[1, 0])
            self._style(ax2)
            if avg is not None and len(avg) == len(freqs):
                ax2.semilogx(freqs, avg, color=GREEN, lw=1.2)
                ax2.fill_between(freqs, avg, avg.min() - 5, color=GREEN, alpha=0.18)
                ax2.set_ylim(max(avg.min() - 5, -100), 3)
            ax2.set_xlim(freqs[0], freqs[-1])
            ax2.set_xticks(ticks)
            ax2.set_xticklabels(labels)
            ax2.minorticks_off()
            ax2.set_title("Average sound balance", fontsize=10)
            ax2.set_ylabel("dB")

            ax3 = self.fig.add_subplot(gs[1, 1])
            self._style(ax3)
            names = PC_NAMES_MIN if minor else PC_NAMES_MAJ
            if chroma:
                scale = [0, 2, 3, 5, 7, 8, 10] if minor else [0, 2, 4, 5, 7, 9, 11]
                in_key = {(key_pc + s) % 12 for s in scale}
                colors = [("#ffb74d" if i == key_pc else (ACCENT if i in in_key else "#555a63"))
                          for i in range(12)]
                ax3.bar(range(12), chroma, color=colors)
                ax3.set_title("Notes used  (orange = home note, blue = in key)", fontsize=10)
            else:
                ax3.text(0.5, 0.5, "finding key...", ha="center", va="center", color=DIM,
                         transform=ax3.transAxes)
                ax3.set_title("Notes used", fontsize=10)
            ax3.set_xticks(range(12))
            ax3.set_xticklabels(names)
            ax3.set_yticks([])
            ax3.set_xlim(-0.6, 11.6)
            self.canvas.draw_idle()

    root = tk.Tk()
    try:
        from PIL import ImageTk
        root._icon = ImageTk.PhotoImage(make_app_icon(64), master=root)
        root.iconphoto(True, root._icon)
    except Exception:
        pass
    try:
        root.app = App(root)
    except RuntimeError as e:
        root.withdraw()
        messagebox.showerror(APP_NAME, str(e))
        return
    root.mainloop()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def run_cli(args):
    db = SongDB(args.db) if args.add else None
    files = expand_paths(args.cli)
    if not files:
        print("no audio files found")
        return 1
    print("%-40s %7s %5s %4s %6s  %s" % ("file", "BPM", "key", "cam", "time", "seconds"))
    for f in files:
        try:
            r = analyse(f, args.engine, args.bpm_range)
        except Exception as e:
            print("%-40s ERROR: %s" % (os.path.basename(f)[:40], e))
            continue
        if db:
            r.update(read_metadata(f))
            db.insert(r)
        print("%-40s %7.2f %5s %4s %6s  %s   (bpm %s, key %s)%s" % (
            r["filename"][:40], r["bpm"], r["key_name"], r["camelot"], fmt_time(r["duration"]),
            r["secs"], confidence_word(r["bpm_conf"]), confidence_word(r["key_conf"]),
            "  -> added" if db else ""))
        if r.get("cues"):
            print("%-40s   marks: %s" % ("", cues_short(r["cues"])))
    return 0


def parse_range(s):
    try:
        a, b = [float(v) for v in s.split("-")]
        if not (20 <= a < b <= 400) or b < a * 2:
            raise ValueError
        return a, b
    except ValueError:
        raise argparse.ArgumentTypeError("use LOW-HIGH with HIGH >= 2*LOW, e.g. 75-180")


def main():
    here = APP_DIR
    ap = argparse.ArgumentParser(description="BPM + key detector with spectrum view and song library")
    ap.add_argument("--db", default=default_db_path(), help="SQLite database file")
    ap.add_argument("--engine", choices=("builtin", "essentia"), default="builtin")
    ap.add_argument("--bpm-range", type=parse_range, default=(75.0, 180.0),
                    help="tempo range to fold results into (default 75-180)")
    ap.add_argument("--cli", nargs="+", metavar="FILE", help="analyse files/folders without the GUI")
    ap.add_argument("--add", action="store_true", help="with --cli: also add them to the library")
    args = ap.parse_args()

    if args.engine == "essentia":
        ok, why = essentia_available()
        if not ok:
            print("Essentia not usable (%s) - falling back to built-in engine" % why, file=sys.stderr)
            args.engine = "builtin"

    if args.cli:
        try:
            sys.exit(run_cli(args))
        except RuntimeError as e:
            sys.exit("error: %s" % e)
    run_gui(args)


if __name__ == "__main__":
    main()
