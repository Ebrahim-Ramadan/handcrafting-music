"""
╔══════════════════════════════════════════════════════════════════════╗
║          AIR VIRTUAL INSTRUMENT  –  v2  (Real Sound Engine)         ║
║                                                                      ║
║  MODES                                                               ║
║  G – Acoustic Guitar  (Karplus-Strong + body resonance + reverb)    ║
║  P – Piano            (inharmonic partials + hammer hardness)        ║
║  H – Harp             (pluck + soundboard + long shimmer tail)       ║
║  B – Bass Guitar      (deep KS + growl + cabinet sim)                ║
║  Q – Quit                                                            ║
║                                                                      ║
║  HAND CONTROLS                                                       ║
║  Spread fingers wide/closed  →  note on scale (low to high)         ║
║  Raise / lower hand          →  volume                               ║
║  Tilt wrist left / right     →  octave down / up                     ║
║  Hold same pose              →  note sustains & rings longer         ║
║  Small movement              →  smooth crossfade to next note        ║
║  Fast / big movement         →  hard pluck / strike attack           ║
║                                                                      ║
║  INSTALL                                                             ║
║  pip install mediapipe opencv-python numpy pygame scipy              ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import cv2
import numpy as np
import pygame
import pygame.sndarray
import math
import time
import os
import urllib.request
from collections import deque
from scipy import signal
from scipy.signal import fftconvolve

# ─────────────────────────────────────────────────────────────────────────────
#  MEDIAPIPE  – auto-detect old (< 0.10) vs new (>= 0.10) API
# ─────────────────────────────────────────────────────────────────────────────
import mediapipe as mp

_NEW_API = False
try:
    from mediapipe.tasks import python as _mp_python
    from mediapipe.tasks.python import vision as _mp_vision
    _NEW_API = True
except ImportError:
    pass

# ─────────────────────────────────────────────────────────────────────────────
#  AUDIO CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
SR           = 44100
CHUNK        = 512
MAX_CHANNELS = 24

# ─────────────────────────────────────────────────────────────────────────────
#  SCALES
# ─────────────────────────────────────────────────────────────────────────────
PENTATONIC = [48, 50, 52, 55, 57, 60, 62, 64, 67, 69, 72, 74]  # C2 pentatonic wide
PIANO_SCALE = [48, 50, 52, 53, 55, 57, 59, 60, 62, 64, 65, 67, 69, 71, 72]  # C maj 2 octaves
HARP_SCALE  = [48, 50, 52, 53, 55, 57, 59, 60, 62, 64, 65, 67, 69, 71, 72]  # same, wide
BASS_SCALE  = [28, 30, 31, 33, 35, 36, 38, 40, 41, 43]          # E1 pentatonic (low)

SCALES = {
    "guitar": PENTATONIC,
    "piano":  PIANO_SCALE,
    "harp":   HARP_SCALE,
    "bass":   BASS_SCALE,
}

PALETTE = {
    "guitar": (40, 200, 80),
    "piano":  (160, 100, 255),
    "harp":   (255, 190, 50),
    "bass":   (50,  160, 255),
}

def midi_to_freq(midi: int) -> float:
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


# ─────────────────────────────────────────────────────────────────────────────
#  SHARED UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
def _env(wave: np.ndarray, atk_s: float, dec_s: float,
         sus_lvl: float, rel_s: float) -> np.ndarray:
    """Full ADSR envelope."""
    n   = len(wave)
    env = np.ones(n)
    a   = min(int(atk_s * SR), n)
    d   = min(int(dec_s * SR), n - a)
    r   = min(int(rel_s * SR), n)

    if a > 0: env[:a]       = np.linspace(0, 1, a)
    if d > 0: env[a:a+d]    = np.linspace(1, sus_lvl, d)
    env[a+d: n-r]           = sus_lvl
    if r > 0: env[n-r:]     = np.linspace(sus_lvl, 0, r)
    return wave * env


def _normalise(wave: np.ndarray, target: float = 0.92) -> np.ndarray:
    mx = np.max(np.abs(wave))
    return wave * (target / mx) if mx > 1e-9 else wave


def _to_int16(wave: np.ndarray, vol: float) -> np.ndarray:
    return np.clip(wave * vol * 32767, -32767, 32767).astype(np.int16)


# ─────────────────────────────────────────────────────────────────────────────
#  REVERB IMPULSE RESPONSE  (Schroeder + early reflections)
#  Built once, reused for all instruments
# ─────────────────────────────────────────────────────────────────────────────
def _build_reverb_ir(room: float = 0.55, damp: float = 0.45,
                     predelay_ms: float = 18.0) -> np.ndarray:
    """Generate a ~0.9 s room reverb IR."""
    ir_len = int(SR * 0.9)
    ir     = np.zeros(ir_len)

    # Early reflections (sparse taps)
    early_ms = [7, 13, 19, 27, 34, 43, 55, 68]
    for t_ms in early_ms:
        idx = int(t_ms * SR / 1000)
        if idx < ir_len:
            ir[idx] += 0.35 * (0.8 ** (early_ms.index(t_ms)))

    # Schroeder comb filters
    comb_samples = [int(x * room) for x in [1831, 1949, 2053, 2251, 2399, 2521]]
    g_vals       = [0.85 - damp * 0.1 * i for i in range(len(comb_samples))]
    for cs, g in zip(comb_samples, g_vals):
        buf   = np.zeros(cs)
        tail  = np.zeros(ir_len)
        state = 0.0
        for i in range(ir_len):
            inp    = ir[i] + g * buf[i % cs]
            out    = buf[i % cs]
            state  = out * (1 - damp * 0.3) + state * damp * 0.3
            buf[i % cs] = inp
            tail[i]     = state
        ir = ir + tail * 0.18

    # Pre-delay
    pd = int(predelay_ms * SR / 1000)
    ir = np.concatenate([np.zeros(pd), ir])[:ir_len]

    # Taper tail
    fade = np.linspace(1, 0, ir_len) ** 1.5
    ir  *= fade
    return _normalise(ir, 0.85)

print("Building reverb IR …", end=" ", flush=True)
_REVERB_IR = _build_reverb_ir()
print("done.")

def _add_reverb(dry: np.ndarray, wet_mix: float = 0.22) -> np.ndarray:
    if wet_mix <= 0:
        return dry
    rev = fftconvolve(dry, _REVERB_IR, mode="full")[:len(dry)]
    return dry * (1 - wet_mix) + rev * wet_mix


# ─────────────────────────────────────────────────────────────────────────────
#  GUITAR  –  Karplus-Strong with stiffness, pick position & body resonance
# ─────────────────────────────────────────────────────────────────────────────
def synth_guitar(freq: float, dur: float, vol: float,
                 brightness: float = 0.55) -> np.ndarray:
    """
    Extended Karplus-Strong:
    - Shaped noise burst (pick noise)
    - Pick-position comb filter (cancel harmonics at pick point)
    - Decay-rate stiffness term
    - Body resonance (low-mid EQ peaks like a spruce top)
    - Fret buzz (subtle high-freq modulation on attack)
    """
    buf_len = max(2, int(SR / freq))
    rng     = np.random.default_rng(seed=42)

    # Shaped initial burst: band-limited noise
    burst = rng.uniform(-1, 1, buf_len).astype(np.float64)
    burst *= np.hanning(buf_len)

    # Pick-position simulation: comb filter to thin out harmonics
    pick_pos   = 0.12   # 12% from nut  →  suppresses every ~8th harmonic
    pick_delay = max(1, int(buf_len * pick_pos))
    for k in range(pick_delay, buf_len):
        burst[k] -= 0.5 * burst[k - pick_delay]

    total = int(SR * dur)
    out   = np.zeros(total)
    buf   = burst.copy()

    # Loss + stiffness coefficients
    loss     = 0.9993 - freq * 2e-6   # higher notes decay faster
    blend    = 0.5 - brightness * 0.08

    for i in range(total):
        idx      = i % buf_len
        nxt      = (idx + 1) % buf_len
        prv      = (idx - 1) % buf_len
        out[i]   = buf[idx]
        # KS averaging filter + stiffness (2nd-order term)
        avg      = blend * buf[prv] + (1 - 2*blend) * buf[idx] + blend * buf[nxt]
        buf[nxt] = avg * loss

    # Body resonance: acoustic guitar peaks
    for f0, Q in [(82, 12), (196, 8), (392, 6), (800, 4), (2500, 3)]:
        if f0 < SR / 2:
            b, a  = signal.iirpeak(f0 / (SR / 2), Q)
            out   = signal.lfilter(b, a, out)

    # Fret buzz: subtle high-freq noise on very early attack
    buzz_len = min(int(0.012 * SR), total)
    buzz     = rng.uniform(-0.04, 0.04, buzz_len) * np.linspace(1, 0, buzz_len)
    out[:buzz_len] += buzz

    out = _normalise(out)
    out = _env(out, atk_s=0.001, dec_s=0.08, sus_lvl=0.7, rel_s=0.35)
    out = _add_reverb(out, wet_mix=0.18)
    return _to_int16(out, vol)


# ─────────────────────────────────────────────────────────────────────────────
#  PIANO  –  Inharmonic partials + hammer hardness + sympathetic resonance
# ─────────────────────────────────────────────────────────────────────────────
def synth_piano(freq: float, dur: float, vol: float,
                hardness: float = 0.72) -> np.ndarray:
    """
    Physical model:
    - Inharmonic partials (B coefficient scales with register)
    - Hammer spectral weighting (harder = brighter attack)
    - Bi-linear decay: each partial decays independently
    - Duplex scale sympathetic shimmer
    - Soft pedal warmth (low-pass on attack)
    """
    total = int(SR * dur)
    t     = np.linspace(0, dur, total, endpoint=False)
    wave  = np.zeros(total)

    # Inharmonicity coefficient: increases with pitch register
    B = 0.00012 * (freq / 440.0) ** 1.8

    for n in range(1, 20):
        # Inharmonic partial frequency
        fn = freq * n * np.sqrt(1 + B * n * n)
        if fn >= SR / 2:
            break

        # Hammer spectral envelope
        amp = np.exp(-hardness * (n - 1) * 0.28) / n

        # Bi-linear decay (two components: initial attack transient + sustain)
        t1 = 0.5 + n * 0.18 + freq / 1800   # fast component
        t2 = 0.08 + freq / 8000              # slow sustain component
        decay = 0.6 * np.exp(-t * t1) + 0.4 * np.exp(-t * t2)

        # Slight detuning between two string unisons (chorus effect)
        detune = 1.0 + 0.0004 * (n % 3 - 1)
        phase  = np.random.uniform(0, 2 * np.pi)
        wave  += amp * decay * (
            np.sin(2 * np.pi * fn * t + phase) +
            0.45 * np.sin(2 * np.pi * fn * detune * t + phase + 0.3)
        )

    # Duplex scale shimmer (high partial sympathetic ring)
    shimmer_freq = freq * 12.5
    if shimmer_freq < SR / 2:
        shimmer_decay = np.exp(-t * (0.3 + freq / 3000))
        wave += 0.04 * shimmer_decay * np.sin(2 * np.pi * shimmer_freq * t)

    # Attack click / hammer noise (brief broadband transient)
    click_len = min(int(0.006 * SR), total)
    click     = np.random.randn(click_len) * np.linspace(0.08, 0, click_len)
    wave[:click_len] += click

    wave = _normalise(wave)
    wave = _env(wave, atk_s=0.003, dec_s=0.12, sus_lvl=0.65, rel_s=0.5)
    wave = _add_reverb(wave, wet_mix=0.25)
    return _to_int16(wave, vol)


# ─────────────────────────────────────────────────────────────────────────────
#  HARP  –  Plucked string with soundboard resonance + long shimmer
# ─────────────────────────────────────────────────────────────────────────────
def synth_harp(freq: float, dur: float, vol: float) -> np.ndarray:
    """
    Extended KS with:
    - Long sustain loss factor (very slow decay)
    - High-register sparkle via high-order harmonics
    - Soundboard low-mid resonance
    - Long spacious reverb tail
    """
    buf_len = max(2, int(SR / freq))
    rng     = np.random.default_rng(seed=7)
    burst   = rng.uniform(-1, 1, buf_len).astype(np.float64)
    burst  *= np.hanning(buf_len)

    total = int(SR * dur)
    out   = np.zeros(total)
    buf   = burst.copy()

    # Very slow loss for harp's long ring
    loss  = 0.9998 - freq * 1e-6

    for i in range(total):
        idx      = i % buf_len
        nxt      = (idx + 1) % buf_len
        prv      = (idx - 1) % buf_len
        out[i]   = buf[idx]
        avg      = 0.45 * buf[prv] + 0.1 * buf[idx] + 0.45 * buf[nxt]
        buf[nxt] = avg * loss

    # Add harmonic shimmer (pluck sparkle)
    t = np.linspace(0, dur, total, endpoint=False)
    for n, (amp, decay_rate) in enumerate(
        [(0.25, 4.0), (0.15, 8.0), (0.08, 14.0), (0.04, 22.0)], start=2
    ):
        fn = freq * n
        if fn < SR / 2:
            out += amp * np.exp(-t * decay_rate) * np.sin(2 * np.pi * fn * t)

    # Soundboard resonance (harp body)
    for f0, Q in [(130, 10), (260, 7), (520, 5)]:
        if f0 < SR / 2:
            b, a = signal.iirpeak(f0 / (SR / 2), Q)
            out  = signal.lfilter(b, a, out)

    out = _normalise(out)
    out = _env(out, atk_s=0.001, dec_s=0.05, sus_lvl=0.85, rel_s=0.6)
    out = _add_reverb(out, wet_mix=0.32)
    return _to_int16(out, vol)


# ─────────────────────────────────────────────────────────────────────────────
#  BASS GUITAR  –  Deep KS + growl harmonics + cabinet simulation
# ─────────────────────────────────────────────────────────────────────────────
def synth_bass(freq: float, dur: float, vol: float,
               growl: float = 0.4) -> np.ndarray:
    """
    Bass guitar model:
    - KS with slow decay & dark tone
    - Finger-pluck thump transient
    - Growl = harmonic distortion (soft-clip + odd harmonics)
    - Cabinet IR simulation (low-mid presence filter, HF rolloff)
    """
    buf_len = max(2, int(SR / freq))
    rng     = np.random.default_rng(seed=13)
    burst   = rng.uniform(-1, 1, buf_len).astype(np.float64)
    burst  *= np.hanning(buf_len)

    total = int(SR * dur)
    out   = np.zeros(total)
    buf   = burst.copy()
    loss  = 0.9991 - freq * 3e-6

    for i in range(total):
        idx      = i % buf_len
        nxt      = (idx + 1) % buf_len
        prv      = (idx - 1) % buf_len
        out[i]   = buf[idx]
        avg      = 0.48 * buf[prv] + 0.04 * buf[idx] + 0.48 * buf[nxt]
        buf[nxt] = avg * loss

    # Finger-pluck thump (low-pass noise burst)
    thump_len = min(int(0.025 * SR), total)
    thump     = rng.uniform(-1, 1, thump_len)
    # Low-pass to make it a "thud"
    b_lp, a_lp = signal.butter(2, 200 / (SR / 2), btype='low')
    thump      = signal.lfilter(b_lp, a_lp, thump)
    thump     *= np.linspace(1, 0, thump_len) ** 0.5
    out[:thump_len] += thump * 0.6

    # Growl: soft-clip distortion → odd harmonics (warm overdrive)
    if growl > 0:
        driven = out * (1 + growl * 2.5)
        clipped = np.tanh(driven * 1.8) / 1.8
        out = out * (1 - growl * 0.4) + clipped * growl * 0.4

    # Cabinet simulation: presence boost, HF rolloff
    b_pres, a_pres = signal.iirpeak(800 / (SR / 2), 3)
    b_hf,   a_hf   = signal.butter(2, 4000 / (SR / 2), btype='low')
    out = signal.lfilter(b_pres, a_pres, out)
    out = signal.lfilter(b_hf,   a_hf,   out)

    out = _normalise(out)
    out = _env(out, atk_s=0.002, dec_s=0.06, sus_lvl=0.75, rel_s=0.3)
    out = _add_reverb(out, wet_mix=0.10)
    return _to_int16(out, vol)


# ─────────────────────────────────────────────────────────────────────────────
#  SOUND CACHE & BUILDER
# ─────────────────────────────────────────────────────────────────────────────
_cache: dict = {}

# Sustain durations per mode (seconds)
_SUSTAIN = {"guitar": 2.2, "piano": 3.5, "harp": 4.0, "bass": 2.0}

def get_sound(freq: float, mode: str, vol: float = 0.7) -> pygame.mixer.Sound:
    key = (round(freq, 2), mode, round(vol, 2))
    if key in _cache:
        return _cache[key]

    dur = _SUSTAIN[mode]
    if   mode == "guitar": samples = synth_guitar(freq, dur, vol)
    elif mode == "piano":  samples = synth_piano (freq, dur, vol)
    elif mode == "harp":   samples = synth_harp  (freq, dur, vol)
    else:                  samples = synth_bass  (freq, dur, vol)

    stereo = np.column_stack([samples, samples])
    sound  = pygame.sndarray.make_sound(stereo)
    _cache[key] = sound
    return sound


# ─────────────────────────────────────────────────────────────────────────────
#  WARM-UP CACHE in background thread
# ─────────────────────────────────────────────────────────────────────────────
import threading

def _warm_cache(mode: str):
    """Pre-synthesise a few common notes so first notes are instant."""
    warm_notes = [48, 52, 55, 60, 64, 67, 72]
    for midi in warm_notes:
        freq = midi_to_freq(midi)
        get_sound(freq, mode, 0.7)
    print(f"  [{mode}] cache warmed up.")


# ─────────────────────────────────────────────────────────────────────────────
#  HAND FEATURE EXTRACTION  (unchanged from v1)
# ─────────────────────────────────────────────────────────────────────────────
FINGER_TIPS = [4, 8, 12, 16, 20]

def _lms(hand):
    return hand if isinstance(hand, (list, tuple)) else hand.landmark

def extract_features(hand, fw: int, fh: int) -> dict:
    lms = _lms(hand)
    pts = [(lm.x * fw, lm.y * fh, lm.z) for lm in lms]
    palm   = np.array(pts[0][:2])
    tips   = np.array([pts[t][:2] for t in FINGER_TIPS])
    spread = float(np.mean(np.linalg.norm(tips - palm, axis=1)))
    height = 1.0 - (pts[0][1] / fh)
    delta  = np.array(pts[9][:2]) - np.array(pts[0][:2])
    tilt   = math.atan2(-delta[1], delta[0])
    curls  = [pts[mcp][1] - pts[tip][1]
              for tip, mcp in zip([8,12,16,20], [5,9,13,17])]
    return dict(spread=spread, height=height, tilt=tilt,
                curl=float(np.mean(curls)), raw_pts=pts)

def features_to_note(feats: dict, scale: list) -> tuple:
    spread_norm  = float(np.clip((feats["spread"] - 25) / 130.0, 0, 1))
    idx          = int(spread_norm * (len(scale) - 1))
    octave       = +12 if feats["tilt"] > 0.38 else (-12 if feats["tilt"] < -0.38 else 0)
    midi         = np.clip(scale[idx] + octave, 21, 108)
    volume       = float(np.clip(0.35 + feats["height"] * 0.65, 0.1, 1.0))
    sustain_mult = float(np.clip(1.0 + feats["curl"] / 70.0, 0.5, 3.0))
    return int(midi), volume, sustain_mult

def pose_distance(a: dict, b: dict) -> float:
    ds = abs(a["spread"] - b["spread"]) / 130.0
    dh = abs(a["height"] - b["height"])
    dt = abs(a["tilt"]   - b["tilt"]) / math.pi
    return ds + dh + dt


# ─────────────────────────────────────────────────────────────────────────────
#  VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────
NOTE_NAMES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

def midi_name(midi: int) -> str:
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"

_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(0,17),(17,18),(18,19),(19,20),
]

def draw_hand(frame, hand, color, note_name: str = ""):
    h, w = frame.shape[:2]
    lms  = _lms(hand)
    # glow lines
    for a, b in _CONNECTIONS:
        ax, ay = int(lms[a].x*w), int(lms[a].y*h)
        bx, by = int(lms[b].x*w), int(lms[b].y*h)
        cv2.line(frame, (ax,ay), (bx,by), tuple(c//3 for c in color), 4, cv2.LINE_AA)
        cv2.line(frame, (ax,ay), (bx,by), color, 2, cv2.LINE_AA)
    # joints
    for i, lm in enumerate(lms):
        cx, cy = int(lm.x*w), int(lm.y*h)
        r = 9 if i in FINGER_TIPS else 4
        cv2.circle(frame, (cx,cy), r+2, (0,0,0), -1)
        cv2.circle(frame, (cx,cy), r,   color,   -1)
    # note label near wrist
    if note_name:
        wx = int(lms[0].x*w) - 22
        wy = int(lms[0].y*h) + 36
        cv2.putText(frame, note_name, (wx, wy),
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, (0,0,0), 5, cv2.LINE_AA)
        cv2.putText(frame, note_name, (wx, wy),
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, color,   1, cv2.LINE_AA)

def draw_ui(frame, mode, note, vol, sustain, change, fps, loading=False):
    h, w  = frame.shape[:2]
    color = PALETTE[mode]

    # Side panel
    ov = frame.copy()
    cv2.rectangle(ov, (w-220, 0), (w, 210), (12,12,12), -1)
    cv2.addWeighted(ov, 0.65, frame, 0.35, 0, frame)

    def txt(s, x, y, sc=0.55, c=None, bold=False):
        c    = c or color
        font = cv2.FONT_HERSHEY_DUPLEX if bold else cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, s, (x,y), font, sc, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(frame, s, (x,y), font, sc, c,       1, cv2.LINE_AA)

    mode_icons = {"guitar": "GUITAR", "piano": "PIANO", "harp": "HARP", "bass": "BASS"}
    txt(f"[ {mode_icons[mode]} ]", w-215, 30, 0.65, bold=True)
    txt(f"NOTE  {midi_name(note)}", w-215, 60)
    txt(f"VOL   {int(vol*100):3d}%",   w-215, 86)
    txt(f"HOLD  {sustain:.1f}x",       w-215, 112)
    txt(f"MOVE  {change:.2f}",         w-215, 138)
    txt(f"FPS   {fps:.0f}",            w-215, 164)

    if loading:
        txt("synthesising…", w-215, 190, 0.45, (200,200,80))

    # Bottom hint bar
    cv2.rectangle(frame, (0, h-32), (w, h), (12,12,12), -1)
    txt("[G]Guitar  [P]Piano  [H]Harp  [B]Bass  [Q]Quit",
        10, h-11, 0.44, (170,170,170))

    # Volume bar
    bx, by, bw = 10, h-58, 220
    cv2.rectangle(frame, (bx, by), (bx+bw, by+9), (40,40,40), -1)
    cv2.rectangle(frame, (bx, by), (bx+int(bw*vol), by+9), color, -1)
    txt("VOL", bx, by-5, 0.38)


# ─────────────────────────────────────────────────────────────────────────────
#  VOICE ENGINE  –  per-hand playback with crossfade logic
# ─────────────────────────────────────────────────────────────────────────────
class Voice:
    BIG_THRESH = 0.32

    def __init__(self):
        self.current_midi  = -1
        self.last_features = None
        self.last_played   = 0.0
        self.cooldown      = 0.10
        self._ch           = None

    def update(self, feats, scale, mode):
        now  = time.time()
        midi, vol, sm = features_to_note(feats, scale)
        freq   = midi_to_freq(midi)
        change = pose_distance(feats, self.last_features) if self.last_features else 1.0

        trigger = hard = False
        if   self.current_midi != midi:        trigger = True; hard = change > self.BIG_THRESH
        elif change > self.BIG_THRESH:          trigger = hard = True
        elif now - self.last_played > 0.85*sm:  trigger = True

        if trigger and (now - self.last_played > self.cooldown):
            # Non-blocking: check cache only, defer synthesis to background
            key = (round(freq, 2), mode, round(vol, 2))
            if key in _cache:
                sound = _cache[key]
                if self._ch and self._ch.get_busy():
                    self._ch.stop() if hard else self._ch.fadeout(200)
                ch = pygame.mixer.find_channel(True)
                if ch:
                    ch.play(sound)
                    self._ch = ch
                self.current_midi = midi
                self.last_played  = now
            else:
                # Synthesise in background, play on next frame
                def _synth_bg():
                    get_sound(freq, mode, vol)
                threading.Thread(target=_synth_bg, daemon=True).start()

        self.last_features = feats
        return midi, vol, sm, change


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL DOWNLOAD (new Tasks API)
# ─────────────────────────────────────────────────────────────────────────────
MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
              "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "hand_landmarker.task")

def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand landmark model (~7 MB) …", flush=True)
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Download complete.")
    return MODEL_PATH


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    pygame.mixer.pre_init(frequency=SR, size=-16, channels=2, buffer=CHUNK)
    pygame.init()
    pygame.mixer.set_num_channels(MAX_CHANNELS)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)

    # ── detector ────────────────────────────────────────────────────────────
    if _NEW_API:
        print("mediapipe >= 0.10  →  Tasks API")
        base = _mp_python.BaseOptions(model_asset_path=ensure_model())
        opts = _mp_vision.HandLandmarkerOptions(
            base_options=base, num_hands=2,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.7,
            min_tracking_confidence=0.6,
        )
        _det = _mp_vision.HandLandmarker.create_from_options(opts)

        def detect(bgr):
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            res = _det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            return res.hand_landmarks
    else:
        print("mediapipe < 0.10  →  legacy solutions API")
        _h = mp.solutions.hands.Hands(
            static_image_mode=False, max_num_hands=2,
            min_detection_confidence=0.7, min_tracking_confidence=0.6,
        )
        def detect(bgr):
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            res = _h.process(rgb)
            return res.multi_hand_landmarks or []

    # ── pre-warm guitar cache in background ─────────────────────────────────
    mode   = "guitar"
    voices = [Voice(), Voice()]
    fps_q  = deque(maxlen=20)
    t_prev = time.time()

    print("\nWarming up guitar sounds …")
    threading.Thread(target=_warm_cache, args=("guitar",), daemon=True).start()

    print("\n  Air Instrument v2 LIVE")
    print("  G = Guitar   P = Piano   H = Harp   B = Bass   Q = Quit\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]
        hands = detect(frame)

        t_now = time.time()
        fps_q.append(1.0 / max(t_now - t_prev, 1e-6))
        t_prev = t_now
        fps = float(np.mean(fps_q))

        disp_midi, disp_vol, disp_sus, disp_chg = 60, 0.0, 1.0, 0.0
        color = PALETTE[mode]

        if hands:
            for i, hand in enumerate(hands[:2]):
                feats = extract_features(hand, w, h)
                midi, vol, sus, chg = voices[i].update(feats, SCALES[mode], mode)
                draw_hand(frame, hand, color, midi_name(midi) if i == 0 else "")
                if i == 0:
                    disp_midi, disp_vol, disp_sus, disp_chg = midi, vol, sus, chg
        else:
            for v in voices:
                if v._ch and v._ch.get_busy():
                    v._ch.fadeout(500)
                v.last_features = None

        # Check if any background synthesis is pending
        loading = any(
            (round(midi_to_freq(features_to_note(v.last_features or
             {"spread":80,"height":0.5,"tilt":0,"curl":0}, SCALES[mode])[0]),2),
             mode, 0.7) not in _cache
            for v in voices if v.last_features
        ) if hands else False

        draw_ui(frame, mode, disp_midi, disp_vol, disp_sus, disp_chg, fps, loading)
        cv2.imshow("Air Instrument v2", frame)

        key = cv2.waitKey(1) & 0xFF
        if   key == ord('q'): break
        elif key == ord('g'):
            mode = "guitar"; _cache.clear()
            threading.Thread(target=_warm_cache, args=("guitar",), daemon=True).start()
            print("-> GUITAR")
        elif key == ord('p'):
            mode = "piano";  _cache.clear()
            threading.Thread(target=_warm_cache, args=("piano",),  daemon=True).start()
            print("-> PIANO")
        elif key == ord('h'):
            mode = "harp";   _cache.clear()
            threading.Thread(target=_warm_cache, args=("harp",),   daemon=True).start()
            print("-> HARP")
        elif key == ord('b'):
            mode = "bass";   _cache.clear()
            threading.Thread(target=_warm_cache, args=("bass",),   daemon=True).start()
            print("-> BASS")

    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()
    print("Bye!")


if __name__ == "__main__":
    main()