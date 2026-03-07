"""
Air Virtual Instrument  v7
──────────────────────────────────────────────────────────────────────
INSTALL:  pip install mediapipe opencv-python numpy pygame scipy
RUN:      python main.py

CONTROLS  (cycle with number keys or letter shortcuts)
  G  Guitar    P  Piano    H  Harp     B  Bass
  V  Violin    F  Flute    S  Sitar    T  Theremin
  O  Organ     M  Marimba
  Q  Quit

HAND → SOUND
  Finger curl               →  pitch  (smooth glide)
  Ring + Pinky curl         →  warmth / body
  Finger tip openness       →  brightness / harmonics
  Index-Middle spread       →  vibrato depth
  Ring-Pinky spread         →  tremolo depth
  Wrist tilt L / R          →  octave bend
  Wrist forward bend        →  reverb mix
  Hand height               →  volume
  Hand MOVEMENT             →  sustain (still = fade out)
  Distance between hands    →  vibrato + sustain length
──────────────────────────────────────────────────────────────────────
"""

import cv2, numpy as np, pygame, pygame.sndarray
import math, time, os, threading, urllib.request
from collections import deque
from scipy import signal

# ── mediapipe ─────────────────────────────────────────────────────
import mediapipe as mp
_NEW_API = False
try:
    from mediapipe.tasks import python as _mpt
    from mediapipe.tasks.python import vision as _mpv
    _NEW_API = True
except ImportError:
    pass

# ── constants ─────────────────────────────────────────────────────
SR       = 44100
CHUNK_S  = 0.15
CHUNK    = int(SR * CHUNK_S)

MOVE_THRESH   = 3.5    # px/frame to count as "moving"
STILL_FADEOUT = 0.7    # seconds of stillness before fade begins
STILL_FULL    = 1.5    # seconds until fully silent

NOTE_NAMES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
def midi_name(m): return f"{NOTE_NAMES[int(m)%12]}{int(m)//12-1}"
def m2f(m):       return 440.0 * 2.0**((m-69)/12.0)

# Scales in a friendlier lower range (max ~E5)
PENTATONIC  = [36,38,40,43,45,48,50,52,55,57,60,62,64,67,69]
MAJOR       = [36,38,40,41,43,45,47,48,50,52,53,55,57,59,60,62]
HARP_SC     = [36,38,40,41,43,45,47,48,50,52,53,55,57,59,60]
BASS_SC     = [24,26,28,29,31,33,35,36,38,40,41,43]
VIOLIN_SC   = [55,57,59,60,62,64,65,67,69,71,72,74,76]   # G3-E5, violin range
FLUTE_SC    = [60,62,64,65,67,69,71,72,74,76,77,79,81]   # C4-A5, bright high
SITAR_SC    = [48,49,51,53,55,56,58,60,61,63,65,67,68]   # Bhairav raga
THEREMIN_SC = [36,37,38,39,40,41,42,43,44,45,46,47,48,
               49,50,51,52,53,54,55,56,57,58,59,60]       # full chromatic
ORGAN_SC    = [36,38,40,41,43,45,47,48,50,52,53,55,57,59,60,62]  # major
MARIMBA_SC  = [48,50,52,53,55,57,59,60,62,64,65,67,69]   # C4-A5 major

SCALES  = {"guitar":PENTATONIC,"piano":MAJOR,"harp":HARP_SC,"bass":BASS_SC,
           "violin":VIOLIN_SC,"flute":FLUTE_SC,"sitar":SITAR_SC,
           "theremin":THEREMIN_SC,"organ":ORGAN_SC,"marimba":MARIMBA_SC}
PALETTE = {"guitar":(40,200,80),"piano":(160,100,255),
           "harp":(255,190,50),"bass":(50,160,255),
           "violin":(220,80,80),"flute":(80,220,220),
           "sitar":(255,140,0),"theremin":(180,0,255),
           "organ":(255,60,180),"marimba":(60,200,140)}

# ── shared state ──────────────────────────────────────────────────
class State:
    def __init__(self):
        self.freq      = 130.81
        self.vol       = 0.0
        self.bright    = 0.5
        self.warm      = 0.3
        self.vib       = 0.0
        self.trem      = 0.0
        self.rev       = 0.18
        self.active    = False
        self.move_gain = 0.0   # 0-1 from vision thread
        self.pluck     = False # one-shot trigger
        self.hand_dist = 0.5   # normalised 0-1
        self._lock     = threading.Lock()

    def write(self, **kw):
        with self._lock:
            for k,v in kw.items(): setattr(self,k,v)

    def read(self):
        with self._lock:
            return (self.freq, self.vol, self.bright, self.warm,
                    self.vib, self.trem, self.rev, self.active,
                    self.move_gain, self.pluck, self.hand_dist)

_state = [State(), State()]

# ── body resonance filter banks ───────────────────────────────────
_PEAKS = {
    "guitar":   [(82,14),(196,9),(392,6),(800,4)],
    "piano":    [(55,8),(220,6),(440,5),(1760,4)],
    "harp":     [(65,14),(130,10),(260,7),(520,5)],
    "bass":     [(41,16),(82,12),(164,8),(330,5)],
    "violin":   [(196,18),(392,14),(588,10),(784,7)],   # rich upper partials
    "flute":    [(262,6),(524,5),(786,4),(1048,3)],     # clean airy peaks
    "sitar":    [(73,20),(147,16),(220,12),(440,9),(880,6)],  # many buzzy peaks
    "theremin": [(110,5),(220,4),(330,3),(440,3)],      # smooth few peaks
    "organ":    [(65,10),(130,9),(195,8),(260,7),(390,6),(520,5)],  # pipe harmonics
    "marimba":  [(262,18),(524,8),(786,5)],             # woody, sparse
}
def make_filters(mode):
    out = []
    for f0,Q in _PEAKS[mode]:
        if f0 < SR/2:
            b,a = signal.iirpeak(f0/(SR/2), Q)
            out.append([b, a, signal.lfilter_zi(b,a).copy()])
    return out

# ── guitar pluck envelope ─────────────────────────────────────────
def guitar_pluck_env(decay_time=0.55):
    """Attack 5ms + exponential string decay."""
    attack_s = int(SR * 0.005)
    total_s  = int(SR * (decay_time + 0.06))
    attack   = np.linspace(0, 1, attack_s, dtype=np.float32)
    decay_s  = total_s - attack_s
    decay    = np.exp(-np.arange(decay_s, dtype=np.float32) / (SR * decay_time))
    return np.concatenate([attack, decay])

def marimba_env(decay_time=0.35):
    """Very fast attack (1ms) + wooden mallet decay."""
    attack_s = int(SR * 0.001)
    total_s  = int(SR * (decay_time + 0.01))
    attack   = np.linspace(0, 1, attack_s, dtype=np.float32)
    decay_s  = total_s - attack_s
    # Marimba has a steeper initial drop then gentle tail
    t        = np.arange(decay_s, dtype=np.float32)
    decay    = np.exp(-t / (SR * decay_time * 0.3)) * 0.7 + \
               np.exp(-t / (SR * decay_time)) * 0.3
    return np.concatenate([attack, decay.astype(np.float32)])

def sitar_env(decay_time=0.8):
    """Attack 3ms + buzzy sitar decay with sympathetic string tail."""
    attack_s = int(SR * 0.003)
    total_s  = int(SR * (decay_time + 0.05))
    attack   = np.linspace(0, 1, attack_s, dtype=np.float32)
    decay_s  = total_s - attack_s
    t        = np.arange(decay_s, dtype=np.float32)
    decay    = np.exp(-t / (SR * decay_time * 0.4)) * 0.6 + \
               np.exp(-t / (SR * decay_time * 1.6)) * 0.4
    return np.concatenate([attack, decay.astype(np.float32)])

# ── synthesis ─────────────────────────────────────────────────────
def _consume_pluck(w: np.ndarray, S: dict):
    """Apply and advance stored pluck envelope in-place (silences if none)."""
    ppos = S.get('pluck_pos', None)
    penv = S.get('pluck_env', None)
    if penv is not None and ppos is not None:
        rem  = len(penv) - ppos
        take = min(CHUNK, rem)
        env_c = np.zeros(CHUNK, dtype=np.float32)
        if take > 0:
            env_c[:take] = penv[ppos:ppos+take]
        S['pluck_pos'] = ppos + take
        w *= env_c
    else:
        w *= 0.0


def synth_chunk(st: State, syn: dict) -> np.ndarray:
    freq_t, vol_t, bri_t, wrm_t, vib_t, tre_t, rev_t, \
        active, move_gain, pluck, hand_dist = st.read()

    if pluck:
        st.write(pluck=False)

    S  = syn
    sp = 0.25
    S['freq']  = S['freq']  * 0.85 + freq_t * 0.15
    S['vol']   = S['vol']   * sp   + vol_t  * (1-sp)
    S['bri']   = S['bri']   * sp   + bri_t  * (1-sp)
    S['wrm']   = S['wrm']   * sp   + wrm_t  * (1-sp)
    S['vib']   = S['vib']   * sp   + vib_t  * (1-sp)
    S['tre']   = S['tre']   * sp   + tre_t  * (1-sp)
    S['rev']   = S['rev']   * sp   + rev_t  * (1-sp)
    S['dist']  = S['dist']  * sp   + hand_dist * (1-sp)

    ms = S.get('move_smooth', 1.0)
    ms = ms * (0.5 if move_gain > ms else 0.92) + move_gain * (0.5 if move_gain > ms else 0.08)
    S['move_smooth'] = ms

    mode = S['mode']
    PLUCK_MODES = {"guitar", "sitar", "marimba", "harp"}

    # ── pluck-style envelope trigger ──────────────────────────────
    if mode in PLUCK_MODES:
        cool = S.get('pluck_cool', 0)
        if pluck or (move_gain > 0.4 and cool <= 0):
            dist_t = S['dist']
            if mode == "guitar":
                S['pluck_env'] = guitar_pluck_env(0.40 + dist_t * 0.55)
            elif mode == "sitar":
                S['pluck_env'] = sitar_env(0.60 + dist_t * 0.70)
            elif mode == "marimba":
                S['pluck_env'] = marimba_env(0.25 + dist_t * 0.30)
            elif mode == "harp":
                S['pluck_env'] = guitar_pluck_env(0.50 + dist_t * 0.80)
            S['pluck_pos']  = 0
            S['pluck_cool'] = max(int(SR * 0.18 / CHUNK), 2)
        elif cool > 0:
            S['pluck_cool'] = cool - 1

    # ── hand presence fade ────────────────────────────────────────
    fade_tgt  = 1.0 if active else 0.0
    S['fade'] = S['fade'] * 0.50 + fade_tgt * 0.50
    if S['fade'] < 0.005 and not active:
        return np.zeros(CHUNK, dtype=np.float32)

    freq = max(S['freq'], 20.0)

    # ── vibrato LFO ───────────────────────────────────────────────
    vib_rates = {"violin":6.2,"flute":5.0,"theremin":3.8,"sitar":5.8}
    vib_rate  = vib_rates.get(mode, 5.3)
    dist_vib  = S['dist'] * 0.022
    vib_ph    = S['vib_ph'] + 2*np.pi * vib_rate * np.arange(CHUNK) / SR
    vib_sig   = np.sin(vib_ph)
    S['vib_ph'] = float(vib_ph[-1]) % (2*np.pi)

    auto_vib  = 0.010 if mode == "violin" else 0.0
    freq_arr  = freq * (1.0 + (S['vib'] + dist_vib + auto_vib) * 0.016 * vib_sig)

    # Theremin: extra-smooth pitch glide
    if mode == "theremin":
        S['th_freq'] = S.get('th_freq', freq) * 0.94 + freq * 0.06
        freq_arr = S['th_freq'] * (1.0 + (S['vib'] + dist_vib) * 0.018 * vib_sig)

    # ── phase accumulator ─────────────────────────────────────────
    inc    = 2*np.pi * freq_arr / SR
    phases = S['phase'] + np.cumsum(inc)
    S['phase'] = float(phases[-1]) % (2*np.pi)

    bri = S['bri']; wrm = S['wrm']

    # ── base harmonic skeleton ────────────────────────────────────
    w  =              np.sin(phases)
    w += bri * 0.55 * np.sin(2*phases)
    w += bri * 0.30 * np.sin(3*phases)
    w += bri * 0.16 * np.sin(4*phases)
    w += bri * 0.08 * np.sin(5*phases)
    w += wrm * 0.22 * np.sin(0.5*phases)
    w += wrm * 0.11 * np.sin(1.5*phases)

    # ── instrument-specific synthesis ────────────────────────────
    if mode == "guitar":
        stretch = 1.0 + 0.0012 * (freq / 110.0)
        w += bri * 0.12 * np.sin(2 * stretch * phases)
        w += bri * 0.06 * np.sin(3 * stretch * phases)
        w += 0.012 * bri * np.random.randn(CHUNK).astype(np.float32)
        _consume_pluck(w, S)

    elif mode == "piano":
        w += bri * 0.06 * np.sin(phases * 2.0015)
        w += bri * 0.03 * np.sin(phases * 3.005)

    elif mode == "harp":
        dec = np.exp(-np.arange(CHUNK) / (SR * 0.25))
        w += 0.10 * np.sin(6*phases) * dec
        _consume_pluck(w, S)

    elif mode == "bass":
        w = np.tanh(w * (1.4 + wrm*1.4)) / (1.4 + wrm*0.5)

    elif mode == "violin":
        # Bowed string: blend sawtooth for richness + bow noise
        w_saw = ((phases % (2*np.pi)) / np.pi - 1.0).astype(np.float32)
        w     = w * 0.55 + w_saw * 0.45
        w    += 0.018 * wrm * np.random.randn(CHUNK).astype(np.float32)
        # Bow-speed shimmer
        w    *= (0.92 + 0.08 * np.sin(vib_ph * 0.37))

    elif mode == "flute":
        # Airy: mostly sine + filtered breath noise + embouchure flutter
        w  = np.sin(phases) * 0.75
        w += np.sin(2*phases) * bri * 0.18
        w += np.sin(3*phases) * bri * 0.06
        breath = 0.055 * np.random.randn(CHUNK).astype(np.float32)
        bp_freq = min(freq * 3 / (SR / 2), 0.95)
        b_lp, a_lp = signal.butter(2, bp_freq)
        zi_key = 'breath_zi'
        if zi_key not in S:
            S[zi_key] = signal.lfilter_zi(b_lp, a_lp)
        breath, S[zi_key] = signal.lfilter(b_lp, a_lp, breath, zi=S[zi_key])
        w += breath * (0.4 + wrm * 0.4)
        flutter_ph = S.get('flutter_ph', 0.0) + 2*np.pi * 11.3 * np.arange(CHUNK) / SR
        S['flutter_ph'] = float(flutter_ph[-1]) % (2*np.pi)
        w *= 1.0 + 0.025 * np.sin(flutter_ph)

    elif mode == "sitar":
        # Buzzy jawari + sympathetic drone
        stretch = 1.0008
        w = np.sin(phases)
        for k in range(2, 10):
            w += bri * (0.55 / k) * np.sin(k * stretch**(k-1) * phases)
        drone_ph = S.get('drone_ph', 0.0) + 2*np.pi*(freq*1.5)*np.arange(CHUNK)/SR
        S['drone_ph'] = float(drone_ph[-1]) % (2*np.pi)
        w += 0.08 * wrm * np.sin(drone_ph)
        w += 0.015 * np.random.randn(CHUNK).astype(np.float32)
        _consume_pluck(w, S)

    elif mode == "theremin":
        # Pure eerie electronic: smooth + ring-mod shimmer
        w  = np.sin(phases) * 0.80
        w += np.sin(2*phases) * bri * 0.25
        w += np.sin(3*phases) * bri * 0.10
        rm_ph = S.get('rm_ph', 0.0) + 2*np.pi*(freq*1.007)*np.arange(CHUNK)/SR
        S['rm_ph'] = float(rm_ph[-1]) % (2*np.pi)
        w *= (0.85 + 0.15 * np.sin(rm_ph))

    elif mode == "organ":
        # Hammond drawbar: 8 partials, Leslie rotary speaker effect
        draws  = [0.9, 1.0, 0.8, 0.5, 0.6, 0.3, 0.4, 0.2]
        ratios = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 8.0, 16.0]
        w = np.zeros(CHUNK, dtype=np.float32)
        for d, r in zip(draws, ratios):
            if r * freq < SR * 0.45:
                w += d * np.sin(r * phases)
        les_rate = 4.5 + wrm * 3.0
        les_ph   = S.get('les_ph', 0.0) + 2*np.pi * les_rate * np.arange(CHUNK) / SR
        S['les_ph'] = float(les_ph[-1]) % (2*np.pi)
        w *= 1.0 + 0.18 * np.sin(les_ph)

    elif mode == "marimba":
        # Wooden bar: strong fundamental + prominent 4th partial
        w  = np.sin(phases) * 0.85
        w += np.sin(4*phases) * bri * 0.35
        w += np.sin(2*phases) * bri * 0.10
        w += 0.008 * np.random.randn(CHUNK).astype(np.float32)
        _consume_pluck(w, S)

    w /= (1 + bri*1.05 + wrm*0.33 + 0.001)

    # ── body resonance ────────────────────────────────────────────
    res = np.zeros(CHUNK, dtype=np.float32)
    for filt in S['filters']:
        b, a, zi = filt
        r, zi_new = signal.lfilter(b, a, w, zi=zi)
        filt[2]   = zi_new
        res      += r
    res_mix = {"sitar":0.55,"organ":0.20,"violin":0.45,"marimba":0.30}.get(mode, 0.35)
    w = w + res * res_mix

    # ── tremolo ───────────────────────────────────────────────────
    tre_ph = S['tre_ph'] + 2*np.pi * 4.8 * np.arange(CHUNK) / SR
    trem   = 1.0 - S['tre'] * 0.28 * (0.5 + 0.5*np.sin(tre_ph))
    S['tre_ph'] = float(tre_ph[-1]) % (2*np.pi)
    w *= trem

    # ── comb reverb ───────────────────────────────────────────────
    rev_times = {"violin":0.09,"flute":0.09,"theremin":0.08,"organ":0.07}
    rtime = rev_times.get(mode, 0.067)
    delay = int(SR * rtime)
    rbuf  = S['rev_buf']; ridx = S['rev_idx']
    wet   = np.empty(CHUNK, dtype=np.float32)
    for k in range(CHUNK):
        ri          = (ridx - delay) % len(rbuf)
        wet[k]      = rbuf[ri]
        rbuf[ridx]  = w[k] + wet[k] * 0.52
        ridx        = (ridx+1) % len(rbuf)
    S['rev_idx'] = ridx
    rv = S['rev']
    if mode in ("theremin", "flute"):
        rv = min(rv + 0.12, 0.60)
    w  = w*(1-rv) + wet*rv

    # ── movement gate for non-pluck modes ────────────────────────
    if mode not in PLUCK_MODES:
        w *= ms

    w *= S['vol'] * S['fade']
    return w.astype(np.float32)


def to_sound(w):
    pcm = np.clip(w*32767, -32767, 32767).astype(np.int16)
    return pygame.sndarray.make_sound(np.column_stack([pcm,pcm]))

# ── audio thread ──────────────────────────────────────────────────
class AudioThread(threading.Thread):
    def __init__(self, st, ch_idx, mode_ref):
        super().__init__(daemon=True)
        self.st=st; self.ch_idx=ch_idx; self.mode_ref=mode_ref
        self._stop=threading.Event()

    def run(self):
        ch   = pygame.mixer.Channel(self.ch_idx)
        mode = self.mode_ref[0]
        syn  = dict(
            freq=130.81, vol=0.0, bri=0.5, wrm=0.3,
            vib=0.0, tre=0.0, rev=0.18, fade=0.0,
            dist=0.5, move_smooth=0.0,
            phase=0.0, vib_ph=0.0, tre_ph=0.0,
            filters=make_filters(mode),
            rev_buf=np.zeros(int(SR*0.12), dtype=np.float32),
            rev_idx=0, mode=mode,
            pluck_env=None, pluck_pos=0, pluck_cool=0,
        )
        while not self._stop.is_set():
            nm = self.mode_ref[0]
            if nm != syn['mode']:
                syn.update(mode=nm, filters=make_filters(nm),
                           pluck_env=None, pluck_pos=0, pluck_cool=0)
                syn['rev_buf'][:] = 0.0

            chunk = synth_chunk(self.st, syn)
            snd   = to_sound(chunk)
            if not ch.get_busy():
                ch.play(snd)
            else:
                t_end = time.perf_counter() + CHUNK_S
                while time.perf_counter() < t_end:
                    if ch.get_queue() is None:
                        ch.queue(snd); break
                    time.sleep(0.004)
                else:
                    ch.stop(); ch.play(snd)
            time.sleep(CHUNK_S * 0.55)

    def stop(self):
        self._stop.set()
        pygame.mixer.Channel(self.ch_idx).fadeout(300)

# ── hand landmark helpers ─────────────────────────────────────────
FINGER_IDS = {"thumb":[1,2,3,4],"index":[5,6,7,8],
              "middle":[9,10,11,12],"ring":[13,14,15,16],"pinky":[17,18,19,20]}
MCP_IDS=[5,9,13,17]; TIP_IDS=[4,8,12,16,20]

def _lms(h): return h if isinstance(h,(list,tuple)) else h.landmark

def _a3(a,b,c):
    va=a-b; vc=c-b
    la=np.linalg.norm(va); lc=np.linalg.norm(vc)
    if la<1e-9 or lc<1e-9: return math.pi/2
    return float(math.acos(np.clip(np.dot(va,vc)/(la*lc),-1,1)))

def extract(hand, fw, fh):
    lms = _lms(hand)
    pts = np.array([(lm.x*fw, lm.y*fh, lm.z*fw) for lm in lms])
    joint = {n:[_a3(pts[ids[i]],pts[ids[i+1]],pts[ids[i+2]])
                for i in range(len(ids)-2)]
             for n,ids in FINGER_IDS.items()}
    spread=[_a3(pts[MCP_IDS[i]],pts[0],pts[MCP_IDS[i+1]])
            for i in range(len(MCP_IDS)-1)]
    d=pts[9]-pts[0]
    tilt=math.atan2(-d[1],d[0])
    bend=math.atan2(-d[2],math.sqrt(d[0]**2+d[1]**2))
    tips=np.array([pts[t,:2] for t in TIP_IDS])
    open_=float(np.mean(np.linalg.norm(tips-pts[0,:2],axis=1)))/fw
    return dict(joint=joint,spread=spread,tilt=tilt,
                bend=bend,height=1-(pts[0,1]/fh),openness=open_,
                wxy=pts[0,:2].copy())

def to_params(ang, scale):
    j=ang['joint']
    def ext(n): return float(np.mean(j[n]))/math.pi if j[n] else 0.5
    pn  = np.clip(ext("index")*0.35+ext("middle")*0.30+
                  ext("ring")*0.20+ext("pinky")*0.10+ext("thumb")*0.05,0,1)
    sf  = pn*(len(scale)-1)
    lo  = int(sf); hi=min(lo+1,len(scale)-1)
    freq= m2f(scale[lo])*(1-(sf-lo))+m2f(scale[hi])*(sf-lo)
    freq*= 2.0**(np.clip(ang['tilt']/(math.pi*0.55),-1,1)*0.55)
    if j['thumb'] and j['index']:
        freq*=2**((j['thumb'][0]-j['index'][0])/math.pi*0.018)
    freq=min(freq, m2f(76))   # hard ceiling E5
    vol   =float(np.clip(0.3+ang['height']*0.70,0.05,1.0))
    bright=float(np.clip(ang['openness']*2.8*0.55+
                         (sum(j['index'])+sum(j['middle']))/(4*math.pi)*0.45,0,1))
    warm  =float(np.clip((1-ext("ring"))*0.5+(1-ext("pinky"))*0.5,0,1))
    vib   =float(np.clip((ang['spread'][0]-0.20)/0.45,0,1)*0.65) if ang['spread'] else 0
    trem  =float(np.clip((ang['spread'][-1]-0.15)/0.40,0,1)*0.45) if ang['spread'] else 0
    rev   =float(np.clip(0.08+ang['bend']*0.40,0.05,0.45))
    return dict(freq=freq,vol=vol,bright=bright,warm=warm,vib=vib,trem=trem,rev=rev)

# ── visuals ───────────────────────────────────────────────────────
_CONN=[(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
       (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),
       (13,17),(0,17),(17,18),(18,19),(19,20)]

def draw_hand(frame, hand, color, label=""):
    h,w=frame.shape[:2]; lms=_lms(hand)
    for a,b in _CONN:
        ax,ay=int(lms[a].x*w),int(lms[a].y*h)
        bx,by=int(lms[b].x*w),int(lms[b].y*h)
        cv2.line(frame,(ax,ay),(bx,by),tuple(c//4 for c in color),5,cv2.LINE_AA)
        cv2.line(frame,(ax,ay),(bx,by),color,2,cv2.LINE_AA)
    for i,lm in enumerate(lms):
        cx,cy=int(lm.x*w),int(lm.y*h)
        r=9 if i in TIP_IDS else (6 if i in [0,5,9,13,17] else 3)
        cv2.circle(frame,(cx,cy),r+2,(0,0,0),-1)
        cv2.circle(frame,(cx,cy),r,color,-1)
    if label:
        wx,wy=int(lms[0].x*w)-22,int(lms[0].y*h)+38
        cv2.putText(frame,label,(wx,wy),cv2.FONT_HERSHEY_DUPLEX,1.2,(0,0,0),5,cv2.LINE_AA)
        cv2.putText(frame,label,(wx,wy),cv2.FONT_HERSHEY_DUPLEX,1.2,color,1,cv2.LINE_AA)

def draw_dist_line(frame, h0, h1, dist_norm, color):
    h,w=frame.shape[:2]
    l0=_lms(h0); l1=_lms(h1)
    p0=(int(l0[0].x*w),int(l0[0].y*h))
    p1=(int(l1[0].x*w),int(l1[0].y*h))
    cv2.line(frame,p0,p1,(100,100,100),1,cv2.LINE_AA)
    mid=((p0[0]+p1[0])//2,(p0[1]+p1[1])//2)
    cv2.putText(frame,f"dist {dist_norm:.2f}",(mid[0]-30,mid[1]-10),
                cv2.FONT_HERSHEY_SIMPLEX,0.40,color,1,cv2.LINE_AA)

def draw_ui(frame, mode, syn, fps, move_gain):
    h,w=frame.shape[:2]; color=PALETTE[mode]
    ov=frame.copy()
    cv2.rectangle(ov,(w-225,0),(w,260),(12,12,12),-1)
    cv2.addWeighted(ov,0.65,frame,0.35,0,frame)

    def T(s,x,y,sc=0.54,c=None,bold=False):
        c=c or color
        f=cv2.FONT_HERSHEY_DUPLEX if bold else cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame,s,(x,y),f,sc,(0,0,0),3,cv2.LINE_AA)
        cv2.putText(frame,s,(x,y),f,sc,c,1,cv2.LINE_AA)

    fm  =syn.get('freq',130)
    note=int(np.clip(round(69+12*math.log2(max(fm,20)/440)),21,108))
    vol =syn.get('vol',0)
    T(f"[ {mode.upper()} ]",w-218,32,0.64,bold=True)
    T(f"NOTE  {midi_name(note)}",w-218,60)
    T(f"FREQ  {fm:.0f} Hz",w-218,84)
    T(f"VOL   {int(vol*100):3d}%",w-218,108)
    T(f"MOVE  {int(move_gain*100):3d}%",w-218,132)
    T(f"FPS   {fps:.0f}",w-218,156)

    bars=[("BRI",syn.get('bri',0)),("WRM",syn.get('wrm',0)),
          ("VIB",syn.get('vib',0)),("TRE",syn.get('tre',0)),
          ("REV",syn.get('rev',0)),("MOV",move_gain)]
    bw2,bh,gap=16,55,4
    for i,(nm,val) in enumerate(bars):
        bx=w-218+i*(bw2+gap); fld=int(val*bh)
        cv2.rectangle(frame,(bx,172),(bx+bw2,172+bh),(35,35,35),-1)
        if fld>0: cv2.rectangle(frame,(bx,172+bh-fld),(bx+bw2,172+bh),color,-1)
        cv2.putText(frame,nm,(bx-2,172+bh+13),cv2.FONT_HERSHEY_SIMPLEX,
                    0.28,(150,150,150),1,cv2.LINE_AA)

    cv2.rectangle(frame,(0,h-32),(w,h),(12,12,12),-1)
    T("[G]Guitar [P]Piano [H]Harp [B]Bass [V]Violin [F]Flute [S]Sitar [T]Theremin [O]Organ [M]Marimba  [Q]Quit",
      10,h-11,0.34,(160,160,160))
    bx2,by2,bw3=10,h-58,220
    for (lbl,val,yo) in [("VOL",vol,0),("PITCH",np.clip((fm-55)/500,0,1),-18),("MOVE",move_gain,-36)]:
        cv2.rectangle(frame,(bx2,by2-yo),(bx2+bw3,by2-yo+9),(40,40,40),-1)
        fv=int(bw3*np.clip(val,0,1))
        mc=(0,200,80) if (lbl=="MOVE" and val>0.15) else (60,60,200) if lbl=="MOVE" else color
        if fv>0: cv2.rectangle(frame,(bx2,by2-yo),(bx2+fv,by2-yo+9),mc,-1)
        T(lbl,bx2,by2-yo-3,0.34)

# ── model download ────────────────────────────────────────────────
MODEL_URL =("https://storage.googleapis.com/mediapipe-models/"
            "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
MODEL_PATH=os.path.join(os.path.dirname(os.path.abspath(__file__)),"hand_landmarker.task")
def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand model (~7 MB)…",flush=True)
        urllib.request.urlretrieve(MODEL_URL,MODEL_PATH)
        print("Done.")
    return MODEL_PATH

# ── main ──────────────────────────────────────────────────────────
def main():
    pygame.mixer.pre_init(frequency=SR,size=-16,channels=2,buffer=1024)
    pygame.init(); pygame.mixer.set_num_channels(8)

    cap=cv2.VideoCapture(0)
    if not cap.isOpened(): print("ERROR: no webcam"); return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)

    if _NEW_API:
        print("mediapipe >= 0.10  →  Tasks API")
        base=_mpt.BaseOptions(model_asset_path=ensure_model())
        opts=_mpv.HandLandmarkerOptions(base_options=base,num_hands=2,
             min_hand_detection_confidence=0.65,
             min_hand_presence_confidence=0.65,
             min_tracking_confidence=0.55)
        _det=_mpv.HandLandmarker.create_from_options(opts)
        def detect(bgr):
            rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
            res=_det.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb))
            return res.hand_landmarks
    else:
        print("mediapipe legacy")
        _h=mp.solutions.hands.Hands(static_image_mode=False,max_num_hands=2,
           min_detection_confidence=0.65,min_tracking_confidence=0.55)
        def detect(bgr):
            rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
            return _h.process(rgb).multi_hand_landmarks or []

    mode_ref=["guitar"]
    threads=[AudioThread(_state[i],i,mode_ref) for i in range(2)]
    for t in threads: t.start()

    ui_syn=dict(freq=130.0,vol=0.0,bri=0.5,wrm=0.3,vib=0.0,tre=0.0,rev=0.18)
    fps_q=deque(maxlen=20); t0=time.perf_counter()

    prev_wxy   =[None,None]
    prev_t     =[time.perf_counter()]*2
    still_since=[time.perf_counter()]*2
    move_speed =[0.0,0.0]

    print("\n  Air Instrument v7  —  LIVE")
    print("  G=Guitar  P=Piano  H=Harp   B=Bass")
    print("  V=Violin  F=Flute  S=Sitar  T=Theremin  O=Organ  M=Marimba")
    print("  MOVE your hands to make sound!\n")

    while True:
        ok,frame=cap.read()
        if not ok: break
        frame=cv2.flip(frame,1)
        fh,fw=frame.shape[:2]
        hands=detect(frame)

        now=time.perf_counter()
        fps_q.append(1.0/max(now-t0,1e-6)); t0=now
        fps=float(np.mean(fps_q))

        # ── inter-hand distance ────────────────────────────────────
        hand_dist_norm=0.5
        if len(hands)>=2:
            l0=_lms(hands[0]); l1=_lms(hands[1])
            p0=np.array([l0[0].x*fw,l0[0].y*fh])
            p1=np.array([l1[0].x*fw,l1[0].y*fh])
            hand_dist_norm=float(np.clip(np.linalg.norm(p0-p1)/fw,0,1))

        active_set=set()
        for i,hand in enumerate(hands[:2]):
            ang=extract(hand,fw,fh)
            p  =to_params(ang,SCALES[mode_ref[0]])
            wxy=ang['wxy']

            if prev_wxy[i] is not None:
                dt   =max(now-prev_t[i],1e-4)
                speed=float(np.linalg.norm(wxy-prev_wxy[i]))/dt
            else:
                speed=0.0

            a=0.6 if speed>move_speed[i] else 0.92
            move_speed[i]=move_speed[i]*a+speed*(1-a)
            if speed>MOVE_THRESH: still_since[i]=now

            sd=now-still_since[i]
            if sd<STILL_FADEOUT: mg=1.0
            elif sd<STILL_FULL:  mg=1.0-(sd-STILL_FADEOUT)/(STILL_FULL-STILL_FADEOUT)
            else:                mg=0.0

            is_pluck=(speed>35 and prev_wxy[i] is not None and
                      float(np.linalg.norm(wxy-prev_wxy[i]))>6)

            _state[i].write(freq=p['freq'],vol=p['vol'],bright=p['bright'],
                            warm=p['warm'],vib=p['vib'],trem=p['trem'],rev=p['rev'],
                            active=True,move_gain=float(mg),pluck=is_pluck,
                            hand_dist=hand_dist_norm)
            prev_wxy[i]=wxy.copy(); prev_t[i]=now
            active_set.add(i)

            if i==0:
                note=int(np.clip(round(69+12*math.log2(max(p['freq'],20)/440)),21,108))
                ui_syn.update(freq=p['freq'],vol=p['vol'],bri=p['bright'],
                              wrm=p['warm'],vib=p['vib'],tre=p['trem'],rev=p['rev'])
                draw_hand(frame,hand,PALETTE[mode_ref[0]],midi_name(note))
            else:
                draw_hand(frame,hand,PALETTE[mode_ref[0]])

        if len(hands)>=2:
            draw_dist_line(frame,hands[0],hands[1],hand_dist_norm,PALETTE[mode_ref[0]])

        for i in range(2):
            if i not in active_set:
                _state[i].write(active=False,move_gain=0.0)
                prev_wxy[i]=None; still_since[i]=now

        mg_ui=float(np.clip(move_speed[0]/150.0,0,1)) if 0 in active_set else 0.0
        draw_ui(frame,mode_ref[0],ui_syn,fps,mg_ui)
        cv2.imshow("Air Instrument v7",frame)

        key=cv2.waitKey(1)&0xFF
        if key==ord('q'): break
        elif key in (ord('g'),ord('p'),ord('h'),ord('b'),
                     ord('v'),ord('f'),ord('s'),ord('t'),ord('o'),ord('m')):
            mode_ref[0]={'g':'guitar','p':'piano','h':'harp','b':'bass',
                         'v':'violin','f':'flute','s':'sitar','t':'theremin',
                         'o':'organ','m':'marimba'}[chr(key)]
            print(f"  → {mode_ref[0].upper()}")

    for t in threads: t.stop()
    cap.release(); cv2.destroyAllWindows(); pygame.quit()
    print("Bye!")

if __name__=="__main__":
    main()