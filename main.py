"""
Air Virtual Instrument  v8 - REAL INSTRUMENTS
──────────────────────────────────────────────────────────────────────
INSTALL:  pip install mediapipe opencv-python numpy pygame scipy soundfile
RUN:      python main.py

CONTROLS  (press key to switch instrument)
  G  Guitar    P  Piano    H  Harp     B  Bass
  V  Violin    F  Flute    S  Sitar    T  Theremin
  O  Organ     M  Marimba  U  Trumpet   K  Saxophone
  Q  Quit

HAND → SOUND
  ─── NOTE TRIGGERING ───
  • Hand enters frame + finger pinch → attack note
  • Hand leaves frame                → release note
  • Finger spread (index-middle)     → vibrato
  ─── CONTINUOUS CONTROL ───
  • Hand height                      → volume
  • Finger curl (which fingers)     → pitch (note selection)
  • Wrist tilt L/R                  → pitch bend / octave
  • Hand movement speed             → brightness / dynamics
  • Open palm                       → brightness
  • Fist (curled fingers)           → warmth
──────────────────────────────────────────────────────────────────────
"""

import cv2, numpy as np, pygame, pygame.sndarray
import math, time, os, threading, urllib.request
from collections import deque
from scipy import signal

_NEW_API = False
try:
    import mediapipe as mp
    from mediapipe.tasks import python as _mpt
    from mediapipe.tasks.python import vision as _mpv
    _NEW_API = True
except ImportError:
    import mediapipe as mp

# ── constants ─────────────────────────────────────────────────────
SR       = 44100
CHUNK_S  = 0.05       # 50ms chunks for tighter response
CHUNK    = int(SR * CHUNK_S)

MOVE_THRESH   = 3.5    
STILL_FADEOUT = 0.3    
STILL_FULL    = 0.8    

NOTE_NAMES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
def midi_name(m): return f"{NOTE_NAMES[int(m)%12]}{int(m)//12-1}"
def m2f(m):       return 440.0 * 2.0**((m-69)/12.0)

# Full piano-like range with multiple octaves
PENTATONIC  = [36,38,40,43,45,48,50,52,55,57,60,62,64,67,69,72,74,76,79,81]
MAJOR       = [36,38,40,41,43,45,47,48,50,52,53,55,57,59,60,62,64,65,67,69,72,74,76]
NATURAL_MIN = [36,38,39,41,43,44,46,48,50,51,53,55,56,58,60,62,63,65,67,69,71,72,74,76]
HARP_SC     = [36,38,40,41,43,45,47,48,50,52,53,55,57,59,60,62,64,65,67,69,72]
BASS_SC     = [28,30,31,33,35,36,38,40,41,43,45,47,48,50,52,53,55]
VIOLIN_SC   = [55,57,59,60,62,64,65,67,69,71,72,74,76,77,79,81,83,84]
FLUTE_SC    = [60,62,64,65,67,69,71,72,74,76,77,79,81,83,84,86]
SITAR_SC    = [40,42,43,45,47,48,50,52,53,55,57,58,60,62,64,65,67,69]
THEREMIN_SC = list(range(36, 72))        # Full 3 octave chromatic
ORGAN_SC    = [36,38,40,41,43,45,47,48,50,52,53,55,57,59,60,62,64,65,67,69,72]
MARIMBA_SC  = [48,50,52,53,55,57,59,60,62,64,65,67,69,71,72,74,76]
TRUMPET_SC  = [55,57,59,60,62,63,65,67,68,70,72,74,75,77,79,80,82,84]
SAX_SC      = [48,50,51,53,55,56,58,60,61,63,65,66,68,70,71,73,75,77,78,80,82]

SCALES  = {"guitar":PENTATONIC,"piano":MAJOR,"harp":HARP_SC,"bass":BASS_SC,
           "violin":VIOLIN_SC,"flute":FLUTE_SC,"sitar":SITAR_SC,
           "theremin":THEREMIN_SC,"organ":ORGAN_SC,"marimba":MARIMBA_SC,
           "trumpet":TRUMPET_SC,"saxophone":SAX_SC}
PALETTE = {"guitar":(40,200,80),"piano":(160,100,255),
           "harp":(255,190,50),"bass":(50,160,255),
           "violin":(220,80,80),"flute":(80,220,220),
           "sitar":(255,140,0),"theremin":(180,0,255),
           "organ":(255,60,180),"marimba":(60,200,140),
           "trumpet":(255,200,50),"saxophone":(255,120,60)}

# ── shared state ──────────────────────────────────────────────────
class State:
    def __init__(self):
        self.freq       = 130.81
        self.vol        = 0.0
        self.bright     = 0.5
        self.warm       = 0.3
        self.vib        = 0.0
        self.trem       = 0.0
        self.rev        = 0.18
        self.active     = False
        self.move_gain  = 0.0
        self.pluck      = False
        self.hand_dist  = 0.5
        self.note_on    = False    # New: note triggering
        self.note_vel   = 0.0      # velocity 0-1
        self._lock      = threading.Lock()

    def write(self, **kw):
        with self._lock:
            for k,v in kw.items(): setattr(self,k,v)

    def read(self):
        with self._lock:
            return (self.freq, self.vol, self.bright, self.warm,
                    self.vib, self.trem, self.rev, self.active,
                    self.move_gain, self.pluck, self.hand_dist,
                    self.note_on, self.note_vel)

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

# ── ADSR envelope ─────────────────────────────────────────────────
def adsr(attack=0.01, decay=0.1, sustain=0.7, release=0.3, length=CHUNK, sample_rate=SR):
    env = np.zeros(length, dtype=np.float32)
    
    at = min(int(attack * sample_rate), length - 1)
    dc = min(int(decay * sample_rate), length - at - 1)
    re = min(int(release * sample_rate), length - at - dc - 1)
    
    if at > 0:
        env[:at] = np.linspace(0, 1, at)
    if dc > 0:
        env[at:at+dc] = np.linspace(1, sustain, dc)
    sus_end = at + dc + re
    if sus_end < length:
        env[at+dc:sus_end] = sustain
    if re > 0 and sus_end < length:
        env[sus_end:] = np.linspace(sustain, 0, length - sus_end)
    
    return env


# ── improved synthesis ─────────────────────────────────────────────
def synth_chunk(st: State, syn: dict) -> np.ndarray:
    freq_t, vol_t, bri_t, wrm_t, vib_t, tre_t, rev_t, \
        active, move_gain, pluck, hand_dist, note_on, note_vel = st.read()

    S = syn
    mode = S['mode']
    
    sp = 0.3
    S['freq']  = S['freq']  * 0.8 + freq_t * 0.2
    S['vol']   = S['vol']   * sp   + vol_t  * (1-sp)
    S['bri']   = S['bri']   * sp   + bri_t  * (1-sp)
    S['wrm']   = S['wrm']   * sp   + wrm_t  * (1-sp)
    S['vib']   = S['vib']   * sp   + vib_t  * (1-sp)
    S['tre']   = S['tre']   * sp   + tre_t  * (1-sp)
    S['rev']   = S['rev']   * sp   + rev_t  * (1-sp)

    # Handle note on/off with ADSR
    prev_on = S.get('note_on', False)
    
    # Simple continuous envelope - ALWAYS PLAY when active
    if note_on and not prev_on:
        S['note_on'] = True
        S['attack_phase'] = 0.0  # For quick attack
        
    elif not note_on and prev_on:
        S['note_on'] = False
        
    # Continuous sound - use volume directly
    if active:
        # Quick attack, then sustained
        if S.get('attack_phase', 1.0) < 1.0:
            S['attack_phase'] = min(1.0, S.get('attack_phase', 0.0) + 0.15)
        env = np.ones(CHUNK, dtype=np.float32) * S['attack_phase']
    else:
        env = np.zeros(CHUNK, dtype=np.float32)
        S['attack_phase'] = 0.0

    freq = max(S['freq'], 20.0)

    # Vibrato
    vib_rates = {"violin":5.0, "flute":4.5, "theremin":3.5, "sitar":5.5,
                 "trumpet":6.0, "saxophone":4.0}
    vib_rate = vib_rates.get(mode, 5.0)
    vib_ph = S.get('vib_ph', 0.0) + 2*np.pi * vib_rate * np.arange(CHUNK) / SR
    vib_sig = np.sin(vib_ph)
    S['vib_ph'] = float(vib_ph[-1]) % (2*np.pi)
    
    freq_arr = freq * (1.0 + S['vib'] * 0.02 * vib_sig)

    # Theremin smooth glide
    if mode == "theremin":
        S['th_freq'] = S.get('th_freq', freq) * 0.92 + freq * 0.08
        freq_arr = S['th_freq'] * (1.0 + S['vib'] * 0.025 * vib_sig)

    # Phase accumulator
    inc = 2*np.pi * freq_arr / SR
    phases = S.get('phase', 0.0) + np.cumsum(inc)
    S['phase'] = float(phases[-1]) % (2*np.pi)

    bri = S['bri']
    wrm = S['wrm']

    # ── BASE WAVEFORMS ────────────────────────────────────────────
    # Multiple waveforms for richness
    w = np.sin(phases)
    
    # Sawtooth for brightness
    w_saw = 2.0 * ((phases % (2*np.pi)) / (2*np.pi)) - 1.0
    
    # Square for thickness
    w_sq = np.sign(np.sin(phases))
    
    # Triangle
    w_tri = 2.0 * np.abs(2.0 * ((phases % (2*np.pi)) / (2*np.pi)) - 1.0) - 1.0

    # ── INSTRUMENT SPECIFIC SYNTHESIS ─────────────────────────────
    if mode == "guitar":
        # Plucked string with harmonics
        w = w * 0.4 + w_saw * 0.35 * bri + w_tri * 0.25 * (1-bri)
        stretch = 1.0 + 0.001 * (freq / 110)
        for h in [2, 3, 4, 5]:
            w += (bri * 0.15 / h) * np.sin(h * stretch * phases)
        # Pick noise
        w += 0.008 * np.random.randn(CHUNK).astype(np.float32)
        
    elif mode == "piano":
        # Bright attack, warm body
        w = w * 0.5 + w_saw * 0.3 * bri + w_tri * 0.2
        # Inharmonic partials (piano has stretched harmonics)
        for h in [2, 3, 4, 5, 6]:
            stretch = 1.0 + 0.01 * (h - 1) * (freq / 200)
            w += (0.08 / h) * np.sin(h * stretch * phases)
            
    elif mode == "harp":
        w = w * 0.6 + w_saw * 0.25 * bri + w_tri * 0.15
        # String resonance
        w += 0.1 * np.sin(2*phases) * np.exp(-np.arange(CHUNK) / (SR * 0.3))
        
    elif mode == "bass":
        # Sub bass + warmth
        w = w * 0.3 + w_saw * 0.2 + np.sin(0.5*phases) * 0.5
        # Distortion for punch
        w = np.tanh(w * (1.5 + wrm * 1.5)) / (1.5 + wrm * 0.3)
        
    elif mode == "violin":
        # Bowed: sawtooth richness + vibrato modulation
        w = w * 0.4 + w_saw * 0.5 + w_sq * 0.1 * wrm
        # Bow noise
        w += 0.015 * wrm * np.random.randn(CHUNK).astype(np.float32)
        # Shimmer from bow speed
        w *= (0.9 + 0.1 * np.sin(vib_ph * 0.5))
        
    elif mode == "flute":
        # Airy with breath noise
        w = w * 0.7 + w_tri * 0.2 + np.sin(2*phases) * 0.1 * bri
        # Breath
        breath = 0.04 * np.random.randn(CHUNK).astype(np.float32)
        b, a = signal.butter(2, 0.3)
        zi = S.get('breath_zi')
        if zi is None:
            zi = signal.lfilter_zi(b, a)
        breath_filt, new_zi = signal.lfilter(b, a, breath, zi=zi)
        S['breath_zi'] = new_zi
        w += breath_filt * (0.3 + wrm * 0.4)
        # Flutter
        flutter = 1.0 + 0.02 * np.sin(2*np.pi * 12 * np.arange(CHUNK) / SR)
        w *= flutter
        
    elif mode == "sitar":
        # Buzzy with drone strings
        w = np.sin(phases)
        for h in range(2, 8):
            w += (bri * 0.4 / h) * np.sin(h * 1.001 * phases)
        # Sympathetic drone
        drone_freq = freq * 1.5
        drone_inc = 2*np.pi * drone_freq / SR
        drone_ph = S.get('drone_ph', 0.0) + np.cumsum(np.full(CHUNK, drone_inc))
        S['drone_ph'] = drone_ph[-1] % (2*np.pi)
        w += 0.06 * wrm * np.sin(drone_ph)
        # Jawari buzz
        w += 0.012 * np.random.randn(CHUNK).astype(np.float32)
        
    elif mode == "theremin":
        # Pure, eerie, with ring modulation
        w = np.sin(phases) * 0.7 + np.sin(2*phases) * 0.2 * bri + np.sin(3*phases) * 0.1 * bri
        # Ring mod
        rm_ph = S.get('rm_ph', 0.0) + 2*np.pi * freq * 1.01 * np.arange(CHUNK) / SR
        S['rm_ph'] = rm_ph[-1] % (2*np.pi)
        w *= (0.8 + 0.2 * np.sin(rm_ph))
        
    elif mode == "organ":
        # Hammond drawbars
        draws = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
        ratios = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 12.0]
        w = np.zeros(CHUNK, dtype=np.float32)
        for d, r in zip(draws, ratios):
            if r * freq < SR * 0.45:
                w += d * np.sin(r * phases)
        # Leslie rotation
        les_rate = 5.0 + wrm * 2.0
        les_ph = S.get('les_ph', 0.0) + 2*np.pi * les_rate * np.arange(CHUNK) / SR
        S['les_ph'] = les_ph[-1] % (2*np.pi)
        w *= (0.85 + 0.15 * np.sin(les_ph))
        
    elif mode == "marimba":
        # Wooden bar: strong fundamental + 4th partial (characteristic)
        w = np.sin(phases) * 0.8 + np.sin(4*phases) * 0.25 * bri + np.sin(2*phases) * 0.1
        # Slight noise for attack
        w += 0.005 * np.random.randn(CHUNK).astype(np.float32)
        
    elif mode == "trumpet":
        # Brassy with fast attack
        w = w * 0.4 + w_saw * 0.5 * (0.5 + bri*0.5) + w_sq * 0.1 * wrm
        # Harmonic series
        for h in [2, 3, 4, 5]:
            w += (0.2 / h) * np.sin(h * phases)
        # Brightness from falloff
        w *= (1.0 + bri * 0.3)
        
    elif mode == "saxophone":
        # Reedy, warm
        w = w * 0.35 + w_saw * 0.4 + w_sq * 0.25 * wrm
        # Rich harmonics
        for h in [2, 3, 4, 5, 6]:
            w += (0.12 / h) * np.sin(h * phases)
        # Breath
        w += 0.02 * np.random.randn(CHUNK).astype(np.float32)

    # Normalize
    w /= (1.0 + bri * 0.5 + wrm * 0.3 + 0.001)

    # ── BODY RESONANCE (formants) ──────────────────────────────────
    res = np.zeros(CHUNK, dtype=np.float32)
    for b, a, zi in S.get('filters', []):
        r, zi_new = signal.lfilter(b, a, w, zi=zi)
        res += r
    res_mix = {"sitar":0.5, "organ":0.15, "violin":0.4, "marimba":0.25,
               "trumpet":0.2, "saxophone":0.35, "flute":0.2, "theremin":0.1}.get(mode, 0.3)
    w = w + res * res_mix

    # ── TREMOLO ───────────────────────────────────────────────────
    tre_ph = S.get('tre_ph', 0.0) + 2*np.pi * 5.0 * np.arange(CHUNK) / SR
    trem = 1.0 - S['tre'] * 0.25 * (0.5 + 0.5*np.sin(tre_ph))
    S['tre_ph'] = tre_ph[-1] % (2*np.pi)
    w *= trem

    # ── REVERB (Schroeder) ─────────────────────────────────────────
    rv = S['rev']
    if mode in ("theremin", "flute", "organ"):
        rv = min(rv + 0.1, 0.5)
    
    rbuf = S.get('rev_buf')
    if rbuf is not None:
        ridx = S.get('rev_idx', 0)
        delay = int(SR * 0.08)
        wet = np.empty(CHUNK, dtype=np.float32)
        for k in range(CHUNK):
            ri = (ridx - delay) % len(rbuf)
            wet[k] = rbuf[ri]
            rbuf[ridx] = w[k] + wet[k] * 0.5
            ridx = (ridx + 1) % len(rbuf)
        S['rev_idx'] = ridx
        w = w * (1 - rv) + wet * rv

    # ── APPLY ENVELOPE & VOLUME ────────────────────────────────────
    # Continuous sound - always play when active
    w *= env  # Attack envelope
    w *= S['vol']  # Volume from hand position
    
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
    
    # Individual finger extensions (0=straight, 1=curled)
    def ext(n): return float(np.mean(j[n]))/math.pi if j[n] else 0.5
    
    # Each finger controls different pitch zone - more responsive
    idx_ext = ext("index")
    mid_ext = ext("middle")
    rng_ext = ext("ring")
    pnk_ext = ext("pinky")
    thm_ext = ext("thumb")
    
    # Combine for note selection - very sensitive to any curl
    pn = np.clip(idx_ext*0.30 + mid_ext*0.25 + rng_ext*0.20 + pnk_ext*0.15 + thm_ext*0.10, 0, 1)
    
    # Map to scale with interpolation for smooth glissando
    sf = pn * (len(scale) - 1)
    lo = int(sf)
    hi = min(lo + 1, len(scale) - 1)
    freq = m2f(scale[lo]) * (1 - (sf - lo)) + m2f(scale[hi]) * (sf - lo)
    
    # Wrist tilt = octave shift (more responsive)
    tilt_factor = np.clip(ang['tilt'] / (math.pi * 0.4), -1, 1)
    freq *= 2.0 ** (tilt_factor * 0.5)
    
    # Thumb-index distance = fine pitch bend
    if j['thumb'] and j['index']:
        freq *= 2 ** ((j['thumb'][0] - j['index'][0]) / math.pi * 0.025)
    
    # Hand vertical position = pitch (higher = higher)
    freq *= 2.0 ** ((1.0 - ang['height']) * 0.25)
    
    # No hard ceiling - let it ring!
    
    # Volume from height - always playing
    vol = float(np.clip(0.5 + ang['height'] * 0.5, 0.2, 1.0))
    
    # Brightness from fingertips - very sensitive
    bright = float(np.clip(ang['openness'] * 2.5 + 
                           (sum(j['index']) + sum(j['middle'])) / (4 * math.pi) * 0.5, 0, 1))
    
    # Warmth from ring/pinky curl
    warm = float(np.clip((1 - rng_ext) * 0.5 + (1 - pnk_ext) * 0.5, 0, 1))
    
    # Vibrato from index-middle spread - very sensitive
    vib = float(np.clip((ang['spread'][0] - 0.15) / 0.5, 0, 1) * 0.7) if ang['spread'] else 0
    
    # Tremolo from ring-pinky spread
    trem = float(np.clip((ang['spread'][-1] - 0.12) / 0.45, 0, 1) * 0.5) if ang['spread'] else 0
    
    # Reverb from forward bend
    rev = float(np.clip(0.10 + ang['bend'] * 0.35, 0.05, 0.5))
    
    return dict(freq=freq, vol=vol, bright=bright, warm=warm, vib=vib, trem=trem, rev=rev)

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
    T("[G]Guitar [P]Piano [H]Harp [B]Bass [V]Violin [F]Flute [S]Sitar [T]Theremin [O]Organ [M]Marimba [U]Trumpet [K]Sax  [Q]Quit",
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
    if not cap.isOpened(): 
        print("ERROR: no webcam found. Make sure a webcam is connected."); 
        return
    
    # Try to set resolution, but don't fail if it doesn't work
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
    
    # Verify we can read a frame
    ok, test_frame = cap.read()
    if not ok or test_frame is None:
        print("ERROR: couldn't read from webcam")
        cap.release()
        return

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

    prev_wxy      =[None,None]
    prev_t        =[time.perf_counter()]*2
    still_since   =[time.perf_counter()]*2
    move_speed    =[0.0,0.0]
    prev_freq_arr ={}

    print("\n  Air Instrument v9  —  ALWAYS ON - CONTINUOUS")
    print("  G=Guitar  P=Piano  H=Harp   B=Bass")
    print("  V=Violin  F=Flute  S=Sitar  T=Theremin  O=Organ  M=Marimba")
    print("  U=Trumpet  K=Saxophone")
    print("  Hand in frame = SOUND! Move fingers to change pitch/timbre\n")

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
        
        # ── ALWAYS ON - Continuous sound mapping ─────────────────────
        for i,hand in enumerate(hands[:2]):
            ang=extract(hand,fw,fh)
            p  =to_params(ang,SCALES[mode_ref[0]])
            wxy=ang['wxy']

            was_active = prev_wxy[i] is not None
            
            # Tiny movement detection - very sensitive
            if was_active:
                dt   =max(now-prev_t[i],1e-4)
                speed=float(np.linalg.norm(wxy-prev_wxy[i]))/dt
            else:
                speed=0.0

            # Much faster smoothing for responsiveness
            a=0.7 if speed>move_speed[i] else 0.92
            move_speed[i]=move_speed[i]*a+speed*(1-a)
            
            # Always update still_since but with very long timeout
            still_since[i]=now

            # Continuous play - NEVER STOP - always full volume when hand detected
            mg = 1.0  # Always full gain

            # Trigger note on first appearance, then continuous
            note_trigger = not was_active
            
            # Very easy pluck trigger
            is_pluck = (speed>15 and was_active)

            # Use hand height more aggressively for volume
            vol = float(np.clip(0.4 + ang['height'] * 0.6, 0.1, 1.0))
            
            # Tiny movements = tiny pitch changes for smooth glissando
            freq = p['freq']
            if was_active:
                # Smooth frequency transitions
                prev_freq = prev_freq_arr.get(i, freq)
                freq = prev_freq * 0.85 + freq * 0.15
            
            prev_freq_arr[i] = freq

            _state[i].write(freq=freq, vol=vol, bright=p['bright'],
                            warm=p['warm'], vib=p['vib'], trem=p['trem'], rev=p['rev'],
                            active=True, move_gain=float(mg), pluck=is_pluck,
                            hand_dist=hand_dist_norm,
                            note_on=note_trigger, note_vel=vol)
            prev_wxy[i]=wxy.copy(); prev_t[i]=now
            active_set.add(i)

            if i==0:
                note=int(np.clip(round(69+12*math.log2(max(freq,20)/440)),21,108))
                ui_syn.update(freq=freq, vol=vol, bri=p['bright'],
                              wrm=p['warm'], vib=p['vib'], tre=p['trem'], rev=p['rev'])
                draw_hand(frame,hand,PALETTE[mode_ref[0]],midi_name(note))
            else:
                draw_hand(frame,hand,PALETTE[mode_ref[0]])

        if len(hands)>=2:
            draw_dist_line(frame,hands[0],hands[1],hand_dist_norm,PALETTE[mode_ref[0]])

        # Hands that disappeared - fade out quickly but not instant
        for i in range(2):
            if i not in active_set:
                _state[i].write(active=False, move_gain=0.0, note_on=False, note_vel=0.0)
                prev_wxy[i]=None; still_since[i]=now

        mg_ui=float(np.clip(move_speed[0]/150.0,0,1)) if 0 in active_set else 0.0
        draw_ui(frame,mode_ref[0],ui_syn,fps,mg_ui)
        cv2.imshow("Air Instrument v8 - Real Instruments",frame)

        key=cv2.waitKey(1)&0xFF
        if key==ord('q'): break
        elif key in (ord('g'),ord('p'),ord('h'),ord('b'),
                     ord('v'),ord('f'),ord('s'),ord('t'),ord('o'),ord('m'),
                     ord('u'),ord('k')):
            mode_ref[0]={'g':'guitar','p':'piano','h':'harp','b':'bass',
                         'v':'violin','f':'flute','s':'sitar','t':'theremin',
                         'o':'organ','m':'marimba','u':'trumpet','k':'saxophone'}[chr(key)]
            print(f"  → {mode_ref[0].upper()}")

    for t in threads: t.stop()
    cap.release(); cv2.destroyAllWindows(); pygame.quit()
    print("Bye!")

if __name__=="__main__":
    main()