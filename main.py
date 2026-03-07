"""
Air Virtual Instrument  v5
──────────────────────────────────────────────────────────────────────
INSTALL:  pip install mediapipe opencv-python numpy pygame scipy
RUN:      python main.py

CONTROLS
  G  Guitar   P  Piano   H  Harp   B  Bass   Q  Quit

HAND → SOUND  (every joint angle tracked)
  Index + Middle curl   →  pitch  (smooth glide)
  Ring  + Pinky  curl   →  warmth / body
  Finger tip angles     →  brightness / harmonics
  Index-Middle spread   →  vibrato depth
  Ring-Pinky   spread   →  tremolo depth
  Wrist tilt  L / R     →  octave bend  (continuous)
  Wrist forward bend    →  reverb mix
  Hand height           →  volume
  Hand speed            →  attack sharpness
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
CHUNK_S  = 0.15               # 150 ms per baked chunk
CHUNK    = int(SR * CHUNK_S)

NOTE_NAMES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
def midi_name(m): return f"{NOTE_NAMES[int(m)%12]}{int(m)//12-1}"
def m2f(m):       return 440.0 * 2.0**((m-69)/12.0)

PENTATONIC = [48,50,52,55,57,60,62,64,67,69,72,74,76,79,81]
MAJOR      = [48,50,52,53,55,57,59,60,62,64,65,67,69,71,72,74]
HARP_SC    = [48,50,52,53,55,57,59,60,62,64,65,67,69,71,72]
BASS_SC    = [28,30,31,33,35,36,38,40,41,43,45,47]
SCALES     = {"guitar":PENTATONIC,"piano":MAJOR,"harp":HARP_SC,"bass":BASS_SC}
PALETTE    = {"guitar":(40,200,80),"piano":(160,100,255),
              "harp":(255,190,50),"bass":(50,160,255)}

# ── shared state (vision thread writes, audio thread reads) ───────
class State:
    def __init__(self):
        self.freq    = 261.63   # target Hz
        self.vol     = 0.0      # target volume  0-1
        self.bright  = 0.5
        self.warm    = 0.3
        self.vib     = 0.0
        self.trem    = 0.0
        self.rev     = 0.18
        self.active  = False    # hand visible?
        self._lock   = threading.Lock()

    def write(self, **kw):
        with self._lock:
            for k,v in kw.items(): setattr(self,k,v)

    def read(self):
        with self._lock:
            return (self.freq, self.vol, self.bright, self.warm,
                    self.vib, self.trem, self.rev, self.active)

_state = [State(), State()]

# ── body resonance filter banks ───────────────────────────────────
_PEAKS = {
    "guitar": [(82,14),(196,9),(392,6),(800,4)],
    "piano":  [(55,8),(220,6),(440,5),(1760,4)],
    "harp":   [(65,14),(130,10),(260,7),(520,5)],
    "bass":   [(41,16),(82,12),(164,8),(330,5)],
}
def make_filters(mode):
    out = []
    for f0,Q in _PEAKS[mode]:
        if f0 < SR/2:
            b,a = signal.iirpeak(f0/(SR/2), Q)
            out.append([b, a, signal.lfilter_zi(b,a).copy()])
    return out

# ── synthesis ─────────────────────────────────────────────────────
def synth_chunk(st: State, syn: dict) -> np.ndarray:
    """
    Render CHUNK float32 samples.
    syn = persistent synthesis state dict (phase accumulators, filter zis, etc.)
    All smoothing happens here so the audio thread owns it.
    """
    freq_t, vol_t, bri_t, wrm_t, vib_t, tre_t, rev_t, active = st.read()

    # ── smooth all params toward targets ──────────────────────────
    S = syn
    sp = 0.25   # smoothing per chunk: 0=instant 1=never-changes
    # pitch uses slower smoothing for portamento feel
    S['freq']  = S['freq']  * 0.80 + freq_t * 0.20
    S['vol']   = S['vol']   * sp   + vol_t  * (1-sp)
    S['bri']   = S['bri']   * sp   + bri_t  * (1-sp)
    S['wrm']   = S['wrm']   * sp   + wrm_t  * (1-sp)
    S['vib']   = S['vib']   * sp   + vib_t  * (1-sp)
    S['tre']   = S['tre']   * sp   + tre_t  * (1-sp)
    S['rev']   = S['rev']   * sp   + rev_t  * (1-sp)

    # ── fade in/out when hand appears/disappears ───────────────────
    fade_tgt = 1.0 if active else 0.0
    S['fade'] = S['fade'] * 0.50 + fade_tgt * 0.50   # fast fade (per chunk)
    if S['fade'] < 0.005 and not active:
        return np.zeros(CHUNK, dtype=np.float32)

    freq = max(S['freq'], 20.0)
    bri  = S['bri'];  wrm = S['wrm']

    # ── vibrato LFO ────────────────────────────────────────────────
    vib_ph = S['vib_ph'] + 2*np.pi * 5.3 * np.arange(CHUNK) / SR
    vib    = np.sin(vib_ph)
    S['vib_ph'] = float(vib_ph[-1]) % (2*np.pi)
    freq_arr = freq * (1.0 + S['vib'] * 0.016 * vib)

    # ── phase accumulator ──────────────────────────────────────────
    inc    = 2*np.pi * freq_arr / SR
    phases = S['phase'] + np.cumsum(inc)
    S['phase'] = float(phases[-1]) % (2*np.pi)

    # ── additive harmonics ─────────────────────────────────────────
    w  =              np.sin(phases)
    w += bri * 0.55 * np.sin(2*phases)
    w += bri * 0.30 * np.sin(3*phases)
    w += bri * 0.16 * np.sin(4*phases)
    w += bri * 0.08 * np.sin(5*phases)
    w += wrm * 0.22 * np.sin(0.5*phases)
    w += wrm * 0.11 * np.sin(1.5*phases)

    # mode colour
    mode = S['mode']
    if mode == "guitar":
        w += bri * 0.02 * np.random.randn(CHUNK).astype(np.float32)
    elif mode == "piano":
        w += bri * 0.06 * np.sin(phases * 2.0015)
        w += bri * 0.03 * np.sin(phases * 3.005)
    elif mode == "harp":
        dec = np.exp(-np.arange(CHUNK) / (SR*0.3))
        w  += 0.10 * np.sin(6*phases) * dec
    elif mode == "bass":
        w = np.tanh(w * (1.4 + wrm*1.4)) / (1.4 + wrm*0.5)

    w /= (1 + bri*1.05 + wrm*0.33 + 0.001)

    # ── body resonance (additive peaks, not serial replacement) ───
    res = np.zeros(CHUNK, dtype=np.float32)
    for filt in S['filters']:
        b,a,zi = filt
        r, zi_new = signal.lfilter(b, a, w, zi=zi)
        filt[2] = zi_new
        res += r
    w = w + res * 0.35   # blend resonance peaks into dry signal

    # ── tremolo ────────────────────────────────────────────────────
    tre_ph = S['tre_ph'] + 2*np.pi * 4.8 * np.arange(CHUNK) / SR
    trem   = 1.0 - S['tre'] * 0.28 * (0.5 + 0.5*np.sin(tre_ph))
    S['tre_ph'] = float(tre_ph[-1]) % (2*np.pi)
    w *= trem

    # ── comb reverb ────────────────────────────────────────────────
    delay = int(SR * 0.067)
    rbuf  = S['rev_buf']
    ridx  = S['rev_idx']
    wet   = np.empty(CHUNK, dtype=np.float32)
    for k in range(CHUNK):
        ri       = (ridx - delay) % len(rbuf)
        wet[k]   = rbuf[ri]
        rbuf[ridx] = w[k] + wet[k] * 0.52
        ridx     = (ridx+1) % len(rbuf)
    S['rev_idx'] = ridx
    rv = S['rev']
    w  = w*(1-rv) + wet*rv

    # ── volume + fade ──────────────────────────────────────────────
    w *= S['vol'] * S['fade']
    return w.astype(np.float32)


def to_sound(w: np.ndarray) -> pygame.mixer.Sound:
    pcm = np.clip(w*32767, -32767, 32767).astype(np.int16)
    return pygame.sndarray.make_sound(np.column_stack([pcm,pcm]))


# ── audio thread ──────────────────────────────────────────────────
class AudioThread(threading.Thread):
    def __init__(self, st: State, ch_idx: int, mode_ref: list):
        super().__init__(daemon=True)
        self.st       = st
        self.ch_idx   = ch_idx
        self.mode_ref = mode_ref
        self._stop    = threading.Event()

    def run(self):
        ch   = pygame.mixer.Channel(self.ch_idx)
        mode = self.mode_ref[0]
        syn  = dict(
            freq=261.63, vol=0.0, bri=0.5, wrm=0.3,
            vib=0.0, tre=0.0, rev=0.18, fade=0.0,
            phase=0.0, vib_ph=0.0, tre_ph=0.0,
            filters=make_filters(mode),
            rev_buf=np.zeros(int(SR*0.12), dtype=np.float32),
            rev_idx=0, mode=mode,
        )

        while not self._stop.is_set():
            # mode change
            new_mode = self.mode_ref[0]
            if new_mode != syn['mode']:
                syn['mode']    = new_mode
                syn['filters'] = make_filters(new_mode)
                syn['rev_buf'][:] = 0.0

            chunk = synth_chunk(self.st, syn)
            snd   = to_sound(chunk)

            # ── reliable play/queue pattern ────────────────────────
            if not ch.get_busy():
                ch.play(snd)
            else:
                # poll for queue slot — max wait = one chunk duration
                t_end = time.perf_counter() + CHUNK_S
                while time.perf_counter() < t_end:
                    if ch.get_queue() is None:
                        ch.queue(snd)
                        break
                    time.sleep(0.004)
                else:
                    # queue never freed — restart channel
                    ch.stop()
                    ch.play(snd)

            # sleep until ~halfway through current chunk
            time.sleep(CHUNK_S * 0.55)

    def stop(self):
        self._stop.set()
        pygame.mixer.Channel(self.ch_idx).fadeout(300)


# ── hand landmark helpers ─────────────────────────────────────────
FINGER_IDS = {"thumb":[1,2,3,4],"index":[5,6,7,8],
              "middle":[9,10,11,12],"ring":[13,14,15,16],"pinky":[17,18,19,20]}
MCP_IDS    = [5,9,13,17]
TIP_IDS    = [4,8,12,16,20]

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
    spread = [_a3(pts[MCP_IDS[i]],pts[0],pts[MCP_IDS[i+1]])
              for i in range(len(MCP_IDS)-1)]
    d    = pts[9]-pts[0]
    tilt = math.atan2(-d[1], d[0])
    bend = math.atan2(-d[2], math.sqrt(d[0]**2+d[1]**2))
    tips = np.array([pts[t,:2] for t in TIP_IDS])
    open_= float(np.mean(np.linalg.norm(tips-pts[0,:2],axis=1)))/fw
    return dict(joint=joint, spread=spread, tilt=tilt,
                bend=bend, height=1-(pts[0,1]/fh), openness=open_,
                wxy=pts[0,:2].copy())

def to_params(ang, scale):
    j = ang['joint']
    def ext(n):
        return float(np.mean(j[n]))/math.pi if j[n] else 0.5

    pn   = np.clip(ext("index")*0.35 + ext("middle")*0.30 +
                   ext("ring")*0.20  + ext("pinky")*0.10  +
                   ext("thumb")*0.05, 0, 1)
    sf   = pn*(len(scale)-1)
    lo   = int(sf); hi = min(lo+1,len(scale)-1)
    freq = m2f(scale[lo])*(1-(sf-lo)) + m2f(scale[hi])*(sf-lo)

    # wrist tilt → continuous octave bend
    freq *= 2.0 ** (np.clip(ang['tilt']/(math.pi*0.55),-1,1) * 0.85)

    # thumb-index micro expression
    if j['thumb'] and j['index']:
        freq *= 2**((j['thumb'][0]-j['index'][0])/math.pi * 0.025)

    vol    = float(np.clip(0.3 + ang['height']*0.70, 0.05, 1.0))
    bright = float(np.clip(ang['openness']*2.8*0.55 +
                           (sum(j['index'])+sum(j['middle']))/(4*math.pi)*0.45, 0,1))
    warm   = float(np.clip((1-ext("ring"))*0.5+(1-ext("pinky"))*0.5, 0,1))
    vib    = float(np.clip((ang['spread'][0]-0.20)/0.45,0,1)*0.65) if ang['spread'] else 0
    trem   = float(np.clip((ang['spread'][-1]-0.15)/0.40,0,1)*0.45) if ang['spread'] else 0
    rev    = float(np.clip(0.08 + ang['bend']*0.55, 0.05, 0.55))
    return dict(freq=freq,vol=vol,bright=bright,warm=warm,
                vib=vib,trem=trem,rev=rev)

# ── visuals ───────────────────────────────────────────────────────
_CONN = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
         (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),
         (13,17),(0,17),(17,18),(18,19),(19,20)]

def draw_hand(frame, hand, color, label=""):
    h,w = frame.shape[:2]; lms = _lms(hand)
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

def draw_ui(frame, mode, st: State, syn: dict, fps):
    h,w=frame.shape[:2]; color=PALETTE[mode]
    ov=frame.copy()
    cv2.rectangle(ov,(w-225,0),(w,235),(12,12,12),-1)
    cv2.addWeighted(ov,0.65,frame,0.35,0,frame)

    def T(s,x,y,sc=0.54,c=None,bold=False):
        c=c or color
        f=cv2.FONT_HERSHEY_DUPLEX if bold else cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame,s,(x,y),f,sc,(0,0,0),3,cv2.LINE_AA)
        cv2.putText(frame,s,(x,y),f,sc,c,1,cv2.LINE_AA)

    fm   = syn.get('freq',261)
    note = int(np.clip(round(69+12*math.log2(max(fm,20)/440)),21,108))
    vol  = syn.get('vol',0)
    T(f"[ {mode.upper()} ]",w-218,32,0.64,bold=True)
    T(f"NOTE  {midi_name(note)}",w-218,60)
    T(f"FREQ  {fm:.0f} Hz",w-218,84)
    T(f"VOL   {int(vol*100):3d}%",w-218,108)
    T(f"FPS   {fps:.0f}",w-218,132)

    # param bars
    bars=[("BRI",syn.get('bri',0)),("WRM",syn.get('wrm',0)),
          ("VIB",syn.get('vib',0)),("TRE",syn.get('tre',0)),
          ("REV",syn.get('rev',0))]
    bw,bh,gap=18,55,6
    for i,(nm,val) in enumerate(bars):
        bx=w-218+i*(bw+gap); fld=int(val*bh)
        cv2.rectangle(frame,(bx,148),(bx+bw,148+bh),(35,35,35),-1)
        if fld>0: cv2.rectangle(frame,(bx,148+bh-fld),(bx+bw,148+bh),color,-1)
        cv2.putText(frame,nm,(bx-1,148+bh+13),cv2.FONT_HERSHEY_SIMPLEX,
                    0.30,(150,150,150),1,cv2.LINE_AA)

    cv2.rectangle(frame,(0,h-32),(w,h),(12,12,12),-1)
    T("[G]Guitar [P]Piano [H]Harp [B]Bass  [Q]Quit",10,h-11,0.42,(160,160,160))

    bx2,by2,bw2=10,h-58,220
    cv2.rectangle(frame,(bx2,by2),(bx2+bw2,by2+9),(40,40,40),-1)
    fv=int(bw2*np.clip(vol,0,1))
    if fv>0: cv2.rectangle(frame,(bx2,by2),(bx2+fv,by2+9),color,-1)
    T("VOL",bx2,by2-5,0.36)
    norm=np.clip((fm-55)/900,0,1)
    cv2.rectangle(frame,(bx2,by2-20),(bx2+bw2,by2-12),(40,40,40),-1)
    fp2=int(bw2*norm)
    if fp2>0: cv2.rectangle(frame,(bx2,by2-20),(bx2+fp2,by2-12),
                             tuple(c//2 for c in color),-1)
    T("PITCH",bx2,by2-23,0.36)

# ── mediapipe model download ──────────────────────────────────────
MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
              "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "hand_landmarker.task")
def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand model (~7 MB)…",flush=True)
        urllib.request.urlretrieve(MODEL_URL,MODEL_PATH)
        print("Done.")
    return MODEL_PATH

# ── main ──────────────────────────────────────────────────────────
def main():
    pygame.mixer.pre_init(frequency=SR,size=-16,channels=2,buffer=1024)
    pygame.init()
    pygame.mixer.set_num_channels(8)

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

    mode_ref = ["guitar"]

    # Start audio threads — one per hand channel
    threads = [AudioThread(_state[i], i, mode_ref) for i in range(2)]
    for t in threads: t.start()

    # Keep a ref to thread[0]'s syn dict for the UI display
    # We expose it by peeking at the thread after first chunk
    ui_syn = dict(freq=261.0,vol=0.0,bri=0.5,wrm=0.3,
                  vib=0.0,tre=0.0,rev=0.18)

    fps_q     = deque(maxlen=20)
    t0        = time.perf_counter()
    prev_wxy  = [None, None]
    prev_t    = [time.perf_counter()]*2

    print("\n  Air Instrument v5  —  LIVE")
    print("  G=Guitar  P=Piano  H=Harp  B=Bass  Q=Quit\n")

    while True:
        ok,frame=cap.read()
        if not ok: break
        frame=cv2.flip(frame,1)
        fh,fw=frame.shape[:2]
        hands=detect(frame)

        now=time.perf_counter()
        fps_q.append(1.0/max(now-t0,1e-6)); t0=now
        fps=float(np.mean(fps_q))

        active_set=set()
        for i,hand in enumerate(hands[:2]):
            ang = extract(hand,fw,fh)
            p   = to_params(ang, SCALES[mode_ref[0]])
            _state[i].write(freq=p['freq'],vol=p['vol'],
                            bright=p['bright'],warm=p['warm'],
                            vib=p['vib'],trem=p['trem'],rev=p['rev'],
                            active=True)
            prev_wxy[i]=ang['wxy'].copy(); prev_t[i]=now
            active_set.add(i)

            if i==0:
                note=int(np.clip(round(69+12*math.log2(max(p['freq'],20)/440)),21,108))
                ui_syn.update(freq=p['freq'],vol=p['vol'],bri=p['bright'],
                              wrm=p['warm'],vib=p['vib'],tre=p['trem'],rev=p['rev'])
                draw_hand(frame,hand,PALETTE[mode_ref[0]],midi_name(note))
            else:
                draw_hand(frame,hand,PALETTE[mode_ref[0]])

        for i in range(2):
            if i not in active_set:
                _state[i].write(active=False)
                prev_wxy[i]=None

        draw_ui(frame,mode_ref[0],_state[0],ui_syn,fps)
        cv2.imshow("Air Instrument v5",frame)

        key=cv2.waitKey(1)&0xFF
        if key==ord('q'): break
        elif key in (ord('g'),ord('p'),ord('h'),ord('b')):
            mode_ref[0]={'g':'guitar','p':'piano','h':'harp','b':'bass'}[chr(key)]
            print(f"  → {mode_ref[0].upper()}")

    for t in threads: t.stop()
    cap.release(); cv2.destroyAllWindows(); pygame.quit()
    print("Bye!")

if __name__=="__main__":
    main()