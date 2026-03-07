"""
python audio_debug.py
Tests exactly the audio thread logic used in the instrument.
"""
import pygame
import numpy as np
import threading
import time
from scipy import signal

SR     = 44100
CHUNK  = int(SR * 0.18)   # 180ms chunks

pygame.mixer.pre_init(frequency=SR, size=-16, channels=2, buffer=1024)
pygame.init()
pygame.mixer.set_num_channels(8)
print("Mixer init:", pygame.mixer.get_init())

# ── synthesise one chunk of 440Hz sine ──────────────────────────────────────
def make_chunk(freq=440.0, vol=0.6):
    t     = np.linspace(0, CHUNK/SR, CHUNK, endpoint=False)
    w     = np.sin(2*np.pi*freq*t) * vol
    w    += np.sin(2*np.pi*freq*2*t) * vol * 0.4
    pcm   = np.clip(w * 32767, -32767, 32767).astype(np.int16)
    stereo = np.column_stack([pcm, pcm])
    return pygame.sndarray.make_sound(stereo)

# ── TEST A: plain play on Channel(0) ────────────────────────────────────────
print("\n[A] Direct Channel(0).play() — you should hear 440Hz for ~0.5s")
ch = pygame.mixer.Channel(0)
snd = make_chunk(440)
ch.play(snd)
print(f"    busy={ch.get_busy()}")
time.sleep(0.5)
print(f"    busy after 0.5s={ch.get_busy()}")

# ── TEST B: find_channel ────────────────────────────────────────────────────
print("\n[B] find_channel() — you should hear 523Hz for ~0.5s")
ch2 = pygame.mixer.find_channel(True)
print(f"    found channel: {ch2}")
if ch2:
    snd2 = make_chunk(523.25)
    ch2.play(snd2)
    print(f"    busy={ch2.get_busy()}")
    time.sleep(0.5)

# ── TEST C: rolling queue loop (exactly what the instrument does) ────────────
print("\n[C] Rolling queue loop — 3 seconds of gliding pitch (330→660Hz)")
print("    You should hear a continuous rising tone...")

ch3  = pygame.mixer.Channel(0)
stop = threading.Event()
played = [0]

def audio_loop():
    freq = 330.0
    while not stop.is_set():
        snd = make_chunk(freq, vol=0.55)
        freq = min(freq + 15, 660)

        if not ch3.get_busy():
            ch3.play(snd)
            played[0] += 1
            print(f"    play()  freq={freq:.0f}  busy={ch3.get_busy()}")
        else:
            # Wait for queue slot
            t0 = time.perf_counter()
            while ch3.get_queue() is not None:
                if time.perf_counter() - t0 > 0.5:
                    print("    TIMEOUT waiting for queue slot — forcing play()")
                    ch3.play(snd)
                    played[0] += 1
                    break
                time.sleep(0.005)
            else:
                ch3.queue(snd)
                played[0] += 1
                print(f"    queue() freq={freq:.0f}")

        time.sleep(0.18 * 0.6)

t = threading.Thread(target=audio_loop, daemon=True)
t.start()
time.sleep(3.0)
stop.set()
t.join(timeout=1)
print(f"    Total chunks played/queued: {played[0]}")
print(f"    Did you hear a rising tone? (played={played[0]} chunks)")

# ── TEST D: what happens with Channel index vs find_channel ─────────────────
print("\n[D] Checking channel availability...")
for i in range(8):
    ch_i = pygame.mixer.Channel(i)
    print(f"    Channel({i}) busy={ch_i.get_busy()} queue={ch_i.get_queue()}")

pygame.quit()
print("\nDone. Paste full output to diagnose.")