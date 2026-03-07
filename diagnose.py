"""
Run this first: python diagnose.py
It will tell us exactly what's broken with audio.
"""
import pygame
import numpy as np
import time

SR = 44100

print("=" * 50)
print("AUDIO DIAGNOSTIC")
print("=" * 50)

# Test 1: init
print("\n[1] Initialising pygame mixer...")
try:
    pygame.mixer.pre_init(frequency=SR, size=-16, channels=2, buffer=512)
    pygame.init()
    info = pygame.mixer.get_init()
    print(f"    OK: freq={info[0]} size={info[1]} channels={info[2]}")
except Exception as e:
    print(f"    FAILED: {e}")
    exit(1)

# Test 2: make a sound
print("\n[2] Generating 440Hz sine wave...")
try:
    t      = np.linspace(0, 1.0, SR, endpoint=False)
    wave   = (np.sin(2 * np.pi * 440 * t) * 20000).astype(np.int16)
    stereo = np.column_stack([wave, wave])
    snd    = pygame.sndarray.make_sound(stereo)
    print(f"    OK: sound object = {snd}")
except Exception as e:
    print(f"    FAILED: {e}")
    exit(1)

# Test 3: play it
print("\n[3] Playing sound for 1.5 seconds (you should hear a tone)...")
try:
    pygame.mixer.set_num_channels(8)
    ch = pygame.mixer.find_channel()
    if ch is None:
        print("    FAILED: no free channel found")
        exit(1)
    ch.play(snd)
    print(f"    Channel busy: {ch.get_busy()}")
    time.sleep(1.5)
    print(f"    Still busy after 1.5s: {ch.get_busy()}")
    print("    Did you hear a tone? If yes → pygame audio works fine.")
    print("    If no → check Windows sound settings / output device.")
except Exception as e:
    print(f"    FAILED: {e}")
    exit(1)

# Test 4: channel queuing (how the synth engine works)
print("\n[4] Testing channel.queue() for streaming...")
try:
    short = int(SR * 0.05)  # 50ms chunks
    t2    = np.linspace(0, 0.05, short, endpoint=False)
    played = 0
    phase  = 0.0
    ch2    = pygame.mixer.find_channel()
    if ch2 is None:
        print("    FAILED: no free channel")
    else:
        for i in range(20):  # 20 × 50ms = 1 second
            freq   = 523.25 if i % 2 == 0 else 440.0
            wave_c = (np.sin(2*np.pi*freq*t2 + phase) * 18000).astype(np.int16)
            phase += 2*np.pi*freq*0.05
            stereo_c = np.column_stack([wave_c, wave_c])
            snd_c    = pygame.sndarray.make_sound(stereo_c)
            if not ch2.get_busy():
                ch2.play(snd_c)
            else:
                ch2.queue(snd_c)
            played += 1
            time.sleep(0.048)
        print(f"    OK: queued {played} chunks. Did you hear alternating tones?")
except Exception as e:
    print(f"    FAILED: {e}")

# Test 5: tiny chunk streaming (what v3 uses)
print("\n[5] Testing tiny-chunk streaming (512 samples = 11ms)...")
try:
    CHUNK = 512
    ch3   = pygame.mixer.find_channel()
    phase = 0.0
    ok_count = 0
    for i in range(100):  # ~1 second
        freq    = 330.0 + i * 2.0  # gliding pitch
        t_c     = np.arange(CHUNK) / SR
        ph_inc  = 2 * np.pi * freq / SR
        phases  = phase + np.cumsum(np.full(CHUNK, ph_inc))
        phase   = phases[-1]
        w       = (np.sin(phases) * 16000).astype(np.int16)
        s       = np.column_stack([w, w])
        snd_t   = pygame.sndarray.make_sound(s)
        if ch3 and not ch3.get_busy():
            ch3.play(snd_t)
            ok_count += 1
        elif ch3:
            ch3.queue(snd_t)
            ok_count += 1
        time.sleep(CHUNK / SR * 0.8)
    print(f"    OK: streamed {ok_count} tiny chunks. Did you hear a rising tone?")
except Exception as e:
    print(f"    FAILED: {e}")

pygame.quit()
print("\n" + "="*50)
print("Paste the output above so we can fix the instrument.")
print("="*50)