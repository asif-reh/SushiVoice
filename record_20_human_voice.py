#!/usr/bin/env python3
"""
Record 20 voice samples for 20 sushi items
This will teach the model YOUR voice for the sushi vocabulary!

Usage:
    python3 record_20_human_voice.py
"""

import os
import numpy as np
import pyaudio
import wave
from pathlib import Path

# Audio settings
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK = 1024
RECORD_SECONDS = 5  # 5 seconds per sample

# Output directory
OUTPUT_DIR = Path("data/my_voice_20")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 20 sushi items
SUSHI_ITEMS = [
    "Chicken Teriyaki",
    "Salmon Nigiri",
    "Tuna Nigiri",
    "California Roll",
    "Spicy Tuna Roll",
    "Eel Avocado Roll",
    "Shrimp Tempura Roll",
    "Dragon Roll",
    "Rainbow Roll",
    "Philadelphia Roll",
    "Salmon Sashimi",
    "Tuna Sashimi",
    "Edamame",
    "Miso Soup",
    "Gyoza",
    "Chicken Katsu",
    "Tempura Shrimp",
    "Vegetable Roll",
    "Avocado Roll",
    "Cucumber Roll",
]


def record_audio(duration=5):
    """Record audio from microphone"""
    p = pyaudio.PyAudio()
    
    print("🎙️  Recording... SPEAK NOW!")
    
    stream = p.open(
        format=pyaudio.paInt16,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK
    )
    
    frames = []
    for i in range(0, int(SAMPLE_RATE / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    print("✅ Recording complete!")
    
    return b''.join(frames)


def save_wav(filename, audio_data):
    """Save audio data as WAV file"""
    with wave.open(str(filename), 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_data)


def main():
    """Record 20 samples for 20 sushi items"""
    
    print("\n" + "="*60)
    print("🍣 SUSHI VOICE - RECORD YOUR VOICE")
    print("="*60)
    print(f"\nWe'll record 20 samples - one for each sushi item.")
    print(f"Each recording is {RECORD_SECONDS} seconds long.")
    print(f"\nTotal time: ~{20 * (RECORD_SECONDS + 5) / 60:.0f} minutes")
    print("\n💡 INSTRUCTIONS:")
    print("   Say: 'print five [item name]'")
    print("   Example: 'print five chicken teriyaki'")
    print("   Speak clearly and naturally!")
    print("\nPress ENTER when ready to start...")
    input()
    
    # Record each item
    for idx, item in enumerate(SUSHI_ITEMS, 1):
        print("\n" + "-"*60)
        print(f"📍 Sample {idx}/20")
        print(f"🍱 Item: {item}")
        print(f"\n🗣️  Say: 'print five {item.lower()}'")
        print("\nPress ENTER when ready to record...")
        input()
        
        # Record
        audio_data = record_audio(RECORD_SECONDS)
        
        # Save
        filename = OUTPUT_DIR / f"sample_{idx:02d}_{item.replace(' ', '_').lower()}.wav"
        save_wav(filename, audio_data)
        print(f"💾 Saved: {filename.name}")
        
        # Break every 5 samples
        if idx % 5 == 0 and idx < 20:
            print("\n" + "="*60)
            print(f"🎉 {idx} samples done! Take a 30-second break.")
            print("="*60)
            print("\nPress ENTER to continue...")
            input()
    
    print("\n" + "="*60)
    print("✅ ALL 20 SAMPLES RECORDED!")
    print("="*60)
    print(f"\n📁 Samples saved to: {OUTPUT_DIR}")
    print(f"📊 Total files: {len(list(OUTPUT_DIR.glob('*.wav')))}")
    
    print("\n🎯 NEXT STEPS:")
    print("   1. Generate synthetic data: python3 generate_training_data_20.py")
    print("   2. Train the model: python3 train_transformer_20.py")
    print("   3. Test it: python3 demo_transformer_voice.py")
    print("\n🚀 Let's train your model!\n")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Recording cancelled by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
