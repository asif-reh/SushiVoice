#!/usr/bin/env python3
"""
Generate synthetic training data for 20 sushi items
Creates TTS audio samples to supplement human voice recordings

Usage:
    python3 generate_training_data_20.py
"""

import os
import random
from pathlib import Path
from tqdm import tqdm

try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    print("❌ pyttsx3 not installed. Install with: pip install pyttsx3")
    TTS_AVAILABLE = False
    exit(1)


# Load 20 sushi items
VOCAB_FILE = Path("data/sushi_items_20.txt")
OUTPUT_DIR = Path("data/synthetic_20")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Command templates
WAKE_WORDS = ["hey yosushi", "yosushi", "yo sushi"]
ACTIONS = ["print", "label", "make"]
QUANTITIES = ["one", "two", "three", "four", "five", "six", "seven", "eight", 
              "nine", "ten", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
FILLERS = ["", "please", "can you"]


def load_sushi_items():
    """Load 20 sushi items from file"""
    with open(VOCAB_FILE, 'r') as f:
        items = [line.strip() for line in f if line.strip()]
    return items


def generate_commands(sushi_items, samples_per_item=20):
    """
    Generate command variations for each item
    
    Args:
        sushi_items: List of sushi item names
        samples_per_item: Number of samples to generate per item
    
    Returns:
        List of (text, filename) tuples
    """
    commands = []
    
    for idx, item in enumerate(sushi_items):
        for sample_num in range(samples_per_item):
            # Random components
            wake = random.choice(WAKE_WORDS)
            action = random.choice(ACTIONS)
            qty = random.choice(QUANTITIES)
            filler = random.choice(FILLERS)
            
            # Generate command variations
            if random.random() < 0.3 and filler:
                # "hey yosushi please print five chicken teriyaki"
                text = f"{wake} {filler} {action} {qty} {item.lower()}"
            else:
                # "yosushi print five chicken teriyaki"
                text = f"{wake} {action} {qty} {item.lower()}"
            
            # Clean up spaces
            text = ' '.join(text.split())
            
            # Filename
            filename = f"synth_{idx:02d}_{sample_num:02d}_{item.replace(' ', '_').lower()}.wav"
            
            commands.append((text, filename))
    
    return commands


def synthesize_audio(text, output_path):
    """Generate audio using pyttsx3"""
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)  # Speed
    
    # Try to use different voices if available
    voices = engine.getProperty('voices')
    if len(voices) > 1:
        engine.setProperty('voice', random.choice(voices).id)
    
    engine.save_to_file(text, str(output_path))
    engine.runAndWait()


def main():
    """Generate synthetic training data"""
    
    print("\n" + "="*60)
    print("🎤 GENERATE SYNTHETIC TRAINING DATA")
    print("="*60)
    
    # Load sushi items
    sushi_items = load_sushi_items()
    print(f"\n📋 Loaded {len(sushi_items)} sushi items")
    
    # Generate commands (20 samples per item = 400 total)
    print(f"🔄 Generating commands (20 samples per item)...")
    commands = generate_commands(sushi_items, samples_per_item=20)
    print(f"✅ Generated {len(commands)} commands")
    
    # Synthesize audio
    print(f"\n🎵 Synthesizing audio with TTS...")
    print(f"📁 Output: {OUTPUT_DIR}")
    
    for text, filename in tqdm(commands, desc="Synthesizing"):
        output_path = OUTPUT_DIR / filename
        try:
            synthesize_audio(text, output_path)
        except Exception as e:
            print(f"\n⚠️  Failed to synthesize '{text}': {e}")
    
    print("\n" + "="*60)
    print("✅ SYNTHETIC DATA GENERATION COMPLETE!")
    print("="*60)
    print(f"\n📊 Statistics:")
    print(f"   Total samples: {len(commands)}")
    print(f"   Sushi items: {len(sushi_items)}")
    print(f"   Samples per item: 20")
    print(f"   Output directory: {OUTPUT_DIR}")
    
    print("\n🎯 NEXT STEP:")
    print("   Record your voice: python3 record_20_human_voice.py")
    print("\n")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Generation cancelled by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
