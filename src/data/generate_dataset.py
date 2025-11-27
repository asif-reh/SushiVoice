#!/usr/bin/env python3
"""
Synthetic Dataset Generator for SushiVoice
Generates audio-text pairs using TTS for training the custom ASR model
"""

import json
import os
import random
from pathlib import Path
from typing import List, Dict
import pyttsx3
from gtts import gTTS
from tqdm import tqdm
import argparse


class SushiDatasetGenerator:
    def __init__(self, vocab_path: str, output_dir: str):
        """
        Initialize dataset generator
        
        Args:
            vocab_path: Path to sushi_vocab.json
            output_dir: Output directory for synthetic audio
        """
        with open(vocab_path, 'r') as f:
            self.vocab = json.load(f)
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.manifest_path = self.output_dir.parent / 'manifest.jsonl'
        
        # Initialize pyttsx3 for offline TTS
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 150)  # Speed
        
    def generate_command_templates(self, num_samples: int = 1000) -> List[Dict[str, str]]:
        """
        Generate varied command templates
        
        Returns:
            List of command dictionaries with 'text' key
        """
        commands = []
        
        for _ in range(num_samples):
            # Randomly select components
            wake_word = random.choice(self.vocab['wake_words'])
            action = random.choice(self.vocab['actions'])
            
            # Quantity (50% numeric, 50% word form)
            if random.random() < 0.5:
                qty = random.choice(self.vocab['quantities_numeric'])
            else:
                qty = random.choice(self.vocab['quantities_word'])
            
            qty_phrase = random.choice(self.vocab['quantity_phrases'])
            label_connector = random.choice(self.vocab['label_connectors'])
            item = random.choice(self.vocab['sushi_items'])
            
            # Optional fillers (30% chance)
            filler = random.choice(self.vocab['fillers']) if random.random() < 0.3 else ""
            
            # Build command with variations
            template_type = random.choice(['full', 'short', 'medium'])
            
            if template_type == 'full':
                # "Hey YoSushi please print 5 times of label of Chicken Teriyaki"
                text = f"{wake_word} {filler} {action} {qty} {qty_phrase} {label_connector} {item}"
            elif template_type == 'short':
                # "YoSushi label 5 Chicken Teriyaki"
                text = f"{wake_word} {action} {qty} {item}"
            else:  # medium
                # "Hey YoSushi print 5 labels of Chicken Teriyaki"
                text = f"{wake_word} {action} {qty} {label_connector} {item}"
            
            # Clean up extra spaces
            text = ' '.join(text.split()).lower()
            
            commands.append({'text': text})
        
        return commands
    
    def synthesize_audio_pyttsx3(self, text: str, output_path: str):
        """
        Generate audio using pyttsx3 (offline, multiple voices)
        
        Args:
            text: Text to synthesize
            output_path: Output audio file path
        """
        # Randomly select voice (if available)
        voices = self.tts_engine.getProperty('voices')
        if len(voices) > 1:
            self.tts_engine.setProperty('voice', random.choice(voices).id)
        
        self.tts_engine.save_to_file(text, output_path)
        self.tts_engine.runAndWait()
    
    def synthesize_audio_gtts(self, text: str, output_path: str):
        """
        Generate audio using gTTS (online, better quality)
        
        Args:
            text: Text to synthesize
            output_path: Output audio file path
        """
        # Try multiple accents
        tld = random.choice(['com', 'co.uk', 'com.au', 'ca'])
        
        try:
            tts = gTTS(text=text, lang='en', tld=tld, slow=False)
            tts.save(output_path)
        except Exception as e:
            print(f"gTTS failed for '{text}': {e}. Falling back to pyttsx3.")
            self.synthesize_audio_pyttsx3(text, output_path)
    
    def generate_dataset(self, num_samples: int = 1000, use_gtts: bool = True):
        """
        Generate complete synthetic dataset
        
        Args:
            num_samples: Number of samples to generate
            use_gtts: Use gTTS (online) if True, pyttsx3 (offline) if False
        """
        print(f"Generating {num_samples} synthetic samples...")
        
        # Generate command templates
        commands = self.generate_command_templates(num_samples)
        
        # Open manifest file
        manifest_entries = []
        
        for idx, command in enumerate(tqdm(commands, desc="Synthesizing audio")):
            text = command['text']
            audio_filename = f"sample_{idx:05d}.wav"
            audio_path = self.output_dir / audio_filename
            
            # Synthesize audio
            if use_gtts:
                self.synthesize_audio_gtts(text, str(audio_path))
            else:
                self.synthesize_audio_pyttsx3(text, str(audio_path))
            
            # Add to manifest
            # Note: We'll compute duration later during augmentation/preprocessing
            manifest_entry = {
                'audio': str(audio_path.relative_to(audio_path.parents[2])),  # Relative to project root
                'text': text,
                'duration': None  # Computed during preprocessing
            }
            manifest_entries.append(manifest_entry)
        
        # Write manifest
        with open(self.manifest_path, 'w') as f:
            for entry in manifest_entries:
                f.write(json.dumps(entry) + '\n')
        
        print(f"\n✅ Generated {len(manifest_entries)} samples")
        print(f"   Audio files: {self.output_dir}")
        print(f"   Manifest: {self.manifest_path}")
        
        # Print statistics
        print("\n📊 Dataset Statistics:")
        unique_items = set(cmd['text'].split()[-2:] for cmd in commands)
        print(f"   Total samples: {len(commands)}")
        print(f"   Unique items covered: ~{len(unique_items)}")
        print(f"   Average text length: {sum(len(c['text']) for c in commands) / len(commands):.1f} chars")


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic SushiVoice dataset')
    parser.add_argument('--vocab', type=str, default='src/data/sushi_vocab.json',
                        help='Path to vocabulary JSON file')
    parser.add_argument('--output', type=str, default='data/synthetic',
                        help='Output directory for synthetic audio')
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='Number of samples to generate (default: 1000)')
    parser.add_argument('--use_pyttsx3', action='store_true',
                        help='Use pyttsx3 instead of gTTS (offline mode)')
    
    args = parser.parse_args()
    
    # Check if vocab file exists
    if not os.path.exists(args.vocab):
        print(f"❌ Vocabulary file not found: {args.vocab}")
        print("   Make sure you're running from the project root directory")
        return
    
    # Generate dataset
    generator = SushiDatasetGenerator(args.vocab, args.output)
    generator.generate_dataset(
        num_samples=args.num_samples,
        use_gtts=not args.use_pyttsx3
    )
    
    print("\n🎉 Dataset generation complete!")
    print("\n💡 Next steps:")
    print("   1. Run augmentation: python src/data/augment_audio.py")
    print("   2. Optionally record real audio and add to manifest")
    print("   3. Train model: python src/models/train_custom_slm.py")


if __name__ == '__main__':
    main()
