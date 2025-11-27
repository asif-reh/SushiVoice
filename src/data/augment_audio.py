#!/usr/bin/env python3
"""
Audio Augmentation for SushiVoice
Expands synthetic dataset with noise, speed/pitch variations, and reverb
"""

import json
import os
import random
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm
import argparse


class AudioAugmenter:
    def __init__(self, manifest_path: str, output_dir: str):
        """
        Initialize audio augmenter
        
        Args:
            manifest_path: Path to manifest.jsonl
            output_dir: Output directory for augmented audio
        """
        self.manifest_path = Path(manifest_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load manifest
        self.manifest_entries = []
        with open(self.manifest_path, 'r') as f:
            for line in f:
                self.manifest_entries.append(json.loads(line))
        
        self.augmented_manifest_path = self.manifest_path.parent / 'manifest_augmented.jsonl'
    
    def add_gaussian_noise(self, audio: np.ndarray, noise_level: float = 0.005) -> np.ndarray:
        """Add Gaussian noise to audio"""
        noise = np.random.normal(0, noise_level, audio.shape)
        return audio + noise
    
    def add_kitchen_noise(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Simulate kitchen noise (clanking, background chatter)
        Uses colored noise as approximation
        """
        # Generate pink noise (1/f noise, common in kitchen environments)
        duration = len(audio) / sr
        samples = int(sr * duration)
        
        # Create pink noise using FFT
        white_noise = np.random.randn(samples)
        fft_noise = np.fft.rfft(white_noise)
        
        # Apply 1/f filter
        freqs = np.fft.rfftfreq(samples, 1/sr)
        freqs[0] = 1  # Avoid division by zero
        pink_filter = 1 / np.sqrt(freqs)
        fft_noise *= pink_filter
        
        pink_noise = np.fft.irfft(fft_noise)
        pink_noise = pink_noise[:len(audio)]
        
        # Normalize and mix
        pink_noise = pink_noise / np.max(np.abs(pink_noise)) * 0.1
        
        return audio + pink_noise
    
    def change_speed(self, audio: np.ndarray, sr: int, speed_factor: float) -> np.ndarray:
        """Change audio speed (0.9-1.1x)"""
        return librosa.effects.time_stretch(audio, rate=speed_factor)
    
    def change_pitch(self, audio: np.ndarray, sr: int, n_steps: float) -> np.ndarray:
        """Shift pitch by n_steps semitones"""
        return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
    
    def add_reverb(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Add simple room reverb effect
        Uses convolution with exponential decay
        """
        # Create simple impulse response (exponential decay)
        reverb_duration = 0.3  # seconds
        reverb_samples = int(sr * reverb_duration)
        
        decay = np.exp(-np.linspace(0, 5, reverb_samples))
        impulse_response = decay * np.random.randn(reverb_samples) * 0.1
        
        # Convolve audio with impulse response
        reverb_audio = np.convolve(audio, impulse_response, mode='same')
        
        # Mix dry and wet signals
        return 0.7 * audio + 0.3 * reverb_audio
    
    def augment_sample(self, audio_path: str, augmentation_type: str) -> tuple:
        """
        Apply augmentation to a single audio file
        
        Returns:
            (augmented_audio, sample_rate)
        """
        # Load audio
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        
        # Apply augmentation
        if augmentation_type == 'noise_gaussian':
            audio = self.add_gaussian_noise(audio, noise_level=0.005)
        elif augmentation_type == 'noise_kitchen':
            audio = self.add_kitchen_noise(audio, sr)
        elif augmentation_type == 'speed_up':
            audio = self.change_speed(audio, sr, speed_factor=1.1)
        elif augmentation_type == 'speed_down':
            audio = self.change_speed(audio, sr, speed_factor=0.9)
        elif augmentation_type == 'pitch_up':
            audio = self.change_pitch(audio, sr, n_steps=2)
        elif augmentation_type == 'pitch_down':
            audio = self.change_pitch(audio, sr, n_steps=-2)
        elif augmentation_type == 'reverb':
            audio = self.add_reverb(audio, sr)
        elif augmentation_type == 'combined':
            # Combine multiple augmentations
            audio = self.add_gaussian_noise(audio, noise_level=0.003)
            audio = self.change_speed(audio, sr, speed_factor=random.uniform(0.95, 1.05))
            if random.random() < 0.5:
                audio = self.add_kitchen_noise(audio, sr)
        
        # Normalize
        audio = audio / np.max(np.abs(audio))
        
        return audio, sr
    
    def augment_dataset(self, augmentations_per_sample: int = 2):
        """
        Augment entire dataset
        
        Args:
            augmentations_per_sample: Number of augmented versions per original sample
        """
        print(f"Augmenting dataset with {augmentations_per_sample} variations per sample...")
        
        augmentation_types = [
            'noise_gaussian',
            'noise_kitchen',
            'speed_up',
            'speed_down',
            'pitch_up',
            'pitch_down',
            'reverb',
            'combined'
        ]
        
        augmented_entries = []
        
        # Add original samples
        augmented_entries.extend(self.manifest_entries)
        
        # Augment each sample
        for entry in tqdm(self.manifest_entries, desc="Augmenting audio"):
            # Get original audio path (relative to project root)
            original_audio_path = Path(entry['audio'])
            
            # Check if file exists (handle both absolute and relative paths)
            if not original_audio_path.exists():
                # Try relative to manifest directory
                original_audio_path = self.manifest_path.parent.parent / entry['audio']
            
            if not original_audio_path.exists():
                print(f"Warning: Audio file not found: {entry['audio']}")
                continue
            
            # Generate augmented versions
            for aug_idx in range(augmentations_per_sample):
                aug_type = random.choice(augmentation_types)
                
                # Augment audio
                try:
                    augmented_audio, sr = self.augment_sample(str(original_audio_path), aug_type)
                except Exception as e:
                    print(f"Error augmenting {original_audio_path}: {e}")
                    continue
                
                # Save augmented audio
                base_name = original_audio_path.stem
                aug_filename = f"{base_name}_aug{aug_idx}_{aug_type}.wav"
                aug_path = self.output_dir / aug_filename
                
                sf.write(str(aug_path), augmented_audio, sr)
                
                # Compute duration
                duration = len(augmented_audio) / sr
                
                # Add to manifest
                augmented_entry = {
                    'audio': str(aug_path.relative_to(aug_path.parents[2])),
                    'text': entry['text'],
                    'duration': round(duration, 2),
                    'augmentation': aug_type
                }
                augmented_entries.append(augmented_entry)
        
        # Update durations for original samples
        for entry in augmented_entries:
            if entry.get('duration') is None:
                # Compute duration for original samples
                audio_path = self.manifest_path.parent.parent / entry['audio']
                if audio_path.exists():
                    audio, sr = librosa.load(str(audio_path), sr=None)
                    entry['duration'] = round(len(audio) / sr, 2)
        
        # Write augmented manifest
        with open(self.augmented_manifest_path, 'w') as f:
            for entry in augmented_entries:
                f.write(json.dumps(entry) + '\n')
        
        print(f"\n✅ Augmentation complete!")
        print(f"   Original samples: {len(self.manifest_entries)}")
        print(f"   Augmented samples: {len(augmented_entries) - len(self.manifest_entries)}")
        print(f"   Total samples: {len(augmented_entries)}")
        print(f"   Augmented audio: {self.output_dir}")
        print(f"   Augmented manifest: {self.augmented_manifest_path}")
        
        # Print augmentation breakdown
        aug_counts = {}
        for entry in augmented_entries:
            aug_type = entry.get('augmentation', 'original')
            aug_counts[aug_type] = aug_counts.get(aug_type, 0) + 1
        
        print("\n📊 Augmentation Breakdown:")
        for aug_type, count in sorted(aug_counts.items()):
            print(f"   {aug_type}: {count}")


def main():
    parser = argparse.ArgumentParser(description='Augment SushiVoice dataset')
    parser.add_argument('--manifest', type=str, default='data/manifest.jsonl',
                        help='Path to manifest file')
    parser.add_argument('--output', type=str, default='data/augmented',
                        help='Output directory for augmented audio')
    parser.add_argument('--aug_per_sample', type=int, default=2,
                        help='Number of augmentations per sample (default: 2)')
    
    args = parser.parse_args()
    
    # Check if manifest exists
    if not os.path.exists(args.manifest):
        print(f"❌ Manifest file not found: {args.manifest}")
        print("   Run generate_dataset.py first to create the dataset")
        return
    
    # Augment dataset
    augmenter = AudioAugmenter(args.manifest, args.output)
    augmenter.augment_dataset(augmentations_per_sample=args.aug_per_sample)
    
    print("\n🎉 Dataset augmentation complete!")
    print("\n💡 Next steps:")
    print("   1. Train model: python src/models/train_custom_slm.py --manifest data/manifest_augmented.jsonl")
    print("   2. Or add real recordings to data/raw/ and update manifest")


if __name__ == '__main__':
    main()
