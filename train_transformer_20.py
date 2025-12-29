#!/usr/bin/env python3
"""
Train Transformer model on 20 sushi items with human + synthetic voice

This combines:
- 20 human voice samples (your recordings)
- 400 synthetic TTS samples (robot voice)
Total: 420 samples to train the model

Usage:
    python3 train_transformer_20.py --epochs 30
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torchaudio

# Setup path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from transformer_audio_slm import TransformerAudioSLM, create_character_vocab


# Load 20 sushi items
VOCAB_FILE = Path("data/sushi_items_20.txt")


def load_sushi_items():
    """Load 20 sushi items"""
    with open(VOCAB_FILE, 'r') as f:
        return [line.strip() for line in f if line.strip()]


class SushiVoiceDataset(Dataset):
    """Dataset for Sushi Voice audio files"""
    
    def __init__(self, audio_dir, target_length=96000):
        """
        Args:
            audio_dir: Directory containing .wav files
            target_length: Target audio length (samples) = 6 seconds at 16kHz
        """
        self.audio_dir = Path(audio_dir)
        self.target_length = target_length
        
        # Get all wav files
        self.audio_files = sorted(list(self.audio_dir.glob('*.wav')))
        print(f"   Found {len(self.audio_files)} audio files in {audio_dir}")
        
        # Create character vocab
        self.vocab, self.char_to_idx = create_character_vocab()
        
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        """
        Returns:
            audio: Tensor of shape (target_length,)
            text_encoded: Tensor of character indices
        """
        audio_path = self.audio_files[idx]
        
        # Load audio
        waveform, sr = torchaudio.load(audio_path)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if needed
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)
        
        # Flatten to 1D
        waveform = waveform.squeeze()
        
        # Pad or truncate to target length
        if waveform.shape[0] < self.target_length:
            # Pad with zeros
            padding = self.target_length - waveform.shape[0]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        elif waveform.shape[0] > self.target_length:
            # Truncate
            waveform = waveform[:self.target_length]
        
        # Extract text from filename or use placeholder
        # For now, extract from filename pattern
        filename = audio_path.stem
        
        # Try to extract item name from filename
        # Format: sample_XX_item_name.wav or synth_XX_XX_item_name.wav
        parts = filename.split('_')
        if filename.startswith('sample'):
            # Human voice: sample_01_chicken_teriyaki
            item_name = ' '.join(parts[2:]).replace('_', ' ').title()
        elif filename.startswith('synth'):
            # Synthetic: synth_00_00_chicken_teriyaki
            item_name = ' '.join(parts[3:]).replace('_', ' ').title()
        else:
            item_name = "Unknown"
        
        # Create full text (what the model should learn)
        text = f"print five {item_name.lower()}"
        
        # Encode text to indices
        text_encoded = [self.char_to_idx.get(c, self.char_to_idx['<unk>']) for c in text]
        text_encoded = torch.tensor(text_encoded, dtype=torch.long)
        
        return waveform, text_encoded


def collate_fn(batch):
    """Collate function for DataLoader"""
    audios, texts = zip(*batch)
    
    # Stack audios (all same length)
    audios = torch.stack(audios)
    
    # Pad text sequences
    text_lengths = torch.tensor([len(t) for t in texts])
    max_text_len = max(text_lengths)
    
    # Pad texts with 0 (blank token)
    texts_padded = torch.zeros(len(texts), max_text_len, dtype=torch.long)
    for i, text in enumerate(texts):
        texts_padded[i, :len(text)] = text
    
    return audios, texts_padded, text_lengths


def train_model(epochs=30, batch_size=8, learning_rate=0.0001):
    """Train the Transformer model"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 
                         'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    # Load vocabulary
    vocab, char_to_idx = create_character_vocab()
    
    # Create datasets
    print("\n📚 Loading datasets...")
    
    # Combine human voice + synthetic data
    datasets = []
    
    # Human voice (if available)
    human_voice_dir = Path("data/my_voice_20")
    if human_voice_dir.exists():
        human_dataset = SushiVoiceDataset(human_voice_dir)
        datasets.append(human_dataset)
    
    # Synthetic data
    synthetic_dir = Path("data/synthetic_20")
    if synthetic_dir.exists():
        synthetic_dataset = SushiVoiceDataset(synthetic_dir)
        datasets.append(synthetic_dataset)
    
    if not datasets:
        print("❌ No training data found!")
        print("   Run: python3 generate_training_data_20.py")
        print("   Run: python3 record_20_human_voice.py")
        return
    
    # Combine datasets
    from torch.utils.data import ConcatDataset
    full_dataset = ConcatDataset(datasets)
    print(f"✅ Total training samples: {len(full_dataset)}")
    
    # Create DataLoader
    train_loader = DataLoader(
        full_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # Create model
    print("\n🏗️  Building Transformer model...")
    model = TransformerAudioSLM(
        vocab_size=len(vocab),
        d_model=256,
        num_layers=12,
        num_heads=4,
        d_ff=1024,
        conv_kernel_size=31,
        dropout=0.1,
        sample_rate=16000
    ).to(device)
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"✅ Model created ({param_count:,} parameters)")
    
    # Loss and optimizer
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Training loop
    print(f"\n🚀 Training for {epochs} epochs...\n")
    
    best_loss = float('inf')
    model_save_path = Path("models/transformer_20_best.pth")
    model_save_path.parent.mkdir(exist_ok=True)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx, (audios, texts, text_lengths) in enumerate(progress):
            audios = audios.to(device)
            texts = texts.to(device)
            text_lengths = text_lengths.to(device)
            
            # Forward pass
            log_probs, output_lengths = model(audios)
            
            # Reshape for CTC loss
            # log_probs: (batch, time, vocab) -> (time, batch, vocab)
            log_probs = log_probs.permute(1, 0, 2)
            
            # CTC loss
            loss = criterion(log_probs, texts, output_lengths, text_lengths)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            progress.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}: Average Loss = {avg_loss:.4f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'vocab': vocab,
                'char_to_idx': char_to_idx,
                'epoch': epoch,
                'loss': best_loss
            }, model_save_path)
            print(f"💾 Saved best model (loss: {best_loss:.4f})")
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    print(f"📁 Model saved to: {model_save_path}")
    print("\n🎯 NEXT STEP:")
    print("   Test it: python3 demo_transformer_voice.py")
    print("\n")


def main():
    parser = argparse.ArgumentParser(description='Train Transformer on 20 sushi items')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("🍣 TRAIN SUSHI VOICE TRANSFORMER - 20 ITEMS")
    print("="*60)
    
    train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )


if __name__ == '__main__':
    main()
