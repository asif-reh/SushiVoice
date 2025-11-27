#!/usr/bin/env python3
"""
Training Script for SushiVoice Custom ASR Model
AdamW optimizer, CTC loss, WER/CER metrics, TensorBoard logging
"""

import json
import os
import argparse
from pathlib import Path
from typing import List, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchaudio
from tqdm import tqdm
import numpy as np

from custom_asr import SushiASRModel, CTCDecoder, create_character_vocab


class SushiVoiceDataset(Dataset):
    """Dataset class for SushiVoice audio-text pairs"""
    
    def __init__(self, manifest_path: str, char_to_idx: dict, max_audio_length: int = 80000):
        """
        Initialize dataset
        
        Args:
            manifest_path: Path to manifest.jsonl
            char_to_idx: Character to index mapping
            max_audio_length: Maximum audio length in samples
        """
        self.char_to_idx = char_to_idx
        self.max_audio_length = max_audio_length
        
        # Load manifest
        self.samples = []
        with open(manifest_path, 'r') as f:
            for line in f:
                entry = json.loads(line)
                self.samples.append(entry)
        
        print(f"Loaded {len(self.samples)} samples from {manifest_path}")
    
    def __len__(self):
        return len(self.samples)
    
    def text_to_indices(self, text: str) -> List[int]:
        """Convert text to list of character indices"""
        return [self.char_to_idx.get(c, 0) for c in text.lower()]
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load audio
        audio_path = sample['audio']
        if not os.path.isabs(audio_path):
            # Convert relative path to absolute
            audio_path = Path(audio_path)
            if not audio_path.exists():
                # Try relative to manifest location
                manifest_dir = Path(os.path.dirname(self.samples[0]['audio'])).parent
                audio_path = manifest_dir / sample['audio']
        
        audio, sr = torchaudio.load(str(audio_path))
        
        # Resample to 16kHz if needed
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            audio = resampler(audio)
        
        # Convert to mono
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        
        # Pad or truncate
        audio_length = audio.shape[1]
        if audio_length > self.max_audio_length:
            audio = audio[:, :self.max_audio_length]
            audio_length = self.max_audio_length
        
        # Convert text to indices
        text = sample['text']
        text_indices = self.text_to_indices(text)
        
        return {
            'audio': audio.squeeze(0),  # (time,)
            'audio_length': audio_length,
            'text': text,
            'text_indices': torch.tensor(text_indices, dtype=torch.long),
            'text_length': len(text_indices)
        }


def collate_fn(batch):
    """Custom collate function for batching"""
    # Sort by audio length (descending) for pack_padded_sequence
    batch = sorted(batch, key=lambda x: x['audio_length'], reverse=True)
    
    # Get max lengths
    max_audio_length = max(item['audio_length'] for item in batch)
    max_text_length = max(item['text_length'] for item in batch)
    
    # Pad sequences
    batch_audio = []
    audio_lengths = []
    batch_text_indices = []
    text_lengths = []
    texts = []
    
    for item in batch:
        # Pad audio
        audio = item['audio']
        padded_audio = torch.zeros(max_audio_length)
        padded_audio[:item['audio_length']] = audio
        batch_audio.append(padded_audio)
        audio_lengths.append(item['audio_length'])
        
        # Pad text
        text_indices = item['text_indices']
        padded_text = torch.zeros(max_text_length, dtype=torch.long)
        padded_text[:item['text_length']] = text_indices
        batch_text_indices.append(padded_text)
        text_lengths.append(item['text_length'])
        
        texts.append(item['text'])
    
    return {
        'audio': torch.stack(batch_audio).unsqueeze(1),  # (batch, 1, time)
        'audio_lengths': torch.tensor(audio_lengths, dtype=torch.long),
        'text_indices': torch.stack(batch_text_indices),
        'text_lengths': torch.tensor(text_lengths, dtype=torch.long),
        'texts': texts
    }


def compute_wer(reference: str, hypothesis: str) -> float:
    """
    Compute Word Error Rate
    
    Args:
        reference: Ground truth text
        hypothesis: Predicted text
    
    Returns:
        WER score (0-1)
    """
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    
    # Levenshtein distance for words
    d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1))
    
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j
    
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                d[i][j] = min(d[i-1][j], d[i][j-1], d[i-1][j-1]) + 1
    
    wer = d[len(ref_words)][len(hyp_words)] / len(ref_words) if len(ref_words) > 0 else 0
    return wer


def train_epoch(model, dataloader, optimizer, criterion, device, decoder):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_wer = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        audio = batch['audio'].to(device)
        audio_lengths = batch['audio_lengths'].to(device)
        text_indices = batch['text_indices'].to(device)
        text_lengths = batch['text_lengths'].to(device)
        
        # Forward pass
        log_probs, output_lengths = model(audio, audio_lengths)
        
        # Transpose for CTC loss: (time, batch, vocab)
        log_probs = log_probs.transpose(0, 1)
        
        # CTC loss
        loss = criterion(log_probs, text_indices, output_lengths, text_lengths)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Compute WER on batch
        with torch.no_grad():
            decoded_texts = decoder.decode(log_probs.transpose(0, 1)[0])
            wer = compute_wer(batch['texts'][0], decoded_texts)
            total_wer += wer
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f"{loss.item():.4f}", 'wer': f"{wer:.3f}"})
    
    avg_loss = total_loss / len(dataloader)
    avg_wer = total_wer / len(dataloader)
    
    return avg_loss, avg_wer


def validate(model, dataloader, criterion, device, decoder):
    """Validate model"""
    model.eval()
    total_loss = 0
    total_wer = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for batch in pbar:
            audio = batch['audio'].to(device)
            audio_lengths = batch['audio_lengths'].to(device)
            text_indices = batch['text_indices'].to(device)
            text_lengths = batch['text_lengths'].to(device)
            
            # Forward pass
            log_probs, output_lengths = model(audio, audio_lengths)
            
            # Transpose for CTC loss
            log_probs_t = log_probs.transpose(0, 1)
            
            # CTC loss
            loss = criterion(log_probs_t, text_indices, output_lengths, text_lengths)
            
            # Compute WER
            for i, text in enumerate(batch['texts']):
                decoded_text = decoder.decode(log_probs[i])
                wer = compute_wer(text, decoded_text)
                total_wer += wer
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    avg_loss = total_loss / len(dataloader)
    avg_wer = total_wer / (len(dataloader) * dataloader.batch_size)
    
    return avg_loss, avg_wer


def main():
    parser = argparse.ArgumentParser(description='Train SushiVoice ASR model')
    parser.add_argument('--manifest', type=str, default='data/manifest_augmented.jsonl',
                        help='Path to training manifest')
    parser.add_argument('--val_split', type=float, default=0.15,
                        help='Validation split ratio')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--checkpoint_dir', type=str, default='models',
                        help='Directory to save checkpoints')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (cuda/mps/cpu/auto)')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Create vocabulary
    vocab, char_to_idx = create_character_vocab()
    print(f"Vocabulary size: {len(vocab)}")
    
    # Load dataset
    full_dataset = SushiVoiceDataset(args.manifest, char_to_idx)
    
    # Split train/val
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0  # Set to 0 for macOS compatibility
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0
    )
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Create model
    model = SushiASRModel(
        vocab_size=len(vocab),
        n_cnn_layers=4,
        n_rnn_layers=3,
        rnn_hidden_size=384,
        rnn_dropout=0.3,
        use_spec_features=True
    ).to(device)
    
    print(f"Model parameters: {model.get_param_count():,}")
    
    # Loss and optimizer
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Decoder
    decoder = CTCDecoder(vocab)
    
    # TensorBoard
    writer = SummaryWriter(log_dir=f'runs/sushivoice_training')
    
    # Training loop
    best_val_wer = float('inf')
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nStarting training...")
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss, train_wer = train_epoch(model, train_loader, optimizer, criterion, device, decoder)
        
        # Validate
        val_loss, val_wer = validate(model, val_loader, criterion, device, decoder)
        
        # Update scheduler
        scheduler.step()
        
        # Log
        print(f"Train Loss: {train_loss:.4f}, Train WER: {train_wer:.3f}")
        print(f"Val Loss: {val_loss:.4f}, Val WER: {val_wer:.3f}")
        
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('WER/train', train_wer, epoch)
        writer.add_scalar('WER/val', val_wer, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        # Save best model
        if val_wer < best_val_wer:
            best_val_wer = val_wer
            checkpoint_path = checkpoint_dir / 'sushi_slm_best.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_wer': val_wer,
                'vocab': vocab,
                'char_to_idx': char_to_idx
            }, checkpoint_path)
            print(f"✅ Saved best model (WER: {val_wer:.3f})")
        
        # Save last checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_wer': val_wer,
            'vocab': vocab,
            'char_to_idx': char_to_idx
        }, checkpoint_dir / 'sushi_slm_last.pth')
    
    writer.close()
    print(f"\n🎉 Training complete! Best val WER: {best_val_wer:.3f}")


if __name__ == '__main__':
    main()
