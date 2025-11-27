#!/usr/bin/env python3
"""
Custom ASR Model for SushiVoice
CNN encoder + Bidirectional LSTM + CTC decoder (~5-10M parameters)
"""

import torch
import torch.nn as nn
import torchaudio


class SushiASRModel(nn.Module):
    """
    Custom CTC-based ASR model optimized for sushi voice commands
    
    Architecture:
        Input: Raw waveform (batch, 1, time) at 16kHz
        1. Feature Extraction: CNN layers (32->64->128->256 filters)
        2. Sequence Encoding: Bidirectional LSTM (2-3 layers, 256-512 hidden)
        3. CTC Decoder: Linear projection to character vocab
    
    Parameters: ~5-8M with default configuration
    """
    
    def __init__(
        self,
        vocab_size: int,
        n_cnn_layers: int = 4,
        cnn_channels: list = None,
        cnn_kernel_sizes: list = None,
        n_rnn_layers: int = 3,
        rnn_hidden_size: int = 384,
        rnn_dropout: float = 0.3,
        use_spec_features: bool = True,
        sample_rate: int = 16000
    ):
        """
        Initialize SushiASR model
        
        Args:
            vocab_size: Size of character vocabulary (including blank token)
            n_cnn_layers: Number of CNN layers
            cnn_channels: List of output channels for each CNN layer
            cnn_kernel_sizes: List of kernel sizes for each CNN layer
            n_rnn_layers: Number of LSTM layers
            rnn_hidden_size: Hidden size for LSTM
            rnn_dropout: Dropout probability
            use_spec_features: Use mel spectrogram features instead of raw waveform
            sample_rate: Audio sample rate
        """
        super(SushiASRModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.use_spec_features = use_spec_features
        self.sample_rate = sample_rate
        
        # Default CNN architecture
        if cnn_channels is None:
            cnn_channels = [32, 64, 128, 256]
        if cnn_kernel_sizes is None:
            cnn_kernel_sizes = [11, 5, 3, 3]
        
        # Mel spectrogram transform (optional)
        if use_spec_features:
            self.mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=512,
                hop_length=160,  # 10ms hop
                n_mels=80
            )
            cnn_input_channels = 1  # Mel spectrogram
        else:
            self.mel_transform = None
            cnn_input_channels = 1  # Raw waveform
        
        # CNN Feature Extractor
        cnn_layers = []
        in_channels = cnn_input_channels
        
        for idx in range(n_cnn_layers):
            out_channels = cnn_channels[idx]
            kernel_size = cnn_kernel_sizes[idx]
            
            # Conv1D layer
            cnn_layers.append(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=2 if idx < 2 else 1,  # Stride 2 for first 2 layers
                    padding=kernel_size // 2
                )
            )
            cnn_layers.append(nn.BatchNorm1d(out_channels))
            cnn_layers.append(nn.ReLU(inplace=True))
            cnn_layers.append(nn.Dropout(0.1))
            
            in_channels = out_channels
        
        self.cnn = nn.Sequential(*cnn_layers)
        
        # Calculate CNN output dimension
        self.cnn_output_dim = cnn_channels[-1]
        
        # Layer normalization before LSTM
        self.layer_norm = nn.LayerNorm(self.cnn_output_dim)
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=self.cnn_output_dim,
            hidden_size=rnn_hidden_size,
            num_layers=n_rnn_layers,
            batch_first=True,
            dropout=rnn_dropout if n_rnn_layers > 1 else 0.0,
            bidirectional=True
        )
        
        # CTC decoder (linear projection)
        self.fc = nn.Linear(rnn_hidden_size * 2, vocab_size)  # *2 for bidirectional
        
    def forward(self, x, input_lengths=None):
        """
        Forward pass
        
        Args:
            x: Input audio tensor (batch, 1, time) or (batch, time) for raw waveform
                or (batch, 1, freq, time) for spectrograms
            input_lengths: Actual lengths of sequences (for CTC loss)
        
        Returns:
            log_probs: Log probabilities (batch, time, vocab_size)
            output_lengths: Output sequence lengths
        """
        batch_size = x.size(0)
        
        # Apply mel spectrogram if needed
        if self.use_spec_features and len(x.shape) == 3:
            # x: (batch, 1, time)
            x = x.squeeze(1)  # (batch, time)
            x = self.mel_transform(x)  # (batch, n_mels, time)
            x = x.log()  # Log mel spectrogram
        
        # Ensure correct shape for CNN: (batch, channels, time)
        if len(x.shape) == 4:  # (batch, 1, freq, time)
            x = x.squeeze(1)  # (batch, freq, time)
        
        # CNN feature extraction
        x = self.cnn(x)  # (batch, cnn_output_dim, time)
        
        # Transpose for LSTM: (batch, time, features)
        x = x.transpose(1, 2)
        
        # Layer normalization
        x = self.layer_norm(x)
        
        # LSTM encoding
        if input_lengths is not None:
            # Calculate output lengths after CNN downsampling
            # Assuming stride=2 for first 2 CNN layers
            output_lengths = input_lengths // 4  # Downsampled by factor of 4
            output_lengths = torch.clamp(output_lengths, min=1)
            
            # Pack padded sequence for efficiency
            x = nn.utils.rnn.pack_padded_sequence(
                x, output_lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            x, _ = self.lstm(x)
            x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        else:
            x, _ = self.lstm(x)
            output_lengths = None
        
        # CTC decoder
        logits = self.fc(x)  # (batch, time, vocab_size)
        log_probs = nn.functional.log_softmax(logits, dim=-1)
        
        return log_probs, output_lengths
    
    def get_param_count(self):
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters())


class CTCDecoder:
    """CTC greedy decoder for inference"""
    
    def __init__(self, vocab: list, blank_token: str = '_'):
        """
        Initialize CTC decoder
        
        Args:
            vocab: List of characters in vocabulary
            blank_token: Blank token for CTC
        """
        self.vocab = vocab
        self.blank_token = blank_token
        self.blank_idx = vocab.index(blank_token)
        
        # Create index to character mapping
        self.idx_to_char = {idx: char for idx, char in enumerate(vocab)}
    
    def decode(self, log_probs, merge_repeated=True):
        """
        Greedy CTC decoding
        
        Args:
            log_probs: Log probabilities (batch, time, vocab_size) or (time, vocab_size)
            merge_repeated: Merge repeated characters
        
        Returns:
            List of decoded strings
        """
        if len(log_probs.shape) == 2:
            # Single sequence
            log_probs = log_probs.unsqueeze(0)
        
        batch_size = log_probs.size(0)
        decoded_texts = []
        
        for b in range(batch_size):
            # Greedy decode: take argmax at each timestep
            indices = torch.argmax(log_probs[b], dim=-1).cpu().numpy()
            
            # Remove blanks and merge repeated
            decoded_chars = []
            prev_idx = None
            
            for idx in indices:
                if idx == self.blank_idx:
                    prev_idx = None
                    continue
                
                if merge_repeated and idx == prev_idx:
                    continue
                
                decoded_chars.append(self.idx_to_char[idx])
                prev_idx = idx
            
            decoded_text = ''.join(decoded_chars)
            decoded_texts.append(decoded_text)
        
        return decoded_texts if batch_size > 1 else decoded_texts[0]
    
    def decode_with_confidence(self, log_probs):
        """
        Decode with confidence score
        
        Args:
            log_probs: Log probabilities (time, vocab_size) for single sequence
        
        Returns:
            (decoded_text, confidence_score)
        """
        # Get best path
        probs = torch.exp(log_probs)
        best_path_probs = torch.max(probs, dim=-1)[0]
        
        # Decode
        decoded_text = self.decode(log_probs, merge_repeated=True)
        
        # Compute confidence (average probability along best path)
        confidence = torch.mean(best_path_probs).item()
        
        return decoded_text, confidence


def create_character_vocab():
    """
    Create character-level vocabulary for English sushi commands
    
    Returns:
        vocab: List of characters
        char_to_idx: Dictionary mapping characters to indices
    """
    # Basic ASCII characters + space + blank
    chars = list(' abcdefghijklmnopqrstuvwxyz0123456789')
    
    # Add blank token for CTC
    vocab = ['_'] + chars  # '_' is blank token at index 0
    
    char_to_idx = {char: idx for idx, char in enumerate(vocab)}
    
    return vocab, char_to_idx


if __name__ == '__main__':
    # Test model
    print("Testing SushiASR Model...\n")
    
    # Create vocabulary
    vocab, char_to_idx = create_character_vocab()
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Characters: {''.join(vocab)}\n")
    
    # Create model
    model = SushiASRModel(
        vocab_size=len(vocab),
        n_cnn_layers=4,
        n_rnn_layers=3,
        rnn_hidden_size=384,
        use_spec_features=True
    )
    
    print(f"Model Parameters: {model.get_param_count():,}")
    print(f"Target: 5-10M parameters\n")
    
    # Test forward pass
    batch_size = 4
    audio_length = 48000  # 3 seconds at 16kHz
    x = torch.randn(batch_size, 1, audio_length)
    input_lengths = torch.tensor([48000, 40000, 35000, 32000])
    
    print(f"Input shape: {x.shape}")
    
    with torch.no_grad():
        log_probs, output_lengths = model(x, input_lengths)
    
    print(f"Output shape: {log_probs.shape}")
    print(f"Output lengths: {output_lengths}\n")
    
    # Test decoder
    decoder = CTCDecoder(vocab)
    decoded = decoder.decode(log_probs[0])
    print(f"Decoded (random): '{decoded}'")
    
    # Test confidence
    decoded_text, confidence = decoder.decode_with_confidence(log_probs[0])
    print(f"Confidence: {confidence:.3f}")
    
    print("\n✅ Model test complete!")
