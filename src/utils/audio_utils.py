#!/usr/bin/env python3
"""Audio utility functions for SushiVoice"""

import numpy as np
import librosa
import soundfile as sf


def load_audio(path: str, sr: int = 16000, mono: bool = True) -> tuple:
    """
    Load audio file
    
    Args:
        path: Path to audio file
        sr: Target sample rate
        mono: Convert to mono
    
    Returns:
        (audio, sample_rate)
    """
    audio, orig_sr = librosa.load(path, sr=sr, mono=mono)
    return audio, sr


def save_audio(path: str, audio: np.ndarray, sr: int = 16000):
    """Save audio to file"""
    sf.write(path, audio, sr)


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """Normalize audio to [-1, 1]"""
    return audio / np.max(np.abs(audio)) if np.max(np.abs(audio)) > 0 else audio


def compute_snr(signal: np.ndarray, noise: np.ndarray) -> float:
    """Compute Signal-to-Noise Ratio in dB"""
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    return 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
