#!/usr/bin/env python3
"""
Voice Activity Detection for SushiVoice
Filters out silence and background noise
"""

import numpy as np
import webrtcvad
import struct


class VoiceActivityDetector:
    """VAD using WebRTC VAD for robust speech detection"""
    
    def __init__(self, sample_rate: int = 16000, frame_duration_ms: int = 30, aggressiveness: int = 2):
        """
        Initialize VAD
        
        Args:
            sample_rate: Audio sample rate (8000, 16000, 32000, or 48000 Hz)
            frame_duration_ms: Frame duration in ms (10, 20, or 30 ms)
            aggressiveness: VAD aggressiveness mode (0-3, higher = more aggressive filtering)
        """
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)
        
        # Initialize WebRTC VAD
        self.vad = webrtcvad.Vad(aggressiveness)
    
    def is_speech(self, audio_frames: bytes) -> bool:
        """
        Check if audio frames contain speech
        
        Args:
            audio_frames: Audio data as bytes (16-bit PCM)
        
        Returns:
            True if speech is detected
        """
        try:
            return self.vad.is_speech(audio_frames, self.sample_rate)
        except Exception as e:
            # Fallback: assume speech if error
            return True
    
    def filter_speech_frames(self, audio: np.ndarray, buffer_frames: int = 5) -> np.ndarray:
        """
        Filter audio to keep only speech frames
        
        Args:
            audio: Audio array (16-bit int or float)
            buffer_frames: Number of frames to buffer before/after speech
        
        Returns:
            Filtered audio containing only speech segments
        """
        # Convert to 16-bit PCM if needed
        if audio.dtype == np.float32 or audio.dtype == np.float64:
            audio = (audio * 32767).astype(np.int16)
        elif audio.dtype != np.int16:
            audio = audio.astype(np.int16)
        
        # Pad audio to be divisible by frame size
        pad_size = self.frame_size - (len(audio) % self.frame_size)
        if pad_size != self.frame_size:
            audio = np.pad(audio, (0, pad_size), mode='constant')
        
        # Split into frames
        num_frames = len(audio) // self.frame_size
        frames = audio.reshape(num_frames, self.frame_size)
        
        # Detect speech in each frame
        speech_flags = []
        for frame in frames:
            # Convert to bytes
            frame_bytes = struct.pack(f'{len(frame)}h', *frame)
            is_speech = self.is_speech(frame_bytes)
            speech_flags.append(is_speech)
        
        # Apply buffering around speech regions
        buffered_flags = speech_flags.copy()
        for i in range(len(speech_flags)):
            if speech_flags[i]:
                # Add buffer before
                for j in range(max(0, i - buffer_frames), i):
                    buffered_flags[j] = True
                # Add buffer after
                for j in range(i + 1, min(len(speech_flags), i + buffer_frames + 1)):
                    buffered_flags[j] = True
        
        # Extract speech frames
        speech_frames = [frame for frame, flag in zip(frames, buffered_flags) if flag]
        
        if len(speech_frames) == 0:
            # No speech detected, return original audio
            return audio
        
        # Concatenate speech frames
        filtered_audio = np.concatenate(speech_frames)
        
        return filtered_audio


class EnergyVAD:
    """Simple energy-based VAD as fallback"""
    
    def __init__(self, threshold: float = 0.02, frame_length: int = 480):
        """
        Initialize energy-based VAD
        
        Args:
            threshold: Energy threshold (0-1)
            frame_length: Frame length in samples
        """
        self.threshold = threshold
        self.frame_length = frame_length
    
    def compute_energy(self, frame: np.ndarray) -> float:
        """Compute RMS energy of frame"""
        return np.sqrt(np.mean(frame ** 2))
    
    def is_speech(self, audio: np.ndarray) -> bool:
        """
        Check if audio contains speech based on energy
        
        Args:
            audio: Audio array (float)
        
        Returns:
            True if energy exceeds threshold
        """
        energy = self.compute_energy(audio)
        return energy > self.threshold
    
    def filter_speech_frames(self, audio: np.ndarray, buffer_frames: int = 3) -> np.ndarray:
        """
        Filter audio by energy threshold
        
        Args:
            audio: Audio array (float)
            buffer_frames: Number of frames to buffer
        
        Returns:
            Filtered audio
        """
        # Split into frames
        num_frames = len(audio) // self.frame_length
        frames = audio[:num_frames * self.frame_length].reshape(num_frames, self.frame_length)
        
        # Compute energy for each frame
        energies = [self.compute_energy(frame) for frame in frames]
        
        # Determine speech frames
        speech_flags = [energy > self.threshold for energy in energies]
        
        # Apply buffering
        buffered_flags = speech_flags.copy()
        for i in range(len(speech_flags)):
            if speech_flags[i]:
                for j in range(max(0, i - buffer_frames), min(len(speech_flags), i + buffer_frames + 1)):
                    buffered_flags[j] = True
        
        # Extract speech frames
        speech_frames = [frame for frame, flag in zip(frames, buffered_flags) if flag]
        
        if len(speech_frames) == 0:
            return audio
        
        return np.concatenate(speech_frames)


if __name__ == '__main__':
    # Test VAD
    print("Testing Voice Activity Detection...\n")
    
    # Test WebRTC VAD
    print("1. WebRTC VAD")
    vad = VoiceActivityDetector(sample_rate=16000, aggressiveness=2)
    
    # Simulate silence (low energy)
    silence = np.random.randn(4800).astype(np.int16) * 100
    silence_bytes = struct.pack(f'{len(silence)}h', *silence)
    print(f"   Silence detected as speech: {vad.is_speech(silence_bytes)}")
    
    # Simulate speech (higher energy)
    speech = np.random.randn(4800).astype(np.int16) * 3000
    speech_bytes = struct.pack(f'{len(speech)}h', *speech)
    print(f"   Speech detected as speech: {vad.is_speech(speech_bytes)}")
    
    # Test Energy VAD
    print("\n2. Energy VAD")
    energy_vad = EnergyVAD(threshold=0.02)
    
    silence_float = silence.astype(np.float32) / 32767
    speech_float = speech.astype(np.float32) / 32767
    
    print(f"   Silence detected as speech: {energy_vad.is_speech(silence_float)}")
    print(f"   Speech detected as speech: {energy_vad.is_speech(speech_float)}")
    
    print("\n✅ VAD test complete!")
