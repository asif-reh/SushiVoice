#!/usr/bin/env python3
"""
Real-Time ASR Inference Pipeline for SushiVoice
Integrates VAD, ASR model, command parser, and printer
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging
from pathlib import Path
import torch
import torchaudio
import numpy as np

from models.custom_asr import SushiASRModel, CTCDecoder, create_character_vocab
from inference.vad import VoiceActivityDetector, EnergyVAD
from inference.parser import CommandParser
from inference.printer import SushiPrinter

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    logging.warning("PyAudio not installed. Real-time microphone input will be disabled.")


class SushiASR:
    """Complete ASR pipeline for voice-to-label printing"""
    
    def __init__(
        self,
        model_path: str,
        device: str = 'cpu',
        confidence_threshold: float = 0.7,
        use_vad: bool = True,
        printer_config: dict = None
    ):
        """
        Initialize SushiASR pipeline
        
        Args:
            model_path: Path to trained model checkpoint
            device: Compute device ('cpu', 'cuda', 'mps')
            confidence_threshold: Minimum confidence to accept transcription
            use_vad: Enable voice activity detection
            printer_config: Printer configuration dict
        """
        self.device = torch.device(device)
        self.confidence_threshold = confidence_threshold
        self.use_vad = use_vad
        
        # Load model
        logging.info(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Get vocabulary
        if 'vocab' in checkpoint:
            self.vocab = checkpoint['vocab']
            self.char_to_idx = checkpoint['char_to_idx']
        else:
            self.vocab, self.char_to_idx = create_character_vocab()
        
        # Create model
        self.model = SushiASRModel(
            vocab_size=len(self.vocab),
            n_cnn_layers=4,
            n_rnn_layers=3,
            rnn_hidden_size=384,
            use_spec_features=True
        ).to(self.device)
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        logging.info(f"Model loaded successfully ({self.model.get_param_count():,} parameters)")
        
        # Initialize decoder
        self.decoder = CTCDecoder(self.vocab)
        
        # Initialize VAD
        if self.use_vad:
            self.vad = VoiceActivityDetector(sample_rate=16000, aggressiveness=2)
            logging.info("VAD initialized")
        else:
            self.vad = None
        
        # Initialize command parser
        self.parser = CommandParser()
        logging.info("Command parser initialized")
        
        # Initialize printer
        if printer_config is None:
            printer_config = {'printer_type': 'dummy', 'dummy_mode': True}
        
        self.printer = SushiPrinter(**printer_config)
        logging.info("Printer initialized")
    
    def preprocess_audio(self, audio: np.ndarray, sr: int = 16000) -> torch.Tensor:
        """
        Preprocess audio for model input
        
        Args:
            audio: Audio array
            sr: Sample rate
        
        Returns:
            Preprocessed audio tensor
        """
        # Convert to float32
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32767.0
        
        # Apply VAD if enabled
        if self.use_vad and self.vad:
            try:
                audio_filtered = self.vad.filter_speech_frames(audio)
                if len(audio_filtered) > 0:
                    audio = audio_filtered
            except Exception as e:
                logging.warning(f"VAD error: {e}, using full audio")
        
        # Convert to tensor
        audio_tensor = torch.from_numpy(audio).float()
        
        # Add batch and channel dimensions
        audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, time)
        
        return audio_tensor
    
    def transcribe(self, audio: np.ndarray, sr: int = 16000) -> dict:
        """
        Transcribe audio to text
        
        Args:
            audio: Audio array
            sr: Sample rate
        
        Returns:
            Dict with 'text', 'confidence', 'raw_logits'
        """
        # Preprocess
        audio_tensor = self.preprocess_audio(audio, sr).to(self.device)
        
        # Inference
        with torch.no_grad():
            log_probs, _ = self.model(audio_tensor)
            
            # Decode
            text, confidence = self.decoder.decode_with_confidence(log_probs[0])
        
        return {
            'text': text,
            'confidence': confidence,
            'raw_logits': log_probs[0].cpu()
        }
    
    def process_command(self, audio: np.ndarray, sr: int = 16000, auto_print: bool = True) -> dict:
        """
        Process audio command end-to-end
        
        Args:
            audio: Audio array
            sr: Sample rate
            auto_print: Automatically print labels if parsed successfully
        
        Returns:
            Result dict with 'transcript', 'parsed', 'printed' keys
        """
        # Transcribe
        transcript_result = self.transcribe(audio, sr)
        
        logging.info(f"Transcribed: '{transcript_result['text']}' (conf: {transcript_result['confidence']:.3f})")
        
        # Check confidence
        if transcript_result['confidence'] < self.confidence_threshold:
            logging.warning(f"Low confidence ({transcript_result['confidence']:.3f}), skipping")
            return {
                'transcript': transcript_result,
                'parsed': None,
                'printed': False,
                'error': 'low_confidence'
            }
        
        # Parse command
        parsed = self.parser.parse_with_fallback(
            transcript_result['text'],
            confidence=transcript_result['confidence']
        )
        
        if not parsed:
            logging.warning(f"Failed to parse command: '{transcript_result['text']}'")
            return {
                'transcript': transcript_result,
                'parsed': None,
                'printed': False,
                'error': 'parse_failed'
            }
        
        logging.info(f"Parsed: {parsed['quantity']}x {parsed['item']}")
        
        # Print labels
        if auto_print:
            try:
                self.printer.print_label(parsed['item'], parsed['quantity'])
                logging.info(f"Printed {parsed['quantity']}x {parsed['item']}")
                printed = True
            except Exception as e:
                logging.error(f"Print error: {e}")
                printed = False
        else:
            printed = False
        
        return {
            'transcript': transcript_result,
            'parsed': parsed,
            'printed': printed
        }
    
    def record_from_mic(self, duration: float = 5.0, sample_rate: int = 16000) -> np.ndarray:
        """
        Record audio from microphone
        
        Args:
            duration: Recording duration in seconds
            sample_rate: Sample rate
        
        Returns:
            Audio array
        """
        if not PYAUDIO_AVAILABLE:
            raise RuntimeError("PyAudio not installed. Cannot record from microphone.")
        
        p = pyaudio.PyAudio()
        
        # Open stream
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=sample_rate,
            input=True,
            frames_per_buffer=1024
        )
        
        logging.info(f"Recording for {duration}s...")
        
        # Record
        frames = []
        for _ in range(int(sample_rate * duration / 1024)):
            data = stream.read(1024)
            frames.append(np.frombuffer(data, dtype=np.int16))
        
        # Close
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        audio = np.concatenate(frames)
        logging.info("Recording complete")
        
        return audio
    
    def run_continuous(self, record_duration: float = 3.0):
        """
        Run continuous voice recognition loop
        
        Args:
            record_duration: Duration of each recording chunk
        """
        if not PYAUDIO_AVAILABLE:
            logging.error("PyAudio not installed. Cannot run continuous mode.")
            return
        
        logging.info("Starting continuous voice recognition...")
        logging.info(f"Confidence threshold: {self.confidence_threshold}")
        logging.info("Press Ctrl+C to stop\n")
        
        try:
            while True:
                # Record
                audio = self.record_from_mic(duration=record_duration)
                
                # Process
                result = self.process_command(audio, auto_print=True)
                
                if result.get('printed'):
                    print(f"\n✅ SUCCESS: Printed {result['parsed']['quantity']}x {result['parsed']['item']}\n")
                elif result.get('error') == 'low_confidence':
                    print(f"⚠️  Low confidence transcription, skipped\n")
                elif result.get('error') == 'parse_failed':
                    print(f"❌ Could not parse command: '{result['transcript']['text']}'\n")
                
        except KeyboardInterrupt:
            logging.info("\nStopping...")
            self.printer.close()


def main():
    parser = argparse.ArgumentParser(description='SushiVoice ASR Inference Pipeline')
    parser.add_argument('--model', type=str, default='models/sushi_slm_best.pth',
                        help='Path to trained model')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device (cpu/cuda/mps)')
    parser.add_argument('--confidence', type=float, default=0.7,
                        help='Confidence threshold')
    parser.add_argument('--no_vad', action='store_true',
                        help='Disable VAD')
    parser.add_argument('--continuous', action='store_true',
                        help='Run in continuous mode')
    parser.add_argument('--test_audio', type=str, default=None,
                        help='Path to test audio file')
    parser.add_argument('--printer_type', type=str, default='dummy',
                        help='Printer type (usb/serial/network/dummy)')
    parser.add_argument('--printer_vid', type=str, default=None,
                        help='USB Vendor ID (hex)')
    parser.add_argument('--printer_pid', type=str, default=None,
                        help='USB Product ID (hex)')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Printer config
    printer_config = {'printer_type': args.printer_type}
    if args.printer_type == 'usb' and args.printer_vid and args.printer_pid:
        printer_config['vendor_id'] = int(args.printer_vid, 16)
        printer_config['product_id'] = int(args.printer_pid, 16)
    else:
        printer_config['dummy_mode'] = True
    
    # Initialize pipeline
    try:
        asr = SushiASR(
            model_path=args.model,
            device=args.device,
            confidence_threshold=args.confidence,
            use_vad=not args.no_vad,
            printer_config=printer_config
        )
    except Exception as e:
        logging.error(f"Failed to initialize ASR pipeline: {e}")
        logging.info("Make sure you have trained a model first!")
        return
    
    # Run
    if args.continuous:
        asr.run_continuous()
    elif args.test_audio:
        # Test with audio file
        logging.info(f"Processing test audio: {args.test_audio}")
        audio, sr = torchaudio.load(args.test_audio)
        audio = audio.numpy()[0]  # Convert to mono numpy
        
        result = asr.process_command(audio, sr, auto_print=True)
        
        print(f"\nTranscript: '{result['transcript']['text']}'")
        print(f"Confidence: {result['transcript']['confidence']:.3f}")
        if result['parsed']:
            print(f"Parsed: {result['parsed']['quantity']}x {result['parsed']['item']}")
            print(f"Printed: {result['printed']}")
    else:
        print("No mode specified. Use --continuous or --test_audio")
        print("Example: python asr_pipeline.py --model models/sushi_slm_best.pth --continuous")


if __name__ == '__main__':
    main()
