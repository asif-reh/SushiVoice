#!/usr/bin/env python3
"""
🎤 SushiVoice Transformer Demo
================================

Real-time voice recognition using the Transformer model.
This is the UPGRADED version with better accuracy!

Usage:
    python demo_transformer_voice.py

After training with:
    python train_transformer_slm.py --epochs 20
"""

import sys
import os
import numpy as np

# Setup path
sys.path.append(os.path.dirname(__file__))

# Check dependencies
try:
    import torch
    import torchaudio
    print("✅ PyTorch installed")
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    sys.exit(1)

try:
    import pyaudio
    print("✅ PyAudio installed")
    PYAUDIO_AVAILABLE = True
except ImportError:
    print("⚠️  PyAudio not installed - file mode only")
    PYAUDIO_AVAILABLE = False

# Import components
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from transformer_audio_slm import TransformerAudioSLM, CTCGreedyDecoder, create_character_vocab
from inference.parser import CommandParser
from inference.printer import SushiPrinter


class TransformerSushiVoice:
    """
    Real-time voice recognition using Transformer model
    """
    
    def __init__(self, model_path='models/transformer_20_best.pth', device='cpu'):
        """
        Initialize Transformer-based voice recognition
        
        Args:
            model_path: Path to trained transformer model
            device: Compute device ('cpu', 'cuda', 'mps')
        """
        print("\n🍣 Initializing SushiVoice (Transformer)...")
        
        self.device = torch.device(device)
        self.sample_rate = 16000
        self.recording_duration = 6.0  # Increased from 4.0 for longer commands
        
        # Load model
        print(f"📦 Loading Transformer model from {model_path}...")
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # Get vocabulary
            if 'vocab' in checkpoint:
                self.vocab = checkpoint['vocab']
                self.char_to_idx = checkpoint.get('char_to_idx', {})
            else:
                self.vocab, self.char_to_idx = create_character_vocab()
            
            # Create Transformer model
            self.model = TransformerAudioSLM(
                vocab_size=len(self.vocab),
                d_model=256,
                num_layers=12,
                num_heads=4,
                d_ff=1024,
                conv_kernel_size=31,
                dropout=0.1,
                sample_rate=self.sample_rate
            ).to(self.device)
            
            # Load weights - handle both old format (state_dict only) and new format (checkpoint dict)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # New format: full checkpoint dictionary
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                # Old format: just state_dict
                self.model.load_state_dict(checkpoint)
            self.model.eval()
            
            # Count parameters
            param_count = sum(p.numel() for p in self.model.parameters())
            print(f"✅ Transformer model loaded ({param_count:,} parameters)")
            
        except FileNotFoundError:
            print(f"❌ Model not found: {model_path}")
            print("\nPlease train the Transformer model first:")
            print("  python train_transformer_slm.py --epochs 20")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        
        # Initialize decoder
        self.decoder = CTCGreedyDecoder(self.vocab)
        print("✅ CTC Decoder initialized")
        
        # Initialize parser
        self.parser = CommandParser()
        print("✅ Command parser initialized")
        
        # Initialize printer
        self.printer = SushiPrinter(printer_type='dummy', dummy_mode=True)
        print("✅ Printer initialized (dummy mode)")
        
        print("\n" + "="*60)
        print("🎉 Transformer SushiVoice is ready!")
        print("="*60 + "\n")
    
    def transcribe_audio(self, audio_array):
        """
        Transcribe audio to text using Transformer
        
        Args:
            audio_array: NumPy array of audio samples
        
        Returns:
            (text, confidence) tuple
        """
        # Convert to tensor
        if audio_array.dtype == np.int16:
            audio_array = audio_array.astype(np.float32) / 32768.0
        
        audio_tensor = torch.from_numpy(audio_array).float()
        audio_tensor = audio_tensor.unsqueeze(0)  # (1, time)
        audio_tensor = audio_tensor.to(self.device)
        
        # Inference
        with torch.no_grad():
            log_probs, output_lengths = self.model(audio_tensor)
            
            # Decode
            text, confidence = self.decoder.decode_with_confidence(log_probs[0])
        
        return text, confidence
    
    def process_command(self, audio_array):
        """
        Process voice command: transcribe → parse → print
        
        Args:
            audio_array: Audio samples
        
        Returns:
            Success status
        """
        print("\n" + "-"*60)
        
        # Step 1: Transcribe
        print("🎤 Transcribing with Transformer...")
        text, confidence = self.transcribe_audio(audio_array)
        
        print(f"📝 Transcript: '{text}'")
        print(f"🎯 Confidence: {confidence:.1%}")
        
        # Check confidence (lowered threshold for better acceptance)
        if confidence < 0.2:
            print("⚠️  Very low confidence - please speak more clearly")
            return False
        
        # Step 2: Parse
        print("🔍 Parsing command...")
        parsed = self.parser.parse_with_fallback(text, confidence)
        
        if not parsed:
            print(f"❌ Could not understand command: '{text}'")
            print("💡 Try saying: 'hey yosushi print 5 chicken teriyaki'")
            return False
        
        print(f"✅ Parsed: {parsed['quantity']}x {parsed['item']}")
        
        # Step 3: Print
        print(f"🖨️  Printing {parsed['quantity']} labels...")
        
        try:
            self.printer.print_label(parsed['item'], parsed['quantity'])
            
            print("\n" + "="*60)
            print(f"✅ SUCCESS! Printed {parsed['quantity']}x {parsed['item']}")
            print("="*60)
            
            return True
            
        except Exception as e:
            print(f"❌ Print error: {e}")
            return False
    
    def record_from_microphone(self):
        """Record audio from microphone"""
        if not PYAUDIO_AVAILABLE:
            print("❌ PyAudio not installed")
            return None
        
        try:
            p = pyaudio.PyAudio()
            
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=1024
            )
            
            print(f"\n🎤 Recording for {self.recording_duration} seconds...")
            print("📢 Speak now!")
            
            frames = []
            num_chunks = int(self.sample_rate * self.recording_duration / 1024)
            
            for i in range(num_chunks):
                data = stream.read(1024, exception_on_overflow=False)
                frames.append(np.frombuffer(data, dtype=np.int16))
                
                if i % 20 == 0:
                    print(".", end="", flush=True)
            
            print(" Done!")
            
            stream.stop_stream()
            stream.close()
            p.terminate()
            
            audio = np.concatenate(frames)
            return audio
            
        except Exception as e:
            print(f"❌ Recording error: {e}")
            return None
    
    def run_interactive(self):
        """Run interactive voice recognition loop"""
        print("\n" + "="*60)
        print("🎤 INTERACTIVE MODE (Transformer)")
        print("="*60)
        print("\nCommands:")
        print("  - Press ENTER to record")
        print("  - Type 'quit' to exit")
        print("\nExample phrases:")
        print("  'hey yosushi print 5 chicken teriyaki'")
        print("  'yosushi label 20 teriyaki rice bowl'")
        print("  'print 10 california roll'")
        print("")
        
        try:
            while True:
                user_input = input("\n👉 Press ENTER to speak (or 'quit' to exit): ").strip().lower()
                
                if user_input in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!")
                    break
                
                audio = self.record_from_microphone()
                
                if audio is None:
                    print("❌ Recording failed")
                    continue
                
                self.process_command(audio)
                
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted by user. Goodbye!")
    
    def test_with_file(self, audio_path):
        """Test with audio file"""
        print(f"\n📂 Loading audio from {audio_path}...")
        
        try:
            audio, sr = torchaudio.load(audio_path)
            
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)
            
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                audio = resampler(audio)
            
            audio = audio.squeeze(0).numpy()
            
            print(f"✅ Loaded {len(audio)/self.sample_rate:.1f}s of audio")
            
            self.process_command(audio)
            
        except Exception as e:
            print(f"❌ Error loading audio: {e}")


def main():
    """Main entry point"""
    print("\n" + "="*60)
    print("🍣 SUSHIVOICE TRANSFORMER - Real-Time Demo")
    print("="*60)
    print("\n🚀 Using UPGRADED Transformer model!")
    print("Expected: Better accuracy than CNN-LSTM\n")
    
    # Check for model
    model_path = 'models/transformer_best.pth'
    if not os.path.exists(model_path):
        print(f"\n❌ Transformer model not found: {model_path}")
        print("\n📝 Training steps:")
        print("  1. Train: python train_transformer_slm.py --epochs 20")
        print("  2. Wait for training to complete (~30-40 minutes)")
        print("  3. Then run: python demo_transformer_voice.py")
        
        # Check if CNN-LSTM model exists
        cnn_model = 'models/sushi_slm_best.pth'
        if os.path.exists(cnn_model):
            print(f"\n💡 You have CNN-LSTM model. Use:")
            print("  python demo_realtime_voice.py")
        
        return
    
    # Initialize
    try:
        sushi_voice = TransformerSushiVoice(model_path=model_path, device='cpu')
    except Exception as e:
        print(f"\n❌ Initialization failed: {e}")
        return
    
    # Run
    if len(sys.argv) > 1:
        # Test with audio file
        audio_path = sys.argv[1]
        sushi_voice.test_with_file(audio_path)
    else:
        # Interactive mode
        if PYAUDIO_AVAILABLE:
            sushi_voice.run_interactive()
        else:
            print("\n❌ PyAudio not installed!")
            print("\nInstall with: pip install pyaudio")
            print("Or test with file: python demo_transformer_voice.py audio.wav")


if __name__ == '__main__':
    main()
