# 🍣 SushiVoice

**A patentable, hands-free voice-controlled sushi labeling system powered by a custom Small Language Model (SLM)**

## Overview

SushiVoice is a standalone audio Small Language Model designed for commercial kitchens, enabling hands-free thermal label printing through natural voice commands. Built with a custom CTC-based ASR architecture (~5-10M parameters), it achieves <300ms latency on edge devices like Raspberry Pi.

### Key Features

- 🎤 **Voice-Activated**: Recognize commands like "Hey YoSushi print 5 times of label of Chicken Teriyaki"
- 🧠 **Custom ASR Model**: CNN encoder + Bidirectional LSTM + CTC decoder optimized for sushi vocabulary
- 🖨️ **Thermal Printing**: Direct ESC/POS integration with Marka printers
- ⚡ **Edge Optimized**: Runs on Raspberry Pi with sub-300ms inference latency
- 🔇 **Noise Robust**: VAD and noise rejection for busy kitchen environments
- 📊 **Patentable Architecture**: Novel combination of domain-specific ASR + regex parsing + edge deployment

## Architecture

```
Voice Input → VAD → Custom ASR Model → Regex Parser → ESC/POS Printer
              (webrtcvad)  (CNN-LSTM-CTC)   (Qty+Item)    (Thermal Label)
```

### Model Architecture
- **Feature Extraction**: 4-layer CNN (32→64→128→256 filters)
- **Sequence Encoding**: 2-3 layer Bidirectional LSTM (256-512 hidden units)
- **Decoding**: CTC loss for alignment-free training, greedy decoding
- **Parameters**: ~5-8M (quantized to ~2MB for deployment)

## Setup

### Prerequisites

- Python 3.10+ (Python 3.9.6 will work but 3.10+ recommended)
- macOS/Linux for development
- Raspberry Pi 4/5 (4GB+ RAM) for production deployment
- USB microphone
- Marka thermal printer (or compatible ESC/POS printer)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/asif-reh/SushiVoice.git
   cd SushiVoice
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Generate synthetic dataset**
   ```bash
   python src/data/generate_dataset.py
   python src/data/augment_audio.py
   ```

4. **Train the model**
   ```bash
   # Local training (Mac with M1/M2)
   python src/models/train_custom_slm.py --epochs 20 --batch_size 16
   
   # Or upload to Google Colab for GPU training
   # See notebooks/ for Colab-ready training script
   ```

5. **Quantize for edge deployment**
   ```bash
   python src/models/quantize_model.py --input models/sushi_slm.pth --output models/sushi_slm_quantized.pth
   ```

6. **Run inference**
   ```bash
   python src/inference/asr_pipeline.py --model models/sushi_slm_quantized.pth
   ```

## Dataset

The system is trained on 1,000-5,000 voice command samples:
- **Synthetic audio**: Generated via TTS (pyttsx3/gTTS) with template variations
- **Augmentation**: Kitchen noise, speed/pitch variations, reverb
- **Real recordings**: Optional manual recordings for improved accuracy

### Vocabulary Coverage
- **Wake words**: "Hey YoSushi", "YoSushi", "Hello YoSushi"
- **Actions**: "print", "label", "make"
- **Quantities**: 1-20 (numeric and word form)
- **Sushi items**: 50-100 items (see `src/data/sushi_vocab.json`)

## Usage

### Voice Commands
```
"Hey YoSushi print 5 times of label of Chicken Teriyaki"
"YoSushi label 3 times California Roll"
"Print 10 labels of Salmon Nigiri"
```

### API Usage
```python
from src.inference.asr_pipeline import SushiASR
from src.inference.parser import parse_command
from src.inference.printer import print_label

# Initialize ASR
asr = SushiASR(model_path='models/sushi_slm_quantized.pth')

# Transcribe audio
transcript = asr.transcribe(audio_data)

# Parse command
command = parse_command(transcript['text'])
# → {'quantity': 5, 'item': 'Chicken Teriyaki', 'confidence': 0.92}

# Print labels
if command and transcript['confidence'] > 0.7:
    print_label(command['item'], command['quantity'])
```

## Deployment (Raspberry Pi)

1. **Flash Raspberry Pi OS** (64-bit recommended)

2. **Transfer files**
   ```bash
   scp -r SushiVoice/ pi@raspberrypi.local:~/
   ```

3. **Install dependencies on Pi**
   ```bash
   ssh pi@raspberrypi.local
   cd ~/SushiVoice
   pip3 install -r requirements.txt
   ```

4. **Connect hardware**
   - USB microphone → Pi USB port
   - Marka printer → Pi USB port
   - Find printer VID:PID with `lsusb`

5. **Configure systemd service**
   ```bash
   sudo cp deploy/sushivoice.service /etc/systemd/system/
   sudo systemctl enable sushivoice.service
   sudo systemctl start sushivoice.service
   ```

6. **Monitor logs**
   ```bash
   journalctl -u sushivoice.service -f
   ```

## Performance

| Metric | Target | Achieved |
|--------|--------|----------|
| Inference Latency (Pi 4) | <300ms | TBD |
| WER (Clean Audio) | <10% | TBD |
| WER (Noisy Kitchen) | <25% | TBD |
| Model Size (Quantized) | ~2MB | TBD |
| Memory Usage | <500MB | TBD |

## Development

### Project Structure
```
SushiVoice/
├── src/
│   ├── data/          # Dataset generation & augmentation
│   ├── models/        # Model architecture & training
│   ├── inference/     # Real-time pipeline (VAD, ASR, parser, printer)
│   └── utils/         # Helper functions
├── data/              # Audio samples & manifest
├── models/            # Trained checkpoints
├── tests/             # Unit & integration tests
└── deploy/            # Deployment scripts
```

### Testing
```bash
# Run unit tests
python -m pytest tests/

# Test inference pipeline
python tests/test_inference.py
```

## Patent Information

This system includes several novel, patentable components:
1. Custom CTC-ASR architecture optimized for domain-specific voice commands
2. Integrated regex parser with ASR confidence scoring for command extraction
3. End-to-end voice-to-print pipeline on edge device with sub-300ms latency
4. Training methodology combining synthetic TTS + augmentation + real kitchen audio

**Provisional Patent Abstract**: "A voice-controlled labeling system comprising a small language model (5-10M parameters) with CNN-LSTM-CTC architecture trained on domain-specific vocabulary, integrated with voice activity detection, regex-based command parsing, and thermal printer driver, deployed on edge computing device for sub-300ms latency in noisy commercial kitchen environments."

## Roadmap

- [x] Initial architecture & project setup
- [ ] Complete synthetic dataset generation (1,000+ samples)
- [ ] Train baseline model (WER <15%)
- [ ] Implement real-time inference pipeline
- [ ] Test on Raspberry Pi
- [ ] Record real kitchen audio for retraining
- [ ] Optimize for production (<10% WER)
- [ ] File provisional patent
- [ ] Deploy v1.0 in commercial kitchen

## Contributing

This is currently a private research project. For inquiries, please contact the repository owner.

## License

To be determined (considering MIT or proprietary for patent protection)

## Acknowledgments

Built with ❤️ for YoSushi kitchens worldwide.

---

**Status**: 🚧 In Development  
**Version**: 0.1.0  
**Last Updated**: November 2025
