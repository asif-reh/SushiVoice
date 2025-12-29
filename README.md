# 🍣 SushiVoice - Voice-Controlled Label Printing

Custom-built Audio-to-Text Transformer model for voice-controlled sushi label printing.

## 🎯 What It Does

Say **"print 5 chicken teriyaki"** → Get 5 printed labels with "Chicken Teriyaki"

## ⚡ Quick Start

**Complete workflow in 4 steps (~45 minutes):**

```bash
# 1. Generate synthetic training data (5 min)
python3 generate_training_data_20.py

# 2. Record your voice (20 min)
python3 record_20_human_voice.py

# 3. Train the model (15 min)
python3 train_transformer_20.py --epochs 30

# 4. Test it!
python3 demo_transformer_voice.py
```

## 📋 20 Supported Sushi Items

1. Chicken Teriyaki
2. Salmon Nigiri
3. Tuna Nigiri
4. California Roll
5. Spicy Tuna Roll
6. Eel Avocado Roll
7. Shrimp Tempura Roll
8. Dragon Roll
9. Rainbow Roll
10. Philadelphia Roll
11. Salmon Sashimi
12. Tuna Sashimi
13. Edamame
14. Miso Soup
15. Gyoza
16. Chicken Katsu
17. Tempura Shrimp
18. Vegetable Roll
19. Avocado Roll
20. Cucumber Roll

## 🏗️ Architecture

**Custom Transformer SLM (18.5M parameters)**
- Input: Raw audio (16kHz, 5-6 seconds)
- Processing: Mel Spectrogram → CNN → Transformer Encoder → CTC Decoder
- Output: Text transcription
- Accuracy: 70-85% with 20 human voice samples

**Components:**
- `transformer_audio_slm.py` - Core Transformer model (encoder, decoder, training)
- `src/inference/parser.py` - Command parser (extracts quantity + item)
- `src/inference/printer.py` - Label printer (console/thermal printer)

## 📦 Installation

```bash
# Clone repository
git clone <repo-url>
cd SushiVoice

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Mac/Linux
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Mac users: Install PyAudio
brew install portaudio
pip install pyaudio
```

## 📖 Documentation

Read **WORKFLOW_20_ITEMS.md** for complete workflow, troubleshooting, and details.

## 📁 Project Structure

```
SushiVoice/
├── data/
│   ├── sushi_items_20.txt          # 20 sushi items
│   ├── synthetic_20/               # Generated TTS data (400 files)
│   └── my_voice_20/                # Your recordings (20 files)
├── models/
│   └── transformer_20_best.pth     # Trained model
├── src/
│   └── inference/
│       ├── parser.py               # Command parser
│       └── printer.py              # Label printer
├── generate_training_data_20.py    # Generate synthetic data
├── record_20_human_voice.py        # Record your voice
├── train_transformer_20.py         # Train model
├── demo_transformer_voice.py       # Test/run model
└── transformer_audio_slm.py        # Core Transformer SLM
```

## 🎤 Usage Example

```bash
# Run demo
python3 demo_transformer_voice.py

# Press ENTER, then say:
"print five chicken teriyaki"

# Output:
📝 Transcript: 'print five chicken teriyaki'
🎯 Confidence: 87.5%
✅ Parsed: 5x chicken teriyaki

📄 PRINTING LABELS:
========================================
  Chicken Teriyaki
  Chicken Teriyaki
  Chicken Teriyaki
  Chicken Teriyaki
  Chicken Teriyaki
========================================
✅ 5 label(s) printed
```

## 🔧 Troubleshooting

**Low accuracy?**
- Record more samples (2-3 takes per item)
- Train longer (`--epochs 50`)
- Check microphone quality

**Model not found?**
```bash
ls models/transformer_20_best.pth
# If missing, retrain:
python3 train_transformer_20.py --epochs 30
```

**PyAudio errors?**
```bash
# Mac
brew install portaudio && pip install pyaudio

# Linux
sudo apt-get install portaudio19-dev && pip install pyaudio
```

## 🚀 Next Steps

1. **Add more items:** Update `data/sushi_items_20.txt`, regenerate data, retrain
2. **Improve accuracy:** Record 2-3 samples per item instead of 1
3. **Real printer:** Connect thermal printer (see `src/inference/printer.py`)
4. **Deploy:** Run on Raspberry Pi or edge device

## 📊 Performance

- **Model size:** 18.5M parameters (~70MB)
- **Inference time:** ~60ms per audio (CPU)
- **Accuracy:** 70-85% with 20 human samples, 10-20% without
- **Training time:** 10-20 minutes on CPU for 30 epochs

## 📝 License

Custom built for YoSushi voice-controlled label printing.

## 🙏 Credits

Built from scratch using PyTorch, Transformers, and love for sushi! 🍣
