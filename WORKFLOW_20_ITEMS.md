# SushiVoice - 20 Item Workflow

Clean, simple workflow for training voice recognition on 20 sushi items with your real voice!

## 🎯 Goal

Train a Transformer model to recognize YOUR voice saying commands like:
- "print 5 chicken teriyaki" → Outputs 5 labels
- "hey yosushi label 3 salmon nigiri" → Outputs 3 labels

## 📋 20 Sushi Items

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

## 🚀 Complete Workflow

### Step 1: Generate Synthetic Training Data (5 min)

Creates 400 synthetic TTS audio samples (20 samples per item):

```bash
python3 generate_training_data_20.py
```

**Output:** `data/synthetic_20/` with 400 .wav files

### Step 2: Record Your Voice (20 min)

Record 20 voice samples - one for each sushi item:

```bash
python3 record_20_human_voice.py
```

**Instructions:**
- Say: "print five [item name]"
- Example: "print five chicken teriyaki"
- Take breaks every 5 samples
- Speak naturally and clearly

**Output:** `data/my_voice_20/` with 20 .wav files

### Step 3: Train the Model (10-20 min)

Train Transformer on 420 total samples (400 synthetic + 20 human):

```bash
python3 train_transformer_20.py --epochs 30
```

**What happens:**
- Loads synthetic + human voice samples
- Trains Transformer model (18.5M parameters)
- Saves best model to `models/transformer_20_best.pth`

**Expected results:**
- Loss should decrease from ~10 to ~2-3
- Training time: 10-20 minutes on CPU

### Step 4: Test It! (Real-time)

Run real-time voice recognition:

```bash
python3 demo_transformer_voice.py
```

**Usage:**
1. Press ENTER to record
2. Say: "print five chicken teriyaki"
3. Wait 5 seconds (auto-recording)
4. See transcription and labels printed!

**Expected output:**
```
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

## 📊 Expected Accuracy

- **Without human voice:** 10-20% (gibberish)
- **With 20 human samples:** 70-85% (good!)
- **With more training:** Can improve to 90%+

## 🔧 Troubleshooting

### Low accuracy after training?

1. **Record more samples:** Re-record unclear items
2. **Train longer:** Use `--epochs 50`
3. **Check audio quality:** Make sure recordings are clear

### Model not found?

```bash
# Check if model exists
ls -lh models/transformer_20_best.pth

# If not, retrain
python3 train_transformer_20.py --epochs 30
```

### PyAudio issues?

```bash
# Mac
brew install portaudio
pip install pyaudio

# Linux
sudo apt-get install portaudio19-dev
pip install pyaudio
```

## 📁 Project Structure

```
SushiVoice/
├── data/
│   ├── sushi_items_20.txt          # 20 sushi items
│   ├── my_voice_20/                # Your voice recordings (20 files)
│   └── synthetic_20/               # Synthetic TTS data (400 files)
├── models/
│   └── transformer_20_best.pth     # Trained model
├── generate_training_data_20.py    # Step 1: Generate synthetic data
├── record_20_human_voice.py        # Step 2: Record your voice
├── train_transformer_20.py         # Step 3: Train model
└── demo_transformer_voice.py       # Step 4: Test real-time
```

## 🎉 Success Criteria

✅ Generated 400 synthetic samples  
✅ Recorded 20 human voice samples  
✅ Trained model for 30 epochs  
✅ Loss decreased to < 3.0  
✅ Can transcribe your voice with 70%+ accuracy  
✅ Labels print correctly with item name and quantity  

## 🚀 Next Steps

Once this works well:
1. Add more sushi items (expand beyond 20)
2. Record multiple samples per item (2-3 takes)
3. Connect to real thermal printer
4. Deploy to Raspberry Pi or edge device

---

**Total Time:** ~45 minutes  
**End Result:** Working voice-controlled sushi label printer! 🍣🎤🖨️
