# 🎤 Complete Audio Dataset Creation Guide for SushiVoice

This guide walks you through creating a high-quality audio dataset for training your custom ASR model.

## Table of Contents
1. [Quick Start (5 minutes)](#quick-start)
2. [Method 1: Synthetic Dataset (Recommended to Start)](#method-1-synthetic-dataset)
3. [Method 2: Recording Real Audio](#method-2-recording-real-audio)
4. [Method 3: Hybrid Approach (Best Results)](#method-3-hybrid-approach)
5. [Dataset Quality Tips](#dataset-quality-tips)
6. [Training Your Model](#training-your-model)

---

## Quick Start

If you have a **text file with sushi names** (one per line), start here:

```bash
cd /Users/code-asif/SushiVoice

# Step 1: Import your sushi names
python src/data/import_sushi_names.py --input /path/to/your/sushi_names.txt

# Step 2: Generate synthetic dataset (1000 samples)
python src/data/generate_dataset.py --num_samples 1000

# Step 3: Augment dataset (creates 3000 total samples)
python src/data/augment_audio.py --aug_per_sample 2

# Done! Dataset is ready for training
```

---

## Method 1: Synthetic Dataset (Recommended to Start)

### Why Synthetic?
- **Fast**: Generate 1000+ samples in 10-15 minutes
- **Diverse**: Multiple voices, accents, and variations
- **Cost-effective**: Free using Google TTS or pyttsx3
- **Good baseline**: Achieves 10-15% WER, perfect for initial testing

### Step-by-Step Process

#### 1.1. Prepare Your Sushi Names

**Option A: You have a text file**
```bash
# Create a file: sushi_names.txt (one item per line)
# Example content:
# Chicken Teriyaki
# Salmon Nigiri
# California Roll
# Spicy Tuna Roll
# ...

python src/data/import_sushi_names.py --input sushi_names.txt
```

**Option B: You have a CSV file**
```bash
# CSV format: first column = sushi name
# item_name,price,category
# Chicken Teriyaki,12.99,Hot Dishes
# Salmon Nigiri,8.99,Nigiri
# ...

python src/data/import_sushi_names.py --input sushi_menu.csv
```

**Option C: Manually edit the vocabulary**
```bash
# Edit the file directly
nano src/data/sushi_vocab.json

# Add your items to the "sushi_items" array:
{
  "sushi_items": [
    "Your Custom Item 1",
    "Your Custom Item 2",
    ...
  ]
}
```

#### 1.2. Generate Synthetic Audio

```bash
cd /Users/code-asif/SushiVoice

# Option A: Using Google TTS (better quality, requires internet)
python src/data/generate_dataset.py --num_samples 1000

# Option B: Using pyttsx3 (offline, faster)
python src/data/generate_dataset.py --num_samples 1000 --use_pyttsx3

# For more samples (recommended 1000-2000 base samples)
python src/data/generate_dataset.py --num_samples 2000
```

**What this does:**
- Generates ~1000 audio files in `data/synthetic/`
- Creates `data/manifest.jsonl` with audio-text mappings
- Uses different voice variations and command templates
- Takes ~10-15 minutes for 1000 samples

#### 1.3. Augment the Dataset

```bash
# This will create 2 augmented versions of each sample
python src/data/augment_audio.py --aug_per_sample 2

# For more variations (3x = 3000 total samples)
python src/data/augment_audio.py --aug_per_sample 3
```

**Augmentations applied:**
- ✅ Kitchen noise (pink noise simulating restaurant background)
- ✅ Speed variations (0.9x - 1.1x)
- ✅ Pitch shifts (±2 semitones)
- ✅ Room reverb
- ✅ Gaussian noise

**Output:**
- Augmented files in `data/augmented/`
- Updated manifest: `data/manifest_augmented.jsonl`
- Total samples: 1000 original + 2000 augmented = **3000 samples**

#### 1.4. Verify Your Dataset

```bash
# Check dataset statistics
python -c "
import json
with open('data/manifest_augmented.jsonl', 'r') as f:
    samples = [json.loads(line) for line in f]
print(f'Total samples: {len(samples)}')
print(f'Average duration: {sum(s.get(\"duration\", 0) for s in samples) / len(samples):.2f}s')
print(f'Sample texts:')
for s in samples[:5]:
    print(f'  - {s[\"text\"]}')
"
```

---

## Method 2: Recording Real Audio

For **best accuracy** (5-10% WER), add real recordings from actual users.

### 2.1. Recording Setup

**Equipment:**
- USB microphone or smartphone
- Quiet room (or noisy kitchen for realism)
- Audacity (free audio editor) or smartphone voice recorder

**Recording Guidelines:**
- 📱 **Sample rate**: 16kHz (or higher, will be downsampled)
- 🎤 **Format**: WAV or MP3
- ⏱️ **Duration**: 2-5 seconds per command
- 👥 **Speakers**: 3-5 different people (diverse accents/genders)
- 🔊 **Volume**: Clear but natural speaking volume

### 2.2. Recording Script

Create a recording script to stay organized:

```bash
# Create recording script
cat > recording_script.txt << 'EOF'
# Record each line 1-2 times with natural variations

Hey YoSushi print 5 times of label of Chicken Teriyaki
YoSushi label 3 times California Roll
Print 10 labels of Salmon Nigiri
Hey YoSushi make 2 labels for Spicy Tuna Roll
Label 7 times Tempura Shrimp
Print 1 label of Dragon Roll
Hey YoSushi print 15 times Rainbow Roll
... (add more based on your vocabulary)
EOF
```

### 2.3. Recording Process

**Using Audacity (Mac):**
1. Install: `brew install audacity`
2. Set sample rate to 16000 Hz
3. Record each command
4. File → Export → Export as WAV
5. Save to `SushiVoice/data/raw/speaker1_001.wav`, etc.

**Using Smartphone:**
1. Use voice recorder app
2. Record in a quiet place
3. Transfer files to `data/raw/`
4. Convert to WAV if needed:
   ```bash
   # Install ffmpeg if needed
   brew install ffmpeg
   
   # Convert all MP3 to WAV
   for f in data/raw/*.mp3; do
     ffmpeg -i "$f" -ar 16000 -ac 1 "${f%.mp3}.wav"
   done
   ```

### 2.4. Annotate Real Recordings

```bash
# Create annotation file
python -c "
import json
import os
from pathlib import Path

# List all audio files in data/raw/
audio_files = list(Path('data/raw').glob('*.wav'))

# Create manifest entries (you'll need to add transcripts manually)
with open('data/raw_manifest.jsonl', 'w') as f:
    for audio_file in audio_files:
        entry = {
            'audio': str(audio_file),
            'text': 'REPLACE_WITH_TRANSCRIPT',  # Edit this manually
            'speaker': 'speaker1'  # Add speaker ID
        }
        f.write(json.dumps(entry) + '\n')

print(f'Created manifest for {len(audio_files)} files')
print('Edit data/raw_manifest.jsonl and replace REPLACE_WITH_TRANSCRIPT with actual text')
"

# Now manually edit data/raw_manifest.jsonl with correct transcripts
nano data/raw_manifest.jsonl
```

### 2.5. Merge with Synthetic Dataset

```bash
# Combine synthetic and real audio manifests
python -c "
import json

# Read both manifests
synthetic = []
with open('data/manifest_augmented.jsonl', 'r') as f:
    synthetic = [json.loads(line) for line in f]

real = []
with open('data/raw_manifest.jsonl', 'r') as f:
    real = [json.loads(line) for line in f]

# Combine
combined = synthetic + real

# Save combined manifest
with open('data/manifest_final.jsonl', 'w') as f:
    for entry in combined:
        f.write(json.dumps(entry) + '\n')

print(f'Combined dataset: {len(synthetic)} synthetic + {len(real)} real = {len(combined)} total')
"
```

---

## Method 3: Hybrid Approach (Best Results)

**Recommended workflow for production:**

1. **Phase 1: Quick Start (Day 1)**
   - Generate 1000 synthetic samples
   - Augment to 3000 samples
   - Train initial model (WER ~15%)

2. **Phase 2: Real Data Collection (Days 2-7)**
   - Record 200-500 real samples from 3-5 speakers
   - Include kitchen noise recordings
   - Mix of clear and noisy conditions

3. **Phase 3: Fine-tuning (Day 8)**
   - Combine synthetic (3000) + real (300) = 3300 samples
   - Retrain model from scratch or fine-tune
   - Expected WER: 5-10%

4. **Phase 4: Active Learning (Ongoing)**
   - Deploy model to test environment
   - Log failures and low-confidence predictions
   - Record corrections and add to dataset
   - Retrain monthly

---

## Dataset Quality Tips

### ✅ Do's
- **Diverse commands**: Use all template variations
- **Natural speech**: Include pauses, "um", slight mispronunciations
- **Background noise**: Add realistic kitchen sounds
- **Multiple speakers**: Different genders, ages, accents
- **Edge cases**: Test with loud music, multiple voices, far-field audio

### ❌ Don'ts
- Don't use only one TTS voice
- Don't record in perfect silence (unrealistic)
- Don't use only formal command structures
- Don't ignore failed predictions during testing
- Don't overtrain on synthetic data alone

### Dataset Size Guidelines

| Purpose | Synthetic | Real | Total | Expected WER |
|---------|-----------|------|-------|--------------|
| Proof of Concept | 500 | 0 | 500 | 20-25% |
| **Initial Deployment** | **1000** | **0** | **3000** (with aug) | **10-15%** |
| Production | 2000 | 300 | 5000+ | 5-10% |
| High Accuracy | 3000 | 500 | 8000+ | 3-5% |

---

## Training Your Model

Once your dataset is ready:

```bash
cd /Users/code-asif/SushiVoice

# Train with default settings
python src/models/train_custom_slm.py \
  --manifest data/manifest_augmented.jsonl \
  --epochs 20 \
  --batch_size 16

# Monitor training
tensorboard --logdir runs/

# Training takes:
# - Mac M1/M2: 2-4 hours
# - Google Colab (free T4): 1-2 hours
```

---

## Next Steps After Dataset Creation

1. **Train model**: Follow training guide above
2. **Test inference**: Try with test audio files
3. **Deploy to Pi**: Use quantized model for edge deployment
4. **Collect feedback**: Log production usage for retraining

---

## Troubleshooting

**Problem: TTS generates low-quality audio**
- Solution: Use gTTS (Google TTS) instead of pyttsx3
- Command: `python src/data/generate_dataset.py --num_samples 1000` (default uses gTTS)

**Problem: Augmentation takes too long**
- Solution: Reduce augmentations per sample
- Command: `python src/data/augment_audio.py --aug_per_sample 1`

**Problem: Model doesn't recognize my sushi items**
- Solution: Check vocabulary includes all items
- Verify: `cat src/data/sushi_vocab.json | grep "your_item"`

**Problem: High WER on validation set**
- Solution: Add more real recordings or increase dataset size
- Target: At least 2000-3000 samples for decent accuracy

---

## Quick Reference Commands

```bash
# Import custom sushi names
python src/data/import_sushi_names.py --input sushi_names.txt

# Generate 1000 synthetic samples
python src/data/generate_dataset.py --num_samples 1000

# Augment dataset (3x multiplier)
python src/data/augment_audio.py --aug_per_sample 2

# Check dataset size
ls -l data/synthetic/*.wav | wc -l
ls -l data/augmented/*.wav | wc -l

# Train model
python src/models/train_custom_slm.py --manifest data/manifest_augmented.jsonl --epochs 20

# Test single audio file
python src/inference/asr_pipeline.py --model models/sushi_slm_best.pth --test_audio data/synthetic/sample_00001.wav
```

---

**Ready to build your dataset? Start with Method 1 (Synthetic) and add real recordings later for best results!** 🚀
