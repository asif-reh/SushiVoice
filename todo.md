# SushiVoice - TODO Workflow

## 🚀 Complete 4-Step Workflow (~45 minutes)

### Step 1: Generate synthetic data (5 min)
```bash
python3 generate_training_data_20.py
```
Creates 400 synthetic TTS audio samples (20 per sushi item)

---

### Step 2: Record your voice (20 min)
```bash
python3 record_20_human_voice.py
```
Records 20 voice samples - one for each sushi item
Say: "print five [item name]"

---

### Step 3: Train model (15 min)
```bash
python3 train_transformer_20.py --epochs 30
```
Trains Transformer on 420 samples (400 synthetic + 20 human)
Model saves to: `models/transformer_20_best.pth`

---

### Step 4: Test it!
```bash
python3 demo_transformer_voice.py
```
Run real-time voice recognition and label printing
Say: "print 5 chicken teriyaki" → See 5 labels printed!

---

## ✅ Success Criteria

- [ ] Generated 400 synthetic samples
- [ ] Recorded 20 human voice samples
- [ ] Trained model for 30 epochs
- [ ] Loss decreased to < 3.0
- [ ] Can transcribe voice with 70%+ accuracy
- [ ] Labels print correctly

## 🎯 Expected Results

**Without human voice:** 10-20% accuracy (gibberish)
**With 20 human samples:** 70-85% accuracy (working!)

---

**Total Time:** ~45 minutes
**End Result:** Working voice-controlled sushi label printer! 🍣🎤🖨️