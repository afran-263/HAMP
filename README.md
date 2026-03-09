# HAMP: Hierarchical Articulatory Phoneme Modeling Framework

**Zero-shot Mispronunciation Detection and Diagnosis using HuBERT**

---

## Overview

HAMP is a hierarchical articulatory phoneme modeling framework for zero-shot Mispronunciation Detection and Diagnosis (MDD). Instead of predicting phoneme labels directly, it uses pretrained HuBERT embeddings and predicts phonemes through a structured hierarchy of articulatory properties.

- **Consonants**: Silence/Speech → Consonant/Vowel → MOA → POA → Voicing  
- **Vowels**: Silence/Speech → Consonant/Vowel → Height → Backness → Rounding  

Trained on **LibriSpeech (train-clean-100)** and evaluated on **L2-Arctic** in a zero-shot setting, achieving a **Phoneme Error Rate (PER) of 18%**.

---

## Repository Structure

```
HAMP/
├── README.md               ← This file
├── requirements.txt        ← Python dependencies
├── src/
│   ├── train.py            ← Training script (RoutedPhonemeClassifier + Config)
│   └── inference.py        ← Inference + evaluation script
├── scripts/
│   ├── extract_hubert.py   ← HuBERT embedding extraction script
│   └── evaluate_per.py     ← PER evaluation on L2-Arctic
├── data/
│   └── README.md           ← Dataset download and preparation instructions
├── results/
│   └── sample_outputs/     ← Sample t-SNE plots, PER logs
└── docs/
    └── paper.pdf           ← HAMP paper (Interspeech 2026 submission)
```

---

## Installation

```bash
git clone https://github.com/your-username/HAMP.git
cd HAMP
pip install -r requirements.txt
```

---

## Quickstart

### 1. Prepare Data

See `data/README.md` for instructions on downloading LibriSpeech and L2-Arctic and running forced alignment via Kaldi.

### 2. Extract HuBERT Embeddings

```bash
python scripts/extract_hubert.py \
    --audio_dir /path/to/librispeech \
    --output_dir /path/to/embeddings/layer_6 \
    --layer 6
```

### 3. Train

Edit the paths in `src/train.py` (the `Config` dataclass):

```python
embedding_dir = "/path/to/embeddings/layer_6"
alignment_dir = "/path/to/alignment/txt"
```

Then run:

```bash
python src/train.py
```

Checkpoints are saved to `saved1/best_model.pth` by default.

### 4. Run Inference / Evaluate

```bash
python src/inference.py \
    --model_path saved1/best_model.pth \
    --audio_path /path/to/audio.wav \
    --reference "phoneme sequence here"
```

---

## Model Architecture

| Component | Description |
|---|---|
| Encoder | Pretrained HuBERT-base (frozen), layer 6 embeddings (dim=768) |
| Shared Projection | 2-layer FFN → 512 hidden units, LayerNorm, ReLU, Dropout |
| Stage 1 | Silence vs. Speech binary classifier |
| Stage 2 | Consonant vs. Vowel binary classifier |
| Stage 3a | MOA classifier (5 classes: stops, fricatives, affricates, nasals, approximants) |
| Stage 4a | POA classifier per MOA class (up to 5 locations each) |
| Stage 5a | Voicing classifier per POA class (voiced / unvoiced) |
| Stage 3b | Vowel Height classifier (high / mid / low / diphthongs) |
| Stage 4b | Backness classifier per Height class (front / central / back) |
| Stage 5b | Rounding classifier per Backness class (rounded / unrounded) |

---

## Results

| Model | PER (%) on L2-Arctic |
|---|---|
| HuBERT Finetuned (baseline) | 35.00 |
| Articulatory-to-Acoustic Inversion | 25.20 |
| Wav2Vec2 Momentum Pseudo-Labeling | 14.36 |
| **HAMP (Proposed)** | **18.00** |

---

## Configuration

All training hyperparameters are controlled via the `Config` dataclass in `src/train.py`:

| Parameter | Default | Description |
|---|---|---|
| `input_dim` | 768 | HuBERT embedding dimension |
| `hidden_dim` | 512 | Shared projection hidden size |
| `dropout` | 0.4 | Dropout rate |
| `batch_size` | 256 | Training batch size (frames) |
| `learning_rate` | 1e-4 | AdamW learning rate |
| `early_stopping_patience` | 15 | Epochs to wait before stopping |
| `use_mixup` | True | Mixup data augmentation |
| `label_smoothing_factor` | 0.1 | Label smoothing coefficient |

---
