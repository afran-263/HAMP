# HAMP Architecture Documentation

## System Overview

HAMP (Hierarchical Articulatory Phoneme Modeling) performs frame-level phoneme recognition
by predicting articulatory attributes hierarchically, rather than predicting phoneme labels directly.

## Components

### 1. Acoustic Representation (HuBERT)

- Model: `facebook/hubert-base-ls960` (frozen)
- Input: raw waveform
- Output: frame-level embeddings H ∈ R^(T×768)
- Frame shift: 20ms
- Layer used: Layer 6 (empirically selected; encodes acoustic + articulatory features)

A shared feed-forward projection maps each frame embedding h_t to a shared representation z_t.

### 2. Primary Phonetic Category Prediction

Three binary classifiers predict the phonetic category of each frame:
- Silence (SIL/SP/SPN)
- Consonant
- Vowel

Each is trained independently with Binary Cross-Entropy (BCE) loss.

### 3. Consonant Hierarchy

```
MOA (Manner of Articulation)
├── stops        → POA: bilabial, alveolar, velar
│                   → Voicing: voiced/unvoiced
├── fricatives   → POA: labiodental, dental, alveolar, postalveolar, glottal
│                   → Voicing: voiced/unvoiced
├── affricates   → POA: postalveolar
│                   → Voicing: voiced/unvoiced
├── nasals       → POA: bilabial, alveolar, velar
│                   → Voicing: voiced (always)
└── approximants → POA: alveolar, postalveolar, labiovelar, palatal
                    → Voicing: voiced (always)
```

- MOA: multi-class CE loss (5 classes)
- POA: per-MOA multi-class CE loss (9 classes total)
- Voicing: binary BCE loss

### 4. Vowel Hierarchy

```
Height (tongue vertical position)
├── high   → Backness: front, back
│              → Rounding: rounded/unrounded
├── mid    → Backness: front, central, back
│              → Rounding: rounded/unrounded
└── low    → Backness: front, central, back
               → Rounding: unrounded (only)
```

- Height: multi-class BCE (3 classes)
- Backness: per-height multi-class CE (3 classes)
- Rounding: binary BCE

### 5. Training Objective

```
L = λ_phn * L_phn
  + λ_moa * L_moa + λ_poa * L_poa + λ_voice * L_voice    (consonant)
  + λ_height * L_height + λ_back * L_back + λ_round * L_round  (vowel)
```

All loss weights (λ) = 1.0.

### 6. Hierarchical Routing

**During training**: Ground-truth articulatory labels determine routing. Each classifier is
trained only on frames belonging to its branch (e.g., POA classifiers only see consonant frames).

**During inference**: Predicted outputs from each stage determine routing decisions.

## Phoneme Reconstruction

Predicted articulatory tuples are mapped back to phoneme labels using a reverse lookup table
built from the CMU phoneme definitions (see `CONSONANT_FEATURES` and `VOWEL_FEATURES` in `src/train.py`).

## Training Details

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 1e-3 |
| Batch size | 1000 frames |
| Dropout | 0.4 |
| Hidden dim | 512 |
| Early stopping patience | 15 epochs |
| LR scheduler | ReduceLROnPlateau (factor=0.5, patience=5) |
| Label smoothing | 0.1 |
| Mixup alpha | 0.2 |
| Gradient clipping | 1.0 |
