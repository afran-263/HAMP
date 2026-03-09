# HAMP: Hierarchical Articulatory Phoneme Modeling

A hierarchical articulatory phoneme modeling framework for zero-shot Mispronunciation Detection and Diagnosis (MDD) using HuBERT representations.

## Overview

HAMP predicts phonemes through structured hierarchical learning of articulatory properties, rather than mapping acoustic features directly to phoneme labels. This improves robustness on non-native speech by exploiting shared articulatory structure.

- **Consonants**: Manner of Articulation (MOA) → Place of Articulation (POA) → Voicing
- **Vowels**: Height → Backness → Roundedness
- **Trained on**: LibriSpeech (train-clean-100, ~21 hours)
- **Evaluated on**: L2-Arctic (zero-shot MDD, 3599 utterances)
- **Result**: 18% PER on L2-Arctic

## Repository Structure

```
HAMP/
├── README.md
├── requirements.txt
├── src/
│   ├── train.py              # Main training script (HAMP model + data pipeline)
│   └── evaluate.py           # Evaluation on L2-Arctic / PER computation
├── scripts/
│   ├── extract_hubert.py     # Extract HuBERT frame embeddings from audio
│   └── forced_align.py       # Forced alignment using Kaldi to get frame labels
├── data/
│   └── README.md             # Dataset download and preprocessing instructions
├── results/
│   └── sample_outputs.md     # Sample PER results and error analysis
└── docs/
    └── paper.pdf             # HAMP paper (Interspeech 2026 submission)
```

## Installation

```bash
pip install -r requirements.txt
```

## Data Preparation

### 1. Download Datasets

- **LibriSpeech** (train-clean-100, dev-clean, test-clean): https://www.openslr.org/12
- **L2-Arctic**: https://psi.engr.tamu.edu/l2-arctic-corpus/

Do NOT commit large audio files to the repo. See `data/README.md` for full instructions.

### 2. Extract HuBERT Embeddings

```bash
python scripts/extract_hubert.py \
  --audio_dir /path/to/librispeech/train-clean-100 \
  --output_dir data/embeddings/layer_6 \
  --layer 6
```

### 3. Run Forced Alignment (Kaldi)

```bash
python scripts/forced_align.py \
  --audio_dir /path/to/librispeech/train-clean-100 \
  --output_dir data/alignments
```

## Training

Update the paths in `src/train.py` (Config class):

```python
embedding_dir = "data/embeddings/layer_6"
alignment_dir = "data/alignments"
```

Then run:

```bash
python src/train.py
```

Model checkpoints are saved to `saved1/best_model.pth`.

## Evaluation

```bash
python src/evaluate.py \
  --model_path saved1/best_model.pth \
  --embedding_dir data/l2arctic_embeddings/layer_6 \
  --alignment_dir data/l2arctic_alignments
```

## Results

| Model | PER (%) |
|---|---|
| HuBERT Finetuned | 35.00 |
| Articulatory-to-Acoustic Inversion | 25.20 |
| Wav2Vec2 Momentum Pseudo-Labeling | 14.36 |
| **HAMP (Proposed)** | **18.00** |

## Citation

```
@inproceedings{hamp2026,
  title={HAMP: A Hierarchical Articulatory Phoneme Modeling Framework for Zero-shot Mispronunciation Detection and Diagnosis with HuBERT},
  booktitle={Interspeech 2026},
  year={2026}
}
```
