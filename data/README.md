# Data

This directory stores dataset information, embeddings, and alignments.
**Do not commit large audio files or raw datasets to this repository.**

## Datasets Used

### LibriSpeech
- Used for training and validation
- Subset: train-clean-100 (~21 hours, 251 speakers)
- Validation: dev-clean
- Test: test-clean
- Download: https://www.openslr.org/12

### L2-Arctic
- Used for zero-shot MDD evaluation
- 24 non-native speakers (Hindi, Korean, Mandarin, Spanish, Arabic, Vietnamese)
- 3599 utterances total with phoneme-level annotations
- Download: https://psi.engr.tamu.edu/l2-arctic-corpus/

## Expected Directory Layout

After preprocessing, your local `data/` folder should look like:

```
data/
├── embeddings/
│   └── layer_6/         # HuBERT layer-6 .npy files (one per utterance)
│       ├── 1272-128104-0000.npy
│       └── ...
├── l2arctic_embeddings/
│   └── layer_6/         # Same format for L2-Arctic
│       └── ...
├── alignments/          # Forced-alignment .txt files (LibriSpeech)
│   ├── 1272-128104-0000.txt
│   └── ...
└── l2arctic_alignments/ # Forced-alignment .txt files (L2-Arctic)
    └── ...
```

## Alignment File Format

Each `.txt` file contains one line per HuBERT frame (20ms):

```
0 1 SIL
1 2 SIL
2 3 HH
3 4 HH
4 5 AH
...
```

Format: `<start_frame> <end_frame> <phoneme>`

## Preprocessing Steps

1. Download LibriSpeech and L2-Arctic
2. Run forced alignment (see `scripts/forced_align.py`)
3. Extract HuBERT embeddings (see `scripts/extract_hubert.py`)
4. Update paths in `src/train.py` Config class

## Notes

- HuBERT model used: `facebook/hubert-base-ls960` (frozen during training)
- Embedding layer: Layer 6 (captures acoustic/articulatory features)
- Frame shift: 20ms
- Phoneme set: CMU ARPAbet (stress markers stripped)
