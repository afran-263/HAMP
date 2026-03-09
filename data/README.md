# Data

This directory contains dataset preparation instructions. Do NOT upload large audio datasets to this repository.

---

## Datasets Used

### 1. LibriSpeech (Training)
- **Subset**: `train-clean-100` (~21 hours used; ~100 hours total)
- **Download**: https://www.openslr.org/12
- **Validation/Test**: `dev-clean` and `test-clean` subsets

```bash
wget https://www.openslr.org/resources/12/train-clean-100.tar.gz
wget https://www.openslr.org/resources/12/dev-clean.tar.gz
wget https://www.openslr.org/resources/12/test-clean.tar.gz
tar -xzf train-clean-100.tar.gz
```

### 2. L2-Arctic (Zero-shot Evaluation)
- **Source**: https://psi.engr.tamu.edu/l2-arctic-corpus/
- **Details**: 24 non-native English speakers (Hindi, Korean, Mandarin, Spanish, Arabic, Vietnamese), ~1 hour per speaker, 3599 utterances total
- **Annotations**: Phoneme-level annotations included

---

## Forced Alignment

Phoneme boundaries are required for frame-level articulatory training.

1. Install Kaldi: https://kaldi-asr.org/doc/install.html
2. Run alignment on LibriSpeech using the standard LibriSpeech recipe:
```bash
# Inside kaldi/egs/librispeech/s5
bash run.sh
```
3. Export alignment text files (one per utterance) to your `alignment_dir`.

Each `.txt` alignment file should be in the format:
```
<phoneme> <start_frame> <end_frame>
SIL 0 5
HH 6 12
...
```

---

## HuBERT Embedding Extraction

After forced alignment, extract HuBERT embeddings using:

```bash
python scripts/extract_hubert.py \
    --audio_dir /path/to/librispeech/train-clean-100 \
    --output_dir /path/to/embeddings \
```

Embeddings are saved as `.npy` files (one per utterance), with shape `[T, 768]` where `T` is the number of 20ms frames.

---

## Expected Directory Layout

```
/path/to/embeddings/
    ├── layer_0/
    |   ├── 1234-5678-0001.npy
    |   ├── 1234-5678-0002.npy
    |   └── ...
    ├── layer_1/
    |   ├── 1234-5678-0001.npy
    |   ├── 1234-5678-0002.npy
    |   └── ...
    .
    .
    .
    .
    └── layer_12/
    ├── 1234-5678-0001.npy
    ├── 1234-5678-0002.npy
    └── ...

/path/to/alignment/txt/
    ├── 1234-5678-0001.txt
    ├── 1234-5678-0002.txt
    └── ...
```

Update `embedding_dir` and `alignment_dir` in `src/train.py` accordingly.
