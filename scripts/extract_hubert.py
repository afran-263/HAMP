"""
extract_hubert.py - Extract HuBERT frame-level embeddings from audio files.

Saves one .npy file per utterance with shape (T, D) where:
  T = number of 20ms frames
  D = 768 (HuBERT base) or 1024 (HuBERT large)

Usage:
    python scripts/extract_hubert.py \
        --audio_dir /path/to/librispeech/train-clean-100 \
        --output_dir data/embeddings/layer_6 \
        --layer 6 \
        --model_name facebook/hubert-base-ls960
"""

import os
import argparse
import numpy as np
import torch
import torchaudio
from tqdm import tqdm
from transformers import HubertModel, Wav2Vec2FeatureExtractor


def load_audio(path, target_sr=16000):
    waveform, sr = torchaudio.load(path)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform.squeeze(0)


def extract_embeddings(model, feature_extractor, audio_path, layer, device):
    waveform = load_audio(audio_path)
    inputs = feature_extractor(waveform.numpy(), sampling_rate=16000, return_tensors="pt")
    input_values = inputs.input_values.to(device)

    with torch.no_grad():
        outputs = model(input_values, output_hidden_states=True)

    # Extract embeddings from specified layer
    hidden = outputs.hidden_states[layer]  # (1, T, D)
    return hidden.squeeze(0).cpu().numpy()  # (T, D)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--layer', type=int, default=6, help='HuBERT layer index (0-12 for base)')
    parser.add_argument('--model_name', type=str, default='facebook/hubert-base-ls960')
    parser.add_argument('--ext', type=str, default='.flac', help='Audio file extension')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading HuBERT model: {args.model_name}")
    
    model = HubertModel.from_pretrained(args.model_name).to(device)
    model.eval()
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.model_name)

    # Walk through audio directory
    audio_files = []
    for root, _, files in os.walk(args.audio_dir):
        for f in files:
            if f.endswith(args.ext):
                audio_files.append(os.path.join(root, f))

    print(f"Found {len(audio_files)} audio files")

    for audio_path in tqdm(audio_files, desc="Extracting embeddings"):
        utt_id = os.path.splitext(os.path.basename(audio_path))[0]
        out_path = os.path.join(args.output_dir, utt_id + '.npy')

        if os.path.exists(out_path):
            continue

        try:
            emb = extract_embeddings(model, feature_extractor, audio_path, args.layer, device)
            np.save(out_path, emb)
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")

    print(f"Embeddings saved to {args.output_dir}")


if __name__ == '__main__':
    main()
