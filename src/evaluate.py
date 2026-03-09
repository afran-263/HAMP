"""
evaluate.py - Evaluate HAMP on L2-Arctic or LibriSpeech test sets.
Computes Phoneme Error Rate (PER) using substitutions, deletions, insertions.

Usage:
    python src/evaluate.py \
        --model_path saved1/best_model.pth \
        --embedding_dir data/l2arctic_embeddings/layer_6 \
        --alignment_dir data/l2arctic_alignments
"""

import os
import argparse
import torch
import numpy as np
import json
from tqdm import tqdm
from collections import defaultdict
import editdistance

# Import model and feature definitions from training script
import sys
sys.path.insert(0, os.path.dirname(__file__))
from train import (
    RoutedPhonemeClassifier, Config,
    CONSONANT_FEATURES, VOWEL_FEATURES, SILENCE_MARKERS,
    ALL_MOA_CLASSES, ALL_POA_CLASSES, ALL_VOICING_CLASSES,
    ALL_HEIGHT_CLASSES, ALL_BACKNESS_CLASSES, ALL_ROUNDING_CLASSES,
    strip_stress_markers
)


# ============================================================================
# Articulatory-to-Phoneme Lookup Table
# ============================================================================

def build_articulatory_to_phoneme_map():
    """Build a reverse lookup: articulatory tuple -> phoneme label."""
    lookup = {}
    for phoneme, (moa, poa, voicing) in CONSONANT_FEATURES.items():
        lookup[(moa, poa, voicing)] = phoneme
    for phoneme, (height, backness, rounding) in VOWEL_FEATURES.items():
        lookup[(height, backness, rounding)] = phoneme
    return lookup


def predict_phoneme_from_articulatory(
    pred_category,   # 'consonant', 'vowel', 'silence'
    pred_moa=None, pred_poa=None, pred_voicing=None,
    pred_height=None, pred_backness=None, pred_rounding=None,
    artic_map=None
):
    """Convert predicted articulatory attributes to a phoneme label."""
    if pred_category == 'silence':
        return 'SIL'
    elif pred_category == 'consonant':
        key = (pred_moa, pred_poa, pred_voicing)
    else:  # vowel
        key = (pred_height, pred_backness, pred_rounding)

    return artic_map.get(key, 'UNK')


# ============================================================================
# PER Calculation
# ============================================================================

def compute_per(ref_sequences, hyp_sequences):
    """
    Compute Phoneme Error Rate across a list of reference/hypothesis sequences.
    PER = (S + D + I) / N  where N = total reference phonemes.
    """
    total_ref = 0
    total_edits = 0
    total_sub = total_del = total_ins = 0

    for ref, hyp in zip(ref_sequences, hyp_sequences):
        total_ref += len(ref)
        total_edits += editdistance.eval(ref, hyp)
        # Simplified counts (full alignment would need dynamic programming)

    per = total_edits / max(total_ref, 1) * 100
    return per


# ============================================================================
# Inference Loop
# ============================================================================

def run_inference(model, embedding_dir, alignment_dir, device, config):
    """Run frame-level inference and decode phoneme sequences."""
    model.eval()
    artic_map = build_articulatory_to_phoneme_map()

    emb_files = sorted([f for f in os.listdir(embedding_dir) if f.endswith('.npy')])
    ref_sequences = []
    hyp_sequences = []

    with torch.no_grad():
        for emb_file in tqdm(emb_files, desc="Evaluating"):
            utt_id = emb_file.replace('.npy', '')
            align_file = os.path.join(alignment_dir, utt_id + '.txt')
            emb_path = os.path.join(embedding_dir, emb_file)

            if not os.path.exists(align_file):
                continue

            embeddings = np.load(emb_path)  # (T, 768)
            embeddings = torch.tensor(embeddings, dtype=torch.float32).to(device)

            # Load ground-truth phoneme labels
            with open(align_file) as f:
                lines = f.read().strip().split('\n')
            ref_phonemes = [strip_stress_markers(l.split()[-1]) for l in lines if l.strip()]

            # Forward pass
            outputs = model(embeddings.unsqueeze(0))

            # Decode predictions frame-by-frame
            hyp_phonemes = []
            T = embeddings.shape[0]
            for t in range(T):
                sil_prob = torch.sigmoid(outputs['silence'][0, t]).item()
                c_prob = torch.sigmoid(outputs['consonant'][0, t]).item()
                v_prob = torch.sigmoid(outputs['vowel'][0, t]).item()

                if sil_prob > 0.5:
                    category = 'silence'
                elif c_prob > v_prob:
                    category = 'consonant'
                else:
                    category = 'vowel'

                if category == 'silence':
                    phoneme = 'SIL'
                elif category == 'consonant':
                    moa = ALL_MOA_CLASSES[outputs['moa'][0, t].argmax().item()]
                    poa = ALL_POA_CLASSES[outputs['poa'][0, t].argmax().item()]
                    voicing = 'voiced' if torch.sigmoid(outputs['voicing'][0, t]).item() > 0.5 else 'unvoiced'
                    phoneme = artic_map.get((moa, poa, voicing), 'UNK')
                else:
                    height = ALL_HEIGHT_CLASSES[outputs['height'][0, t].argmax().item()]
                    backness = ALL_BACKNESS_CLASSES[outputs['backness'][0, t].argmax().item()]
                    rounding = 'rounded' if torch.sigmoid(outputs['rounding'][0, t]).item() > 0.5 else 'unrounded'
                    phoneme = artic_map.get((height, backness, rounding), 'UNK')

                hyp_phonemes.append(phoneme)

            ref_sequences.append(ref_phonemes)
            hyp_sequences.append(hyp_phonemes)

    return ref_sequences, hyp_sequences


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate HAMP model")
    parser.add_argument('--model_path', type=str, required=True, help='Path to saved model checkpoint')
    parser.add_argument('--embedding_dir', type=str, required=True, help='Directory with .npy HuBERT embeddings')
    parser.add_argument('--alignment_dir', type=str, required=True, help='Directory with .txt alignment files')
    parser.add_argument('--output_file', type=str, default='results/eval_results.json')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load checkpoint
    checkpoint = torch.load(args.model_path, map_location=device)
    config = checkpoint.get('config', Config())

    model = RoutedPhonemeClassifier(
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        dropout=0.0  # No dropout at inference
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint.get('epoch', '?')}")

    # Run inference
    ref_seqs, hyp_seqs = run_inference(model, args.embedding_dir, args.alignment_dir, device, config)

    # Compute PER
    per = compute_per(ref_seqs, hyp_seqs)
    print(f"\nPER: {per:.2f}%")
    print(f"Evaluated on {len(ref_seqs)} utterances")

    # Save results
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump({'per': per, 'num_utterances': len(ref_seqs)}, f, indent=2)
    print(f"Results saved to {args.output_file}")


if __name__ == '__main__':
    main()
