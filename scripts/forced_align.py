"""
forced_align.py - Run Kaldi-style forced alignment to get frame-level phoneme labels.

This script assumes you have:
  - MFA (Montreal Forced Aligner) or Kaldi installed, OR
  - Pre-computed TextGrid / CTM alignment files from Kaldi

It converts alignment outputs to per-utterance .txt files in the format:
    <start_frame> <end_frame> <phoneme>

One line per HuBERT frame (20ms windows), matching the embedding files.

Usage (with MFA):
    python scripts/forced_align.py \
        --audio_dir /path/to/audio \
        --transcript_dir /path/to/transcripts \
        --output_dir data/alignments \
        --aligner mfa

Usage (from existing CTM/TextGrid):
    python scripts/forced_align.py \
        --ctm_dir /path/to/kaldi/ctm \
        --output_dir data/alignments \
        --aligner ctm
"""

import os
import argparse
from tqdm import tqdm


FRAME_SHIFT_MS = 20  # HuBERT frame shift in milliseconds


def time_to_frame(time_sec):
    return int(round(time_sec * 1000 / FRAME_SHIFT_MS))


def parse_textgrid(tg_path):
    """Parse a Praat TextGrid file and extract phoneme intervals."""
    phonemes = []
    with open(tg_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Simple line-based parser for standard TextGrid format
    lines = content.split('\n')
    in_phones_tier = False
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if 'phones' in line.lower() or 'phone' in line.lower():
            in_phones_tier = True
        if in_phones_tier and line.startswith('xmin'):
            xmin = float(line.split('=')[1].strip())
            xmax = float(lines[i+1].split('=')[1].strip())
            text = lines[i+2].split('=')[1].strip().strip('"').upper()
            phonemes.append((xmin, xmax, text))
            i += 3
            continue
        i += 1
    return phonemes


def parse_ctm(ctm_path):
    """Parse a Kaldi CTM file: utt_id channel start_time duration phoneme"""
    utterances = {}
    with open(ctm_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            utt_id, _, start, dur, phoneme = parts[:5]
            start, dur = float(start), float(dur)
            if utt_id not in utterances:
                utterances[utt_id] = []
            utterances[utt_id].append((start, start + dur, phoneme.upper()))
    return utterances


def intervals_to_frame_labels(intervals, total_frames=None):
    """Convert (start_sec, end_sec, phoneme) intervals to per-frame labels."""
    if not intervals:
        return []
    max_time = intervals[-1][1]
    if total_frames is None:
        total_frames = time_to_frame(max_time)

    labels = ['SIL'] * total_frames
    for start, end, phoneme in intervals:
        f_start = time_to_frame(start)
        f_end = time_to_frame(end)
        for f in range(f_start, min(f_end, total_frames)):
            labels[f] = phoneme
    return labels


def write_alignment(labels, out_path):
    """Write frame-level phoneme labels to file."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        for i, label in enumerate(labels):
            f.write(f"{i} {i+1} {label}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--aligner', choices=['mfa', 'ctm', 'textgrid'], default='textgrid')
    parser.add_argument('--audio_dir', type=str, default=None)
    parser.add_argument('--transcript_dir', type=str, default=None)
    parser.add_argument('--ctm_dir', type=str, default=None)
    parser.add_argument('--textgrid_dir', type=str, default=None)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.aligner == 'ctm' and args.ctm_dir:
        # Parse all CTM files
        for ctm_file in tqdm(os.listdir(args.ctm_dir), desc="Parsing CTM"):
            if not ctm_file.endswith('.ctm'):
                continue
            utterances = parse_ctm(os.path.join(args.ctm_dir, ctm_file))
            for utt_id, intervals in utterances.items():
                labels = intervals_to_frame_labels(intervals)
                write_alignment(labels, os.path.join(args.output_dir, utt_id + '.txt'))

    elif args.aligner == 'textgrid' and args.textgrid_dir:
        for tg_file in tqdm(os.listdir(args.textgrid_dir), desc="Parsing TextGrid"):
            if not tg_file.endswith('.TextGrid'):
                continue
            utt_id = tg_file.replace('.TextGrid', '')
            intervals = parse_textgrid(os.path.join(args.textgrid_dir, tg_file))
            labels = intervals_to_frame_labels(intervals)
            write_alignment(labels, os.path.join(args.output_dir, utt_id + '.txt'))

    elif args.aligner == 'mfa':
        print("MFA alignment: run MFA externally first, then use --aligner textgrid with the output dir.")
        print("  mfa align <audio_dir> <lexicon> <model> <output_dir>")
        return

    print(f"Alignment files saved to {args.output_dir}")


if __name__ == '__main__':
    main()
