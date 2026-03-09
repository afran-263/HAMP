import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List
import librosa
from transformers import HubertModel, Wav2Vec2FeatureExtractor
import matplotlib.pyplot as plt
from collections import defaultdict


CONSONANT_FEATURES = {
    "P": ("stops", "bilabial", "unvoiced"),
    "B": ("stops", "bilabial", "voiced"),
    "T": ("stops", "alveolar", "unvoiced"),
    "D": ("stops", "alveolar", "voiced"),
    "K": ("stops", "velar", "unvoiced"),
    "G": ("stops", "velar", "voiced"),
    
    "M": ("nasals", "bilabial", "voiced"),
    "N": ("nasals", "alveolar", "voiced"),
    "NG": ("nasals", "velar", "voiced"),
    
    "F": ("fricatives", "labiodental", "unvoiced"),
    "V": ("fricatives", "labiodental", "voiced"),
    "TH": ("fricatives", "dental", "unvoiced"),
    "DH": ("fricatives", "dental", "voiced"),
    "S": ("fricatives", "alveolar", "unvoiced"),
    "Z": ("fricatives", "alveolar", "voiced"),
    "SH": ("fricatives", "postalveolar", "unvoiced"),
    "ZH": ("fricatives", "postalveolar", "voiced"),
    "HH": ("fricatives", "glottal", "unvoiced"),
    
    "CH": ("affricates", "postalveolar", "unvoiced"),
    "JH": ("affricates", "postalveolar", "voiced"),
    
    "L": ("approximants", "alveolar", "voiced"),
    "R": ("approximants", "postalveolar", "voiced"),
    
    "W": ("approximants", "labiovelar", "voiced"),
    "Y": ("approximants", "palatal", "voiced"),
}

VOWEL_FEATURES = {
    "IY": ("high", "front", "unrounded"),
    "IH": ("high", "front", "unrounded"),
    "UW": ("high", "back", "rounded"),
    "UH": ("high", "back", "rounded"),
    
    "EH": ("mid", "front", "unrounded"),
    "EY": ("mid", "front", "unrounded"),
    "ER": ("mid", "central", "unrounded"),
    "OW": ("mid", "back", "rounded"),
    "AO": ("mid", "back", "rounded"),
    "AX": ("mid", "central", "unrounded"),
    
    "AH": ("low", "central", "unrounded"),
    "AE": ("low", "front", "unrounded"),
    "AA": ("low", "back", "unrounded"),
    
    "AY": ("diphthongs", "front", "unrounded"),
    "AW": ("diphthongs", "back", "rounded"),
    "OY": ("diphthongs", "back", "rounded"),
}

# Create unified class lists (sorted for consistency)
ALL_MOA_CLASSES = sorted(list(set(f[0] for f in CONSONANT_FEATURES.values())))
ALL_POA_CLASSES = sorted(list(set(f[1] for f in CONSONANT_FEATURES.values())))
ALL_VOICING_CLASSES = sorted(list(set(f[2] for f in CONSONANT_FEATURES.values())))
ALL_HEIGHT_CLASSES = sorted(list(set(f[0] for f in VOWEL_FEATURES.values())))
ALL_BACKNESS_CLASSES = sorted(list(set(f[1] for f in VOWEL_FEATURES.values())))
ALL_ROUNDING_CLASSES = sorted(list(set(f[2] for f in VOWEL_FEATURES.values())))

# ============================================================================
# Routing Maps: which sub-classes are valid for each parent class
# ============================================================================

# For each MOA class: which POA classes are possible?
MOA_TO_POA: Dict[str, List[str]] = {}
for moa in ALL_MOA_CLASSES:
    MOA_TO_POA[moa] = sorted(set(
        f[1] for f in CONSONANT_FEATURES.values() if f[0] == moa
    ))

# For each POA class: which voicing classes are possible?
POA_TO_VOICING: Dict[str, List[str]] = {}
for poa in ALL_POA_CLASSES:
    POA_TO_VOICING[poa] = sorted(set(
        f[2] for f in CONSONANT_FEATURES.values() if f[1] == poa
    ))

# For each Height class: which Backness classes are possible?
HEIGHT_TO_BACKNESS: Dict[str, List[str]] = {}
for height in ALL_HEIGHT_CLASSES:
    HEIGHT_TO_BACKNESS[height] = sorted(set(
        f[1] for f in VOWEL_FEATURES.values() if f[0] == height
    ))

# For each Backness class: which Rounding classes are possible?
BACKNESS_TO_ROUNDING: Dict[str, List[str]] = {}
for back in ALL_BACKNESS_CLASSES:
    BACKNESS_TO_ROUNDING[back] = sorted(set(
        f[2] for f in VOWEL_FEATURES.values() if f[1] == back
    ))

# ============================================================================
# Feature → Phoneme Conversion
# ============================================================================

def features_to_phoneme(pred):

    if pred['is_silence']:
        return "SIL"

    if pred.get('is_consonant', False):
        key = (pred.get("moa"), pred.get("poa"), pred.get("voicing"))
        for phoneme, features in CONSONANT_FEATURES.items():
            if features == key:
                return phoneme
        return "UNK-C"

    else:
        key = (pred.get("height"), pred.get("backness"), pred.get("rounding"))
        for phoneme, features in VOWEL_FEATURES.items():
            if features == key:
                return phoneme
        return "UNK-V"

# ============================================================================
# Model Architecture – Routed / Decision-Tree Classifier
# ============================================================================
#
# Routing tree:
#   ALL FRAMES
#   └── silence_head  →  [silence | speech]
#       └── cv_head   →  [vowel | consonant]
#           ├── CONSONANT PATH
#           │   └── moa_head  →  one of MOA classes
#           │       └── for each MOA class m:
#           │               poa_heads[m]     → POA classes valid for m
#           │               voicing_heads[m] → voicing classes valid for m
#           └── VOWEL PATH
#               └── height_head  →  one of Height classes
#                   └── for each Height class h:
#                           backness_heads[h]  → backness classes valid for h
#
# During INFERENCE the routed heads are selected by the model's own predictions
# at each stage.
# ============================================================================

class RoutedPhonemeClassifier(nn.Module):

    def __init__(self,
                 input_dim=768,
                 hidden_dim=512,
                 dropout=0.3):

        super().__init__()

        self.shared_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Stage 1 – silence vs speech
        self.silence_head = nn.Linear(hidden_dim, 2)

        # Stage 2 – consonant vs vowel (on speech frames)
        self.cv_head = nn.Linear(hidden_dim, 2)

        # Stage 3a – MOA (on consonant frames)
        self.moa_head = nn.Linear(hidden_dim, len(ALL_MOA_CLASSES))

        # Stage 4a – POA heads, one per MOA class
        self.poa_heads = nn.ModuleDict({
            moa: nn.Linear(hidden_dim, len(poa_classes))
            for moa, poa_classes in MOA_TO_POA.items()
        })

        # Stage 4a – Voicing heads, one per POA class (voicing constrained by POA)
        # Create a voicing head for each POA; each head outputs the voicing
        # classes valid for that POA.
        self.voicing_heads = nn.ModuleDict({
            poa: nn.Linear(hidden_dim, len(voicing_classes))
            for poa, voicing_classes in POA_TO_VOICING.items()
        })

        # Stage 3b – Height (on vowel frames)
        self.height_head = nn.Linear(hidden_dim, len(ALL_HEIGHT_CLASSES))

        # Stage 4b – Backness heads, one per Height class
        self.backness_heads = nn.ModuleDict({
            height: nn.Linear(hidden_dim, len(backness_classes))
            for height, backness_classes in HEIGHT_TO_BACKNESS.items()
        })
        # Stage 4c – Rounding heads, one per Backness class
        self.rounding_heads = nn.ModuleDict({
            back: nn.Linear(hidden_dim, len(rounding_classes))
            for back, rounding_classes in BACKNESS_TO_ROUNDING.items()
        })

    def forward(self, x):
        """
        Returns dictionary of logits for all routing stages.
        Caller selects which heads to use based on routing decisions.
        """
        features = self.shared_encoder(x)

        silence_logits = self.silence_head(features)
        cv_logits = self.cv_head(features)
        moa_logits = self.moa_head(features)
        height_logits = self.height_head(features)

        # Per-MOA heads
        poa_logits = {moa: head(features) for moa, head in self.poa_heads.items()}
        # Per-POA voicing heads
        voicing_logits = {poa: head(features) for poa, head in self.voicing_heads.items()}

        # Per-Height heads
        backness_logits = {h: head(features) for h, head in self.backness_heads.items()}
        # Per-Backness heads (rounding)
        rounding_logits = {b: head(features) for b, head in self.rounding_heads.items()}

        return {
            'is_silence': silence_logits,
            'is_consonant': cv_logits,
            'moa': moa_logits,
            'poa': poa_logits,
            'voicing': voicing_logits,
            'height': height_logits,
            'backness': backness_logits,
            'rounding': rounding_logits,
        }


# ============================================================
# Safe loader
# ============================================================

def load_checkpoint_safe(model_path, device):

    try:
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        return ckpt
    except Exception:
        sd = torch.load(model_path, map_location=device, weights_only=True)
        if isinstance(sd, dict) and 'model_state_dict' in sd:
            return sd
        return {'model_state_dict': sd}


# ============================================================
# Predictor
# ============================================================

class PhonemePredictor:

    def __init__(self, model_path, device=None):

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        ckpt = load_checkpoint_safe(model_path, self.device)

        self.model = RoutedPhonemeClassifier().to(self.device)

        if 'model_state_dict' in ckpt:
            self.model.load_state_dict(ckpt['model_state_dict'])
        else:
            self.model.load_state_dict(ckpt)

        self.model.eval()

    # --------------------------------------------------------

    def predict_from_audio(self, audio_path):

        waveform, sr = librosa.load(audio_path, sr=16000)

        fe = Wav2Vec2FeatureExtractor.from_pretrained(
            "facebook/hubert-base-ls960"
        )

        hubert = HubertModel.from_pretrained(
            "facebook/hubert-base-ls960"
        ).to(self.device)

        hubert.eval()
        for p in hubert.parameters():
            p.requires_grad = False

        inputs = fe(
            waveform,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )

        with torch.no_grad():
            out = hubert(
                inputs.input_values.to(self.device),
                output_hidden_states=True
            )

        embeddings = out.hidden_states[6].squeeze(0).cpu().numpy()

        return self.predict_from_embeddings(embeddings)

    # --------------------------------------------------------

    def predict_from_embeddings(self, embeddings):

        x = torch.from_numpy(embeddings).float().to(self.device)

        preds = []

        bs = 128

        with torch.no_grad():
            for s in range(0, x.shape[0], bs):

                batch = x[s:s+bs]
                logits_dict = self.model(batch)

                for i in range(logits_dict['is_silence'].size(0)):
                    preds.append(
                        self._parse_routed_prediction(logits_dict, i)
                    )

        return preds

    # --------------------------------------------------------

    def _parse_routed_prediction(self, logits_dict, idx):
        """
        Parse prediction from routed model using inference-time routing.
        
        Routing logic:
        1. Silence head selects: silence or speech
        2. If speech, CV head selects: consonant or vowel
        3. If consonant:
           - MOA head selects MOA class
           - Use MOA-specific POA/Voicing heads
        4. If vowel:
           - Height head selects height class
           - Use Height-specific Backness heads
        """
        pred = {}

        # Stage 1: Silence
        silence_logits = logits_dict['is_silence'][idx:idx+1]
        silence_probs = F.softmax(silence_logits, dim=1)[0]
        silence_pred = torch.argmax(silence_probs).item()

        pred['is_silence'] = silence_pred == 1
        pred['silence_prob'] = silence_probs[silence_pred].item()

        if pred['is_silence']:
            return pred

        # Stage 2: Consonant vs Vowel
        cv_logits = logits_dict['is_consonant'][idx:idx+1]
        cv_probs = F.softmax(cv_logits, dim=1)[0]
        cv_pred = torch.argmax(cv_probs).item()

        pred['is_consonant'] = cv_pred == 1
        pred['cv_prob'] = cv_probs[cv_pred].item()

        if pred['is_consonant']:
            # CONSONANT PATH
            # Stage 3a: MOA
            moa_logits = logits_dict['moa'][idx:idx+1]
            moa_probs = F.softmax(moa_logits, dim=1)[0]
            moa_idx = torch.argmax(moa_probs).item()
            moa_class = ALL_MOA_CLASSES[moa_idx]
            
            pred['moa'] = moa_class
            pred['moa_prob'] = moa_probs[moa_idx].item()

            # Stage 4a: POA (routed by MOA)
            poa_logits = logits_dict['poa'][moa_class][idx:idx+1]
            poa_probs = F.softmax(poa_logits, dim=1)[0]
            poa_idx = torch.argmax(poa_probs).item()
            poa_class = MOA_TO_POA[moa_class][poa_idx]
            
            pred['poa'] = poa_class
            pred['poa_prob'] = poa_probs[poa_idx].item()

            # Stage 4a: Voicing (routed by POA)
            voicing_logits = logits_dict['voicing'][poa_class][idx:idx+1]
            voicing_probs = F.softmax(voicing_logits, dim=1)[0]
            voicing_idx = torch.argmax(voicing_probs).item()
            voicing_class = POA_TO_VOICING[poa_class][voicing_idx]

            pred['voicing'] = voicing_class
            pred['voicing_prob'] = voicing_probs[voicing_idx].item()

        else:
            # VOWEL PATH
            # Stage 3b: Height
            height_logits = logits_dict['height'][idx:idx+1]
            height_probs = F.softmax(height_logits, dim=1)[0]
            height_idx = torch.argmax(height_probs).item()
            height_class = ALL_HEIGHT_CLASSES[height_idx]
            
            pred['height'] = height_class
            pred['height_prob'] = height_probs[height_idx].item()

            # Stage 4b: Backness (routed by Height)
            backness_logits = logits_dict['backness'][height_class][idx:idx+1]
            backness_probs = F.softmax(backness_logits, dim=1)[0]
            backness_idx = torch.argmax(backness_probs).item()
            backness_class = HEIGHT_TO_BACKNESS[height_class][backness_idx]
            
            pred['backness'] = backness_class
            pred['backness_prob'] = backness_probs[backness_idx].item()

            # Stage 4c: Rounding (routed by Backness)
            round_logits = logits_dict['rounding'][backness_class][idx:idx+1]
            round_probs = F.softmax(round_logits, dim=1)[0]
            round_idx = torch.argmax(round_probs).item()
            round_class = BACKNESS_TO_ROUNDING[backness_class][round_idx]

            pred['rounding'] = round_class
            pred['rounding_prob'] = round_probs[round_idx].item()

        return pred


# ============================================================
# CLI
# ============================================================

def main():

    model_path = input("Model path: ").strip()
    wav_folder = input("Path to wav folder: ").strip()

    predictor = PhonemePredictor(model_path)

    # ------------------------------------------
    # Accumulators
    # ------------------------------------------
    total_per_sum = 0.0
    total_files = 0

    total_frames_all = 0
    total_subs_all = 0
    total_ins_all = 0
    total_dels_all = 0

    # Substitution confusion accumulator
    from collections import defaultdict
    sub_confusion = defaultdict(lambda: defaultdict(int))
    phoneme_set = set()
    
    # Collect all predictions and references for per-phoneme stats
    all_pred_phonemes = []
    all_ref_phonemes = []

    print("\nProcessing all WAV files...")
    print("=" * 80)

    all_files = [f for f in os.listdir(wav_folder) if f.lower().endswith(".wav")]
    selected_files = all_files

    for file in selected_files:

        audio_path = os.path.join(wav_folder, file)
        audio_name = os.path.splitext(file)[0]

        parent_dir = os.path.dirname(wav_folder)
        tg_folder = os.path.join(parent_dir, "annotation")
        tg_path = os.path.join(tg_folder, audio_name + ".TextGrid")

        if not os.path.exists(audio_path):
            print(f"[SKIPPED] Audio missing: {file}")
            continue

        if not os.path.exists(tg_path):
            print(f"[SKIPPED] TextGrid missing: {file}")
            continue


        print(f"\nProcessing ({total_files+1}/{len(selected_files)}): {file}")

        # ------------------------------------------
        # Predict
        # ------------------------------------------
        preds = predictor.predict_from_audio(audio_path)
        predicted_phonemes = [features_to_phoneme(pr) for pr in preds]

        intervals = parse_textgrid(tg_path)
        reference_phonemes = get_reference_per_frame(
            intervals,
            len(predicted_phonemes)
        )
        
        # ------------------------------------------
        # Frame-Level Error Details (FOR THIS AUDIO)
        # ------------------------------------------
        print_frame_level_details(
            predicted_phonemes,
            reference_phonemes,
            title=f"FRAME-LEVEL ERRORS (FILE: {file})"
        )
        
        
        # -------------------------------------------------
        # Remove singleton frames (run-length = 1)
        # -------------------------------------------------
        reference_phonemes, predicted_phonemes = remove_singleton_frames(
            reference_phonemes,
            predicted_phonemes
        )
        
        merged_ref, merged_pred = print_merged_sequences(
        reference_phonemes,
        predicted_phonemes,
        ref_title=f"MERGED REFERENCE PHONEMES (FILE: {file})",
        pred_title=f"MERGED PREDICTED PHONEMES (FILE: {file})"
        )
        
        rows, sub_cnt, del_cnt, ins_cnt = print_phoneme_alignment(
            merged_ref, merged_pred
        )

        total_subs_all += sub_cnt
        total_dels_all += del_cnt
        total_ins_all  += ins_cnt
        total_frames_all += len(merged_ref)
        total_files += 1
        
    print("\n" + "=" * 60)
    print("OVERALL DATASET PER (MERGED PHONEMES)")
    print("=" * 60)

    if total_frames_all == 0:
        print("No reference phonemes found.")
    else:
        total_errors = total_subs_all + total_dels_all + total_ins_all
        per = total_errors / total_frames_all

        print(f"Total reference phonemes : {total_frames_all}")
        print(f"Total substitutions      : {total_subs_all}")
        print(f"Total deletions          : {total_dels_all}")
        print(f"Total insertions         : {total_ins_all}")
        print(f"Overall PER              : {per:.4f}")

def parse_textgrid(textgrid_path):
    """
    Extract only PHONEME tier from TextGrid.
    Treat AH and AH* as the same.
    """

    intervals = []
    in_phone_tier = False

    with open(textgrid_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    xmin = xmax = text = None

    for line in lines:
        line = line.strip()

        # Detect phoneme tier
        if 'name = "phones"' in line.lower():
            in_phone_tier = True
            continue

        # Stop when next tier starts
        if in_phone_tier and line.startswith("item ["):
            break

        if not in_phone_tier:
            continue

        if line.startswith("xmin ="):
            xmin = float(line.split("=")[1].strip())

        elif line.startswith("xmax ="):
            xmax = float(line.split("=")[1].strip())

        elif line.startswith("text ="):
            text = line.split("=")[1].strip().replace('"', '')

            # ------------------------------------------
            # Handle silence
            # ------------------------------------------
            if text == "" or text.lower() in ["sil", "sp", "silence"]:
                text = "SIL"

            else:
                # Handle multi-phoneme like D,S,s
                if "," in text:
                    parts = text.split(",")
                    if len(parts) >= 2:
                        text = parts[1].strip()
                    else:
                        text = parts[0].strip()

                # Remove stress numbers (AO1 → AO)
                text = ''.join([c for c in text if not c.isdigit()])

                # Remove star markers (AH* → AH)
                text = text.replace("*", "")

                text = text.upper()

            intervals.append((xmin, xmax, text))

    return intervals



def get_reference_per_frame(intervals, num_frames, frame_shift=0.02):

    ref_phonemes = []

    for i in range(num_frames):
        time = i * frame_shift
        phoneme = "SIL"

        for xmin, xmax, label in intervals:
            if xmin <= time < xmax:
                phoneme = label
                break

        ref_phonemes.append(phoneme)

    return ref_phonemes
    
    
def print_frame_level_details(pred_seq, ref_seq, 
                              title="Frame-Level Details",
                              frame_shift=0.02):
    """
    Print ALL frame-level phoneme comparison.
    Shows frame index, time, reference phoneme, predicted phoneme, and error type.
    """

    print("\n" + "=" * 120)
    print(title)
    print("=" * 120)
    print(f"{'Frame':<8} {'Time(s)':<10} {'Reference':<12} {'Predicted':<12} {'Error Type':<15}")
    print("-" * 120)

    for i, (p, r) in enumerate(zip(pred_seq, ref_seq)):

        time_sec = i * frame_shift

        if r == p:
            error_type = "Correct"
        elif r == "SIL" and p != "SIL":
            error_type = "Insertion"
        elif r != "SIL" and p == "SIL":
            error_type = "Deletion"
        else:
            error_type = "Substitution"

        print(f"{i:<8} {time_sec:<10.2f} {r:<12} {p:<12} {error_type:<15}")

    print("=" * 120)
    

def remove_singleton_frames(ref_seq, pred_seq, frame_shift=0.02,
                            title="PREDICTED FRAMES AFTER SINGLETON REMOVAL"):

    """
    Remove frames only from the predicted sequence where the predicted
    phoneme occurs only once in a continuous run.

    Reference sequence is kept unchanged.

    Prints remaining predicted frames.
    """

    n = len(pred_seq)

    def compute_run_lengths(seq):
        run_len = [0] * n
        i = 0
        while i < n:
            j = i + 1
            while j < n and seq[j] == seq[i]:
                j += 1

            length = j - i
            for k in range(i, j):
                run_len[k] = length

            i = j
        return run_len

    pred_runs = compute_run_lengths(pred_seq)

    new_pred = []

    # keep track of original frame index (useful for display)
    kept_indices = []

    for i in range(n):
        # remove ONLY if predicted run is singleton
        if pred_runs[i] == 1:
            continue

        new_pred.append(pred_seq[i])
        kept_indices.append(i)

    # ------------------------------------------------------
    # Print remaining predicted frames
    # ------------------------------------------------------
    print("\n" + "=" * 110)
    print(title)
    print("=" * 110)
    print(f"{'NewFrame':<10} {'OrigFrame':<12} {'Time(s)':<10} {'Predicted':<12}")
    print("-" * 110)

    for new_i, orig_i in enumerate(kept_indices):
        t = orig_i * frame_shift
        print(f"{new_i:<10} {orig_i:<12} {t:<10.2f} {new_pred[new_i]:<12}")

    print("=" * 110)

    # reference is returned unchanged
    return ref_seq, new_pred


def print_merged_sequences(ref_seq, pred_seq,
                           ref_title="MERGED REFERENCE PHONEMES",
                           pred_title="MERGED PREDICTED PHONEMES"):

    # ---------------------------------------
    # Phoneme equivalence groups
    # ---------------------------------------
    equiv_groups = [
        {"IY", "IH"},
        {"UW", "UH"},
        {"EH", "EY"},
        {"AX", "ER"},
        {"OW", "AO"},
        {"AW", "OY"}
    ]

    def same_phoneme(a, b):
        if a == b:
            return True
        for g in equiv_groups:
            if a in g and b in g:
                return True
        return False

    def merge_runs(seq):
        if not seq:
            return []

        merged = [seq[0]]

        for p in seq[1:]:
            # merge also if they are equivalent
            if same_phoneme(p, merged[-1]):
                continue
            else:
                merged.append(p)

        return merged

    merged_ref  = merge_runs(ref_seq)
    merged_pred = merge_runs(pred_seq)

    # ---------------------------
    # Print merged reference
    # ---------------------------
    print("\n" + "=" * 90)
    print(ref_title)
    print("=" * 90)
    print(f"{'Index':<8} {'Phoneme':<12}")
    print("-" * 90)

    for i, ph in enumerate(merged_ref):
        print(f"{i:<8} {ph:<12}")

    print("=" * 90)

    # ---------------------------
    # Print merged prediction
    # ---------------------------
    print("\n" + "=" * 90)
    print(pred_title)
    print("=" * 90)
    print(f"{'Index':<8} {'Phoneme':<12}")
    print("-" * 90)

    for i, ph in enumerate(merged_pred):
        print(f"{i:<8} {ph:<12}")

    print("=" * 90)

    return merged_ref, merged_pred


def print_phoneme_alignment(ref, pred):

    # -------------------------------------------------
    # Phoneme equivalence groups
    # -------------------------------------------------
    equiv_groups = [
        {"IY", "IH"},
        {"UW", "UH"},
        {"EH", "EY"},
        {"AX", "ER"},
        {"OW", "AO"},
        {"AW", "OY"}
    ]

    def same_phoneme(a, b):
        if a == b:
            return True
        for g in equiv_groups:
            if a in g and b in g:
                return True
        return False

    n = len(ref)
    m = len(pred)

    dp = [[0]*(m+1) for _ in range(n+1)]
    back = [[None]*(m+1) for _ in range(n+1)]

    for i in range(1, n+1):
        dp[i][0] = i
        back[i][0] = "DEL"

    for j in range(1, m+1):
        dp[0][j] = j
        back[0][j] = "INS"

    # -------------------------------------------------
    # DP
    # -------------------------------------------------
    for i in range(1, n+1):
        for j in range(1, m+1):

            if same_phoneme(ref[i-1], pred[j-1]):
                dp[i][j] = dp[i-1][j-1]
                back[i][j] = "OK"
            else:
                sub = dp[i-1][j-1] + 1
                ins = dp[i][j-1] + 1
                dele = dp[i-1][j] + 1

                best = min(sub, ins, dele)
                dp[i][j] = best

                # same priority as your previous code
                if best == sub:
                    back[i][j] = "SUB"
                elif best == dele:
                    back[i][j] = "DEL"
                else:
                    back[i][j] = "INS"

    # -------------------------------------------------
    # Backtrace
    # -------------------------------------------------
    i, j = n, m
    rows = []

    sub_cnt = 0
    del_cnt = 0
    ins_cnt = 0

    while i > 0 or j > 0:

        op = back[i][j]

        if op == "OK":
            # print reference symbol on both sides
            rows.append((ref[i-1], ref[i-1], "Correct"))
            i -= 1
            j -= 1

        elif op == "SUB":
            rows.append((ref[i-1], pred[j-1], "Substitution"))
            sub_cnt += 1
            i -= 1
            j -= 1

        elif op == "DEL":
            rows.append((ref[i-1], "-", "Deletion"))
            del_cnt += 1
            i -= 1

        elif op == "INS":
            rows.append(("-", pred[j-1], "Insertion"))
            ins_cnt += 1
            j -= 1

    rows.reverse()

    # -------------------------------------------------
    # Print
    # -------------------------------------------------
    print("\n" + "="*40)
    print(f"{'REF':<4} {'PRED':<6} OP")
    print("-"*40)

    for r, p, op in rows:
        print(f"{r:<4} {p:<6} {op}")

    print("-"*40)
    print(f"Substitution: {sub_cnt}")
    print(f"Deletion    : {del_cnt}")
    print(f"Insertion   : {ins_cnt}")

    return rows, sub_cnt, del_cnt, ins_cnt


if __name__ == "__main__":
    main()
