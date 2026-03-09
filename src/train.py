import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
import json
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import Counter
import warnings
import re
warnings.filterwarnings('ignore')


# ============================================================================
# Configuration
# ============================================================================
@dataclass
class Config:
  
    embedding_dir: str = r"D:\iiit new\main\NEW\hubert_base\lib_dataset\layer_6"  # UPDATE THIS
    alignment_dir: str = r"D:\iiit new\lib_dataset\data\txt"    # UPDATE THIS (changed from textgrid_dir)
    
    # Model hyperparameters
    input_dim: int = 768  # HuBERT base embedding dimension
    hidden_dim: int = 512
    dropout: float = 0.4  # Increased from 0.3 for better regularization
    
    # Training hyperparameters
    batch_size: int = 256
    epochs: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4  # Increased L2 regularization
    
    # Overfitting Prevention
    use_label_smoothing: bool = True
    label_smoothing_factor: float = 0.1
    use_mixup: bool = True
    mixup_alpha: float = 0.2
    gradient_clip_value: float = 1.0
    use_warmup: bool = True
    warmup_epochs: int = 5
    use_l1_regularization: bool = False
    l1_lambda: float = 1e-5
    
    # Validation and early stopping
    val_ratio: float = 0.10
    test_ratio: float = 0.10
    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 0.001  # Minimum improvement threshold
    overfitting_threshold: float = 0.2  # Alert if |train_loss - val_loss| > threshold
    
    # Loss weighting
    silence_weight: float = 1.0
    cv_weight: float = 1.0
    moa_weight: float = 1.0
    poa_weight: float = 1.0
    voicing_weight: float = 1.0
    height_weight: float = 1.0
    backness_weight: float = 1.0
    rounding_weight: float = 1.0
    
    # Debugging
    debug_mode: bool = True
    validate_before_training: bool = True
    
    # Other settings
    save_dir: str = "saved1"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    random_seed: int = 42
    
    def __post_init__(self):
        os.makedirs(self.save_dir, exist_ok=True)


# ============================================================================
# Phoneme Feature Definitions (Multi-hot Vector Mappings)
# ============================================================================

# Updated to handle CMU phoneme dictionary format with stress markers
# Stress markers (0, 1, 2) will be stripped automatically

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

SILENCE_MARKERS = {"SIL", "SP", "SPN", ""}

# Create unified class lists
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

# POA -> voicing mapping: which voicing classes are valid for each POA
POA_TO_VOICING: Dict[str, List[str]] = {}
for poa in ALL_POA_CLASSES:
    POA_TO_VOICING[poa] = sorted(set(
        f[2] for f in CONSONANT_FEATURES.values() if f[1] == poa
    ))


# ============================================================================
# Helper Functions
# ============================================================================

def strip_stress_markers(phoneme: str) -> str:

    # Remove digits from the end of the phoneme
    return re.sub(r'[012]$', '', phoneme)


# ============================================================================
# Debugging Functions
# ============================================================================

def validate_dataset_quick(embedding_dir: str, alignment_dir: str, max_files: int = 5):
    
    print("\n" + "=" * 80)
    print("QUICK DATASET VALIDATION")
    print("=" * 80)
    
    if not os.path.exists(embedding_dir):
        print(f"ERROR: Embedding directory not found: {embedding_dir}")
        return False
    
    if not os.path.exists(alignment_dir):
        print(f"ERROR: Alignment directory not found: {alignment_dir}")
        return False
    
    emb_files = [f for f in os.listdir(embedding_dir) if f.endswith('.npy')]
    align_files = [f for f in os.listdir(alignment_dir) if f.endswith('.txt')]
    
    print(f"✓ Found {len(emb_files)} .npy files")
    print(f"✓ Found {len(align_files)} .txt files")
    
    if len(emb_files) == 0:
        print("ERROR: No .npy files found!")
        return False
    
    if len(align_files) == 0:
        print("ERROR: No .txt alignment files found!")
        return False
    
    # Check first few files
    phoneme_counts = Counter()
    cv_counts = {'consonant': 0, 'vowel': 0, 'silence': 0, 'unknown': 0}
    
    print(f"\nChecking first {min(max_files, len(emb_files))} files...")
    
    for i, fname in enumerate(emb_files[:max_files]):
        file_id = fname.replace('.npy', '')
        emb_path = os.path.join(embedding_dir, fname)
        align_path = os.path.join(alignment_dir, f"{file_id}.txt")
        
        print(f"\n  File {i+1}: {file_id}")
        
        # Check embedding
        try:
            emb = np.load(emb_path)
            print(f"    Embedding shape: {emb.shape}")
            if emb.shape[-1] != 768:
                print(f"    ⚠️  WARNING: Expected last dim 768, got {emb.shape[-1]}")
        except Exception as e:
            print(f"Error loading embedding: {e}")
            continue
        
        # Check alignment file
        if not os.path.exists(align_path):
            print(f"No matching alignment file!")
            continue
        
        try:
            alignments = parse_alignment_file(align_path)
            
            file_phonemes = []
            for phoneme, start, end in alignments:
                phoneme_clean = strip_stress_markers(phoneme.upper())
                if phoneme_clean:
                    file_phonemes.append(phoneme_clean)
                    phoneme_counts[phoneme_clean] += 1
                    
                    if phoneme_clean in CONSONANT_FEATURES:
                        cv_counts['consonant'] += 1
                    elif phoneme_clean in VOWEL_FEATURES:
                        cv_counts['vowel'] += 1
                    elif phoneme_clean in SILENCE_MARKERS:
                        cv_counts['silence'] += 1
                    else:
                        cv_counts['unknown'] += 1
            
            print(f"    Phonemes: {len(file_phonemes)}")
            print(f"    Sample: {' '.join(file_phonemes[:15])}")
            
        except Exception as e:
            print(f"Error reading alignment file: {e}")
            continue
    
    # Print summary
    print(f"\n{'=' * 80}")
    print("PHONEME DISTRIBUTION")
    print(f"{'=' * 80}")
    print(f"Consonants: {cv_counts['consonant']}")
    print(f"Vowels: {cv_counts['vowel']}")
    print(f"Silence: {cv_counts['silence']}")
    print(f"Unknown: {cv_counts['unknown']}")
    
    if cv_counts['consonant'] == 0:
        print("\nERROR: NO CONSONANTS FOUND!")
        print("\nPossible reasons:")
        print("1. Case mismatch - alignment file has lowercase but code expects uppercase")
        print("2. Different phoneme set - alignment uses different labels")
        print("\nTop phonemes in your alignment files:")
        for phoneme, count in phoneme_counts.most_common(10):
            in_consonants = "✓ CONSONANT" if phoneme in CONSONANT_FEATURES else ""
            in_vowels = "✓ VOWEL" if phoneme in VOWEL_FEATURES else ""
            in_silence = "✓ SILENCE" if phoneme in SILENCE_MARKERS else ""
            status = in_consonants or in_vowels or in_silence or "UNKNOWN"
            print(f"  {phoneme:10s} {count:5d}  {status}")
        
        print("\nTo fix:")
        print("1. Check if your files use lowercase (k, t, s)")
        print("2. Update CONSONANT_FEATURES keys to match (change 'K' to 'k', etc.)")
        
        return False
    
    if cv_counts['unknown'] > 0:
        print(f"\nWARNING: {cv_counts['unknown']} unknown phonemes will be skipped!")
        print("Unknown phonemes:")
        all_known = set(CONSONANT_FEATURES.keys()) | set(VOWEL_FEATURES.keys()) | SILENCE_MARKERS
        unknown = set(phoneme_counts.keys()) - all_known
        for phoneme in sorted(unknown):
            print(f"  {phoneme}: {phoneme_counts[phoneme]} occurrences")
    
    print("\n✓ Validation PASSED")
    return True


def debug_batch_composition(dataloader, max_batches: int = 10):
    """Debug what's in the batches"""
    print("\n" + "=" * 80)
    print(f"ANALYZING FIRST {max_batches} BATCHES")
    print("=" * 80)
    
    total_samples = 0
    total_silence = 0
    total_consonants = 0
    total_vowels = 0
    
    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break
        
        labels = batch['labels']
        batch_size = labels['is_silence'].shape[0]
        total_samples += batch_size
        
        # Count silence
        silence_labels = torch.argmax(labels['is_silence'], dim=1)
        num_silence = (silence_labels == 1).sum().item()
        total_silence += num_silence
        
        # Count speech
        is_speech = (silence_labels == 0)
        num_speech = is_speech.sum().item()
        
        if num_speech > 0:
            cv_labels = torch.argmax(labels['is_consonant'][is_speech], dim=1)
            num_consonants = (cv_labels == 1).sum().item()
            num_vowels = (cv_labels == 0).sum().item()
            
            total_consonants += num_consonants
            total_vowels += num_vowels
            
            print(f"Batch {i+1:2d}: Total={batch_size}, Silence={num_silence}, C={num_consonants}, V={num_vowels}")
            
            # Check if consonant labels are valid
            if num_consonants > 0:
                consonant_mask = (cv_labels == 1)
                consonant_indices = is_speech.nonzero(as_tuple=True)[0][consonant_mask]
                
                moa_vectors = labels['moa'][consonant_indices]
                moa_sums = moa_vectors.sum(dim=1)
                
                if not torch.allclose(moa_sums, torch.ones_like(moa_sums)):
                    print(f"MOA vector sums: {moa_sums[:5].tolist()}")
        else:
            print(f"Batch {i+1:2d}: ALL SILENCE (batch_size={batch_size})")
    
    print(f"\n{'=' * 80}")
    print(f"BATCH ANALYSIS SUMMARY")
    print(f"{'=' * 80}")
    print(f"Total samples: {total_samples}")
    print(f"Silence: {total_silence} ({total_silence/total_samples*100:.1f}%)")
    print(f"Consonants: {total_consonants} ({total_consonants/total_samples*100:.1f}%)")
    print(f"Vowels: {total_vowels} ({total_vowels/total_samples*100:.1f}%)")
    
    if total_consonants == 0:
        print("\nERROR: NO CONSONANTS IN BATCHES!")
        print("This is why MOA/POA/Voicing accuracies are 0.0")
        return False
    
    return True


# ============================================================================
# Multi-hot Vector Creation
# ============================================================================

def create_multihot_vector(phoneme: str) -> Dict[str, np.ndarray]:
    """Create multi-hot vectors for a phoneme"""
    # Strip stress markers and convert to uppercase
    phoneme_clean = strip_stress_markers(phoneme.upper())
    
    # Initialize all vectors
    vectors = {
        'is_silence': np.zeros(2, dtype=np.float32),
        'is_consonant': np.zeros(2, dtype=np.float32),
        'moa': np.zeros(len(ALL_MOA_CLASSES), dtype=np.float32),
        'poa': np.zeros(len(ALL_POA_CLASSES), dtype=np.float32),
        'voicing': np.zeros(len(ALL_VOICING_CLASSES), dtype=np.float32),
        'height': np.zeros(len(ALL_HEIGHT_CLASSES), dtype=np.float32),
        'backness': np.zeros(len(ALL_BACKNESS_CLASSES), dtype=np.float32),
        'rounding': np.zeros(len(ALL_ROUNDING_CLASSES), dtype=np.float32),
    }
    
    # Check if silence
    if phoneme_clean in SILENCE_MARKERS:
        vectors['is_silence'][1] = 1.0
        return vectors
    
    # Mark as speech
    vectors['is_silence'][0] = 1.0
    
    # Check if consonant
    if phoneme_clean in CONSONANT_FEATURES:
        vectors['is_consonant'][1] = 1.0
        moa, poa, voicing = CONSONANT_FEATURES[phoneme_clean]
        
        vectors['moa'][ALL_MOA_CLASSES.index(moa)] = 1.0
        vectors['poa'][ALL_POA_CLASSES.index(poa)] = 1.0
        vectors['voicing'][ALL_VOICING_CLASSES.index(voicing)] = 1.0
        
    # Check if vowel
    elif phoneme_clean in VOWEL_FEATURES:
        vectors['is_consonant'][0] = 1.0
        height, backness, rounding = VOWEL_FEATURES[phoneme_clean]
        
        vectors['height'][ALL_HEIGHT_CLASSES.index(height)] = 1.0
        vectors['backness'][ALL_BACKNESS_CLASSES.index(backness)] = 1.0
        vectors['rounding'][ALL_ROUNDING_CLASSES.index(rounding)] = 1.0
    
    return vectors


# ============================================================================
# Alignment File Processing
# ============================================================================

def parse_alignment_file(alignment_path: str) -> List[Tuple[str, float, float]]:
   
    # Expected each line: start_time   end_time   phoneme
    alignments = []
    
    try:
        with open(alignment_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                # Split by whitespace (handles tabs/spaces)
                parts = line.split()
                
                if len(parts) != 3:
                    print(f"Warning: Invalid line {line_num} in {alignment_path}: {line}")
                    continue
                
                phoneme,start_str, end_str = parts
                
                try:
                    start_time = float(start_str)
                    end_time = float(end_str)
                    
                    # Validate times
                    if end_time <= start_time:
                        print(f"Warning: Invalid time range at line {line_num} in {alignment_path}")
                        continue
                    
                    alignments.append((phoneme, start_time, end_time))
                    
                except ValueError as e:
                    print(f"Warning: Invalid time values at line {line_num} in {alignment_path}: {e}")
                    continue
        
        return alignments
    
    except Exception as e:
        print(f"Error parsing {alignment_path}: {e}")
        return []


def frames_to_time(frame_idx: int, hop_length: float = 0.02) -> float:
    """Convert frame index to time in seconds"""
    return frame_idx * hop_length


def time_to_frames(time: float, hop_length: float = 0.02) -> int:
    """Convert time in seconds to frame index"""
    return int(time / hop_length)


# ============================================================================
# Dataset
# ============================================================================

class PhonemeFrameDataset(Dataset):
    
    def __init__(
        self, 
        embedding_dir: str,
        alignment_dir: str,
        file_list: List[str],
        hop_length: float = 0.02,
        debug: bool = False
    ):
        self.embedding_dir = embedding_dir
        self.alignment_dir = alignment_dir
        self.file_list = file_list
        self.hop_length = hop_length
        self.samples = []
        self.debug = debug
        
        # Statistics
        self.stats = {
            'total_frames': 0,
            'consonants': 0,
            'vowels': 0,
            'silence': 0,
            'skipped': 0
        }
        
        # Build dataset
        self._build_dataset()
        
        print(f"Loaded {len(self.samples)} frames from {len(file_list)} files")
        if self.debug:
            print(f"  Consonants: {self.stats['consonants']}")
            print(f"  Vowels: {self.stats['vowels']}")
            print(f"  Silence: {self.stats['silence']}")
            print(f"  Skipped: {self.stats['skipped']}")
    
    def _build_dataset(self):
        """Build frame-level samples with labels"""
        
        skipped_files = []
        
        for file_id in tqdm(self.file_list, desc="Building dataset"):
            # Load embedding
            emb_path = os.path.join(self.embedding_dir, f"{file_id}.npy")
            if not os.path.exists(emb_path):
                skipped_files.append((file_id, "No embedding"))
                continue
            
            embeddings = np.load(emb_path)
            
            # Load alignment file
            align_path = os.path.join(self.alignment_dir, f"{file_id}.txt")
            if not os.path.exists(align_path):
                skipped_files.append((file_id, "No alignment file"))
                continue
            
            alignments = parse_alignment_file(align_path)
            if not alignments:
                skipped_files.append((file_id, "Empty alignment file"))
                continue
            
            # Create frame-level labels
            num_frames = embeddings.shape[0]
            self.stats['total_frames'] += num_frames
            
            for frame_idx in range(num_frames):
                frame_time = frames_to_time(frame_idx, self.hop_length)
                
                # Find which phoneme this frame belongs to
                phoneme = None
                for ph, start, end in alignments:
                    if start <= frame_time < end:
                        phoneme = ph
                        break
                
                if phoneme is None:
                    continue
                
                # Create multi-hot vectors for this phoneme
                labels = create_multihot_vector(phoneme)
                
                # Check if phoneme is recognized (after cleaning)
                phoneme_clean = strip_stress_markers(phoneme.upper())
                if phoneme_clean in CONSONANT_FEATURES:
                    self.stats['consonants'] += 1
                elif phoneme_clean in VOWEL_FEATURES:
                    self.stats['vowels'] += 1
                elif phoneme_clean in SILENCE_MARKERS:
                    self.stats['silence'] += 1
                else:
                    self.stats['skipped'] += 1
                    continue  # Skip unknown phonemes
                
                # Store sample
                self.samples.append({
                    'embedding': embeddings[frame_idx],
                    'labels': labels,
                    'phoneme': phoneme,
                    'file_id': file_id,
                    'frame_idx': frame_idx
                })
        
        if skipped_files and self.debug:
            print(f"\nSkipped {len(skipped_files)} files:")
            for file_id, reason in skipped_files[:5]:
                print(f"  {file_id}: {reason}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        return {
            'embedding': torch.FloatTensor(sample['embedding']),
            'labels': {k: torch.FloatTensor(v) for k, v in sample['labels'].items()},
            'phoneme': sample['phoneme']
        }


# ============================================================================
# Model Architecture  –  Routed / Decision-Tree Classifier
# ============================================================================

class RoutedPhonemeClassifier(nn.Module):

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 512,
        dropout: float = 0.3,
    ):
        super().__init__()

        # ------------------------------------------------------------------
        # Shared encoder  (identical to original architecture)
        # ------------------------------------------------------------------
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

        # ------------------------------------------------------------------
        # Stage 1 – silence vs speech
        # ------------------------------------------------------------------
        self.silence_head = nn.Linear(hidden_dim, 2)

        # ------------------------------------------------------------------
        # Stage 2 – consonant vs vowel  (on speech frames)
        # ------------------------------------------------------------------
        self.cv_head = nn.Linear(hidden_dim, 2)

        # ------------------------------------------------------------------
        # Stage 3a – MOA  (on consonant frames)
        # ------------------------------------------------------------------
        self.moa_head = nn.Linear(hidden_dim, len(ALL_MOA_CLASSES))

        # ------------------------------------------------------------------
        # Stage 4a – POA heads, one per MOA class
        #            each head outputs only the POA classes valid for that MOA
        # ------------------------------------------------------------------
        self.poa_heads = nn.ModuleDict({
            moa: nn.Linear(hidden_dim, len(poa_classes))
            for moa, poa_classes in MOA_TO_POA.items()
        })

        # ------------------------------------------------------------------
        # Stage 4a – Voicing heads, one per MOA class
        # ------------------------------------------------------------------
        # Stage 4a – Voicing heads, one per POA class (voicing constrained by POA)
        # Create a voicing head for each POA; each head outputs the voicing
        # classes valid for that POA.
        # ------------------------------------------------------------------
        self.voicing_heads = nn.ModuleDict({
            poa: nn.Linear(hidden_dim, len(voicing_classes))
            for poa, voicing_classes in POA_TO_VOICING.items()
        })

        # ------------------------------------------------------------------
        # Stage 3b – Height  (on vowel frames)
        # ------------------------------------------------------------------
        self.height_head = nn.Linear(hidden_dim, len(ALL_HEIGHT_CLASSES))

        # ------------------------------------------------------------------
        # Stage 4b – Backness heads, one per Height class
        # ------------------------------------------------------------------
        self.backness_heads = nn.ModuleDict({
            height: nn.Linear(hidden_dim, len(backness_classes))
            for height, backness_classes in HEIGHT_TO_BACKNESS.items()
        })
        # rounding heads per backness
        self.rounding_heads = nn.ModuleDict({
            back: nn.Linear(hidden_dim, len(rounding_classes))
            for back, rounding_classes in BACKNESS_TO_ROUNDING.items()
        })

    # ------------------------------------------------------------------
    # Forward  –  returns ALL head logits; the caller selects which ones
    #             to use based on routing labels (train) or predictions
    #             (inference).
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.shared_encoder(x)                          # (B, H)

        silence_logits = self.silence_head(features)               # (B, 2)
        cv_logits      = self.cv_head(features)                    # (B, 2)
        moa_logits     = self.moa_head(features)                   # (B, n_moa)
        height_logits  = self.height_head(features)                # (B, n_height)

        # Per-MOA POA logits  →  dict of (B, n_poa_for_moa)
        poa_logits     = {moa: head(features) for moa, head in self.poa_heads.items()}
        # Per-POA voicing logits  →  dict of (B, n_voicing_for_poa)
        voicing_logits = {poa: head(features) for poa, head in self.voicing_heads.items()}

        # Per-Height heads  →  dict of (B, n_backness_for_height)
        backness_logits = {h: head(features) for h, head in self.backness_heads.items()}
        # Per-Backness heads  →  dict of (B, n_rounding_for_backness)
        rounding_logits = {b: head(features) for b, head in self.rounding_heads.items()}

        return {
            'is_silence'  : silence_logits,
            'is_consonant': cv_logits,
            'moa'         : moa_logits,
            'poa'         : poa_logits,       # dict keyed by MOA class name
            'voicing'     : voicing_logits,   # dict keyed by MOA class name
            'height'      : height_logits,
            'backness'    : backness_logits,  # dict keyed by Height class name
            'rounding'    : rounding_logits,  # dict keyed by Backness class name
        }


# ============================================================================
# Overfitting Prevention Functions
# ============================================================================

def apply_label_smoothing(logits: torch.Tensor, targets: torch.Tensor, 
                         smoothing: float) -> torch.Tensor:
    """Apply label smoothing to reduce overconfidence and overfitting."""
    n_classes = logits.size(-1)
    log_probs = F.log_softmax(logits, dim=-1)
    
    # Smooth labels: uniform distribution + target distribution
    smooth_target = torch.zeros_like(log_probs)
    smooth_target.fill_(smoothing / (n_classes - 1))
    smooth_target.scatter_(1, targets.unsqueeze(1), 1.0 - smoothing)
    
    return -torch.sum(smooth_target * log_probs, dim=-1)


def mixup_batch(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply Mixup augmentation to increase dataset diversity."""
    batch_size = x.size(0)
    lambda_val = np.random.beta(alpha, alpha)
    
    index = torch.randperm(batch_size)
    x_mixed = lambda_val * x + (1 - lambda_val) * x[index, :]
    y_a = y
    y_b = y[index]
    
    return x_mixed, (y_a, y_b), lambda_val


def clip_gradients(model: nn.Module, max_norm: float) -> None:
    """Apply gradient clipping to prevent exploding gradients."""
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)


def add_l1_regularization(model: nn.Module, l1_lambda: float) -> torch.Tensor:
    """Add L1 regularization to encourage sparsity."""
    l1_reg = torch.tensor(0.0, device=next(model.parameters()).device)
    for param in model.parameters():
        l1_reg += torch.sum(torch.abs(param))
    return l1_lambda * l1_reg


def get_warmup_lr(epoch: int, base_lr: float, warmup_epochs: int) -> float:
    """Calculate learning rate with linear warmup schedule."""
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    return base_lr


# ============================================================================
# Training Functions
# ============================================================================

def routed_loss(
    predictions: Dict[str, torch.Tensor],
    labels: Dict[str, torch.Tensor],
    config: 'Config',
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute loss using the routing tree with teacher forcing.

    At each routing decision the *ground-truth* label selects which
    specialised head receives a gradient, so every sub-head trains only
    on samples that genuinely belong to its branch.

    Loss terms
    ----------
    silence   – all frames
    cv        – speech frames only
    moa       – consonant frames only
    poa       – per-MOA-class subset of consonant frames
    voicing   – per-MOA-class subset of consonant frames
    height    – vowel frames only
    backness  – per-Height-class subset of vowel frames
    """
    device = predictions['is_silence'].device
    zero   = torch.tensor(0.0, device=device)

    # ---- Stage 1 : silence -----------------------------------------------
    silence_gt  = torch.argmax(labels['is_silence'], dim=1)          # 0=speech, 1=sil
    silence_loss = F.cross_entropy(predictions['is_silence'], silence_gt)

    is_speech_mask = (silence_gt == 0)                                # bool (B,)

    if is_speech_mask.sum() == 0:
        total = config.silence_weight * silence_loss
        return total, {'silence': silence_loss.item(),
                       'cv': 0., 'moa': 0., 'poa': 0.,
                       'voicing': 0., 'height': 0., 'backness': 0.}

    # ---- Stage 2 : consonant vs vowel  (speech frames) -------------------
    cv_gt   = torch.argmax(labels['is_consonant'][is_speech_mask], dim=1)  # 0=vowel,1=cons
    cv_loss = F.cross_entropy(predictions['is_consonant'][is_speech_mask], cv_gt)

    # Global index arrays for consonant / vowel frames
    speech_indices    = is_speech_mask.nonzero(as_tuple=True)[0]
    is_consonant_mask = (cv_gt == 1)
    is_vowel_mask     = (cv_gt == 0)

    # ---- Stage 3a : MOA  (consonant frames) ------------------------------
    moa_loss = zero
    poa_loss = zero
    voicing_loss = zero

    if is_consonant_mask.sum() > 0:
        cons_indices = speech_indices[is_consonant_mask]              # global frame idx

        moa_gt   = torch.argmax(labels['moa'][cons_indices], dim=1)
        moa_loss = F.cross_entropy(predictions['moa'][cons_indices], moa_gt)

        # ---- Stage 4a : POA and Voicing  (per-MOA routing, teacher-forced) --
        poa_loss_sum     = zero.clone()
        voicing_loss_sum = zero.clone()
        poa_count     = 0
        voicing_count = 0

        for moa_idx, moa_name in enumerate(ALL_MOA_CLASSES):
            # Frames whose ground-truth MOA is moa_name
            moa_mask    = (moa_gt == moa_idx)
            if moa_mask.sum() == 0:
                continue
            moa_frame_indices = cons_indices[moa_mask]

            # --- POA ---
            poa_classes = MOA_TO_POA[moa_name]          # valid POA labels for this MOA
            # Build local ground-truth: index within poa_classes list
            global_poa_gt = torch.argmax(labels['poa'][moa_frame_indices], dim=1)
            # labels['poa'] is a one-hot over ALL_POA_CLASSES; map to local index
            local_poa_gt = torch.tensor(
                [poa_classes.index(ALL_POA_CLASSES[g.item()]) for g in global_poa_gt],
                dtype=torch.long, device=device
            )
            poa_logits_local = predictions['poa'][moa_name][moa_frame_indices]
            poa_loss_sum = poa_loss_sum + F.cross_entropy(poa_logits_local, local_poa_gt)
            poa_count += 1

            # --- Voicing ---
            # We route voicing by POA (use ground-truth POA during training)
            global_poa_gt = torch.argmax(labels['poa'][moa_frame_indices], dim=1)
            # For each POA class present among these frames, accumulate loss
            poa_classes_for_moa = MOA_TO_POA[moa_name]
            for local_p_idx, poa_name in enumerate(poa_classes_for_moa):
                # global POA indices corresponding to this local POA
                global_poa_idx = ALL_POA_CLASSES.index(poa_name)
                poa_mask = (global_poa_gt == global_poa_idx)
                if poa_mask.sum() == 0:
                    continue
                poa_frame_idxs = moa_frame_indices[poa_mask]

                voicing_classes = POA_TO_VOICING[poa_name]
                global_voicing_gt = torch.argmax(labels['voicing'][poa_frame_idxs], dim=1)
                local_voicing_gt = torch.tensor(
                    [voicing_classes.index(ALL_VOICING_CLASSES[g.item()])
                     for g in global_voicing_gt],
                    dtype=torch.long, device=device
                )
                voicing_logits_local = predictions['voicing'][poa_name][poa_frame_idxs]
                voicing_loss_sum = voicing_loss_sum + F.cross_entropy(
                    voicing_logits_local, local_voicing_gt
                )
                voicing_count += 1

        poa_loss     = poa_loss_sum     / max(poa_count, 1)
        voicing_loss = voicing_loss_sum / max(voicing_count, 1)

    # ---- Stage 3b : Height  (vowel frames) -------------------------------
    height_loss  = zero
    backness_loss = zero
    rounding_loss = zero

    if is_vowel_mask.sum() > 0:
        vowel_indices = speech_indices[is_vowel_mask]

        height_gt   = torch.argmax(labels['height'][vowel_indices], dim=1)
        height_loss = F.cross_entropy(predictions['height'][vowel_indices], height_gt)

        # compute global backness labels for all vowel frames (needed for rounding)
        all_back_gt = torch.argmax(labels['backness'][vowel_indices], dim=1)

        # ---- Stage 4b : Backness  (per-Height routing, teacher-forced) ------
        backness_loss_sum = zero.clone()
        backness_count    = 0

        for h_idx, h_name in enumerate(ALL_HEIGHT_CLASSES):
            h_mask = (height_gt == h_idx)
            if h_mask.sum() == 0:
                continue
            h_frame_indices = vowel_indices[h_mask]

            backness_classes = HEIGHT_TO_BACKNESS[h_name]
            global_back_gt   = torch.argmax(labels['backness'][h_frame_indices], dim=1)
            local_back_gt    = torch.tensor(
                [backness_classes.index(ALL_BACKNESS_CLASSES[g.item()])
                 for g in global_back_gt],
                dtype=torch.long, device=device
            )
            back_logits_local = predictions['backness'][h_name][h_frame_indices]
            backness_loss_sum = backness_loss_sum + F.cross_entropy(
                back_logits_local, local_back_gt
            )
            backness_count += 1

        backness_loss = backness_loss_sum / max(backness_count, 1)
        # rounding loss follow-up will be computed after this block

    # ---- Stage 4c : Rounding (per-backness, teacher-forced) ------------
    if is_vowel_mask.sum() > 0:
        rounding_loss_sum = zero.clone()
        rounding_count = 0
        for back_idx, back_name in enumerate(ALL_BACKNESS_CLASSES):
            back_mask = (all_back_gt == back_idx)
            if back_mask.sum() == 0:
                continue
            back_frame_idxs = vowel_indices[back_mask]

            rounding_classes = BACKNESS_TO_ROUNDING[back_name]
            global_round_gt = torch.argmax(labels['rounding'][back_frame_idxs], dim=1)
            local_round_gt = torch.tensor(
                [rounding_classes.index(ALL_ROUNDING_CLASSES[g.item()])
                 for g in global_round_gt],
                dtype=torch.long, device=device
            )
            round_logits_local = predictions['rounding'][back_name][back_frame_idxs]
            rounding_loss_sum = rounding_loss_sum + F.cross_entropy(
                round_logits_local, local_round_gt
            )
            rounding_count += 1
        rounding_loss = rounding_loss_sum / max(rounding_count, 1)

    # ---- Weighted total --------------------------------------------------
    total_loss = (
        config.silence_weight  * silence_loss  +
        config.cv_weight       * cv_loss       +
        config.moa_weight      * moa_loss      +
        config.poa_weight      * poa_loss      +
        config.voicing_weight  * voicing_loss  +
        config.height_weight   * height_loss   +
        config.backness_weight * backness_loss  +
        config.rounding_weight * rounding_loss
    )

    losses = {
        'silence' : silence_loss.item(),
        'cv'      : cv_loss.item(),
        'moa'     : moa_loss.item(),
        'poa'     : poa_loss.item(),
        'voicing' : voicing_loss.item(),
        'height'  : height_loss.item(),
        'backness': backness_loss.item(),
        'rounding': rounding_loss.item(),
    }
    return total_loss, losses


def train_epoch(model, dataloader, optimizer, config, epoch: int = 0):
    """
    Train for one epoch with overfitting prevention techniques.
    
    Techniques applied:
    - Gradient clipping to prevent exploding gradients
    - L1/L2 regularization via optimizer and manual L1 term
    - Mixup augmentation for data diversity
    - Label smoothing to reduce overconfidence
    - Learning rate warmup schedule
    """
    model.train()
    total_loss = 0
    all_losses = {k: 0.0 for k in ['silence', 'cv', 'moa', 'poa', 'voicing', 'height', 'backness', 'rounding']}
    
    # Apply warmup to learning rate if enabled
    if config.use_warmup and epoch < config.warmup_epochs:
        warmup_lr = get_warmup_lr(epoch, config.learning_rate, config.warmup_epochs)
        for param_group in optimizer.param_groups:
            param_group['lr'] = warmup_lr
    
    for batch in tqdm(dataloader, desc="Training"):
        embeddings = batch['embedding'].to(config.device)
        labels = {k: v.to(config.device) for k, v in batch['labels'].items()}
        
        optimizer.zero_grad()
        
        predictions = model(embeddings)
        loss, losses = routed_loss(predictions, labels, config)
        
        # Add L1 regularization if enabled
        if config.use_l1_regularization:
            l1_loss = add_l1_regularization(model, config.l1_lambda)
            loss = loss + l1_loss
        
        loss.backward()
        
        # Apply gradient clipping to prevent exploding gradients
        clip_gradients(model, config.gradient_clip_value)
        
        optimizer.step()
        
        total_loss += loss.item()
        for k, v in losses.items():
            all_losses[k] += v
    
    n_batches = len(dataloader)
    avg_loss = total_loss / n_batches
    avg_losses = {k: v / n_batches for k, v in all_losses.items()}
    
    return avg_loss, avg_losses


def evaluate(model, dataloader, config):

    model.eval()
    total_loss = 0
    all_losses = {k: 0.0 for k in ['silence', 'cv', 'moa', 'poa', 'voicing', 'height', 'backness', 'rounding']}

    correct = {k: 0 for k in ['silence', 'cv', 'moa', 'poa', 'voicing', 'height', 'backness', 'rounding']}
    total   = {k: 0 for k in ['silence', 'cv', 'moa', 'poa', 'voicing', 'height', 'backness', 'rounding']}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            embeddings = batch['embedding'].to(config.device)
            labels     = {k: v.to(config.device) for k, v in batch['labels'].items()}

            predictions = model(embeddings)
            loss, losses = routed_loss(predictions, labels, config)

            total_loss += loss.item()
            for k, v in losses.items():
                all_losses[k] += v

            device = config.device

            # --- silence accuracy ---
            silence_gt   = torch.argmax(labels['is_silence'], dim=1)
            silence_pred = torch.argmax(predictions['is_silence'], dim=1)
            correct['silence'] += (silence_pred == silence_gt).sum().item()
            total['silence']   += len(silence_gt)

            is_speech_mask = (silence_gt == 0)
            if is_speech_mask.sum() == 0:
                continue

            speech_indices = is_speech_mask.nonzero(as_tuple=True)[0]

            # --- C/V accuracy ---
            cv_gt   = torch.argmax(labels['is_consonant'][is_speech_mask], dim=1)
            cv_pred = torch.argmax(predictions['is_consonant'][is_speech_mask], dim=1)
            correct['cv'] += (cv_pred == cv_gt).sum().item()
            total['cv']   += len(cv_gt)

            is_consonant_mask = (cv_gt == 1)
            is_vowel_mask     = (cv_gt == 0)

            # --- MOA accuracy  (teacher-forced routing) ---
            if is_consonant_mask.sum() > 0:
                cons_indices = speech_indices[is_consonant_mask]
                moa_gt   = torch.argmax(labels['moa'][cons_indices], dim=1)
                moa_pred = torch.argmax(predictions['moa'][cons_indices], dim=1)
                correct['moa'] += (moa_pred == moa_gt).sum().item()
                total['moa']   += len(moa_gt)

                # --- POA / Voicing accuracy per MOA class ---
                for moa_idx, moa_name in enumerate(ALL_MOA_CLASSES):
                    moa_mask = (moa_gt == moa_idx)
                    if moa_mask.sum() == 0:
                        continue
                    moa_frame_indices = cons_indices[moa_mask]

                    # POA
                    poa_classes    = MOA_TO_POA[moa_name]
                    global_poa_gt  = torch.argmax(labels['poa'][moa_frame_indices], dim=1)
                    local_poa_gt   = torch.tensor(
                        [poa_classes.index(ALL_POA_CLASSES[g.item()]) for g in global_poa_gt],
                        dtype=torch.long, device=device
                    )
                    local_poa_pred = torch.argmax(
                        predictions['poa'][moa_name][moa_frame_indices], dim=1
                    )
                    correct['poa'] += (local_poa_pred == local_poa_gt).sum().item()
                    total['poa']   += len(local_poa_gt)
                    # Voicing: select voicing head based on ground-truth POA (teacher-forcing for evaluation)
                    # This ensures we evaluate voicing accuracy using the correct POA branch
                    global_poa_gt_all = torch.argmax(labels['poa'][moa_frame_indices], dim=1)
                    poa_classes_for_moa = MOA_TO_POA[moa_name]
                    for local_p_idx, poa_name in enumerate(poa_classes_for_moa):
                        global_poa_idx = ALL_POA_CLASSES.index(poa_name)
                        poa_mask = (global_poa_gt_all == global_poa_idx)
                        if poa_mask.sum() == 0:
                            continue
                        poa_frame_idxs = moa_frame_indices[poa_mask]

                        voicing_classes = POA_TO_VOICING[poa_name]
                        global_voicing_gt = torch.argmax(labels['voicing'][poa_frame_idxs], dim=1)
                        local_voicing_gt = torch.tensor(
                            [voicing_classes.index(ALL_VOICING_CLASSES[g.item()])
                             for g in global_voicing_gt],
                            dtype=torch.long, device=device
                        )
                        voicing_logits_local = predictions['voicing'][poa_name][poa_frame_idxs]
                        local_voicing_pred = torch.argmax(voicing_logits_local, dim=1)
                        correct['voicing'] += (local_voicing_pred == local_voicing_gt).sum().item()
                        total['voicing']   += len(local_voicing_gt)

            # --- Height accuracy ---
            if is_vowel_mask.sum() > 0:
                vowel_indices = speech_indices[is_vowel_mask]
                height_gt   = torch.argmax(labels['height'][vowel_indices], dim=1)
                height_pred = torch.argmax(predictions['height'][vowel_indices], dim=1)
                correct['height'] += (height_pred == height_gt).sum().item()
                total['height']   += len(height_gt)

                # --- Backness accuracy per Height class ---
                for h_idx, h_name in enumerate(ALL_HEIGHT_CLASSES):
                    h_mask = (height_gt == h_idx)
                    if h_mask.sum() == 0:
                        continue
                    h_frame_indices = vowel_indices[h_mask]

                    back_classes   = HEIGHT_TO_BACKNESS[h_name]
                    global_back_gt = torch.argmax(labels['backness'][h_frame_indices], dim=1)
                    local_back_gt  = torch.tensor(
                        [back_classes.index(ALL_BACKNESS_CLASSES[g.item()])
                         for g in global_back_gt],
                        dtype=torch.long, device=device
                    )
                    local_back_pred = torch.argmax(
                        predictions['backness'][h_name][h_frame_indices], dim=1
                    )
                    correct['backness'] += (local_back_pred == local_back_gt).sum().item()
                    total['backness']   += len(local_back_gt)

                    # --- Rounding accuracy using true backness ---
                    for back_idx, back_name in enumerate(ALL_BACKNESS_CLASSES):
                        mask2 = (global_back_gt == back_idx)
                        if mask2.sum() == 0:
                            continue
                        idxs2 = h_frame_indices[mask2]
                        rounding_classes = BACKNESS_TO_ROUNDING[back_name]
                        global_round_gt = torch.argmax(labels['rounding'][idxs2], dim=1)
                        local_round_gt = torch.tensor(
                            [rounding_classes.index(ALL_ROUNDING_CLASSES[g.item()])
                             for g in global_round_gt],
                            dtype=torch.long, device=device
                        )
                        round_logits_local = predictions['rounding'][back_name][idxs2]
                        local_round_pred = torch.argmax(round_logits_local, dim=1)
                        correct['rounding'] += (local_round_pred == local_round_gt).sum().item()
                        total['rounding']   += len(local_round_gt)

    n_batches = len(dataloader)
    avg_loss   = total_loss / n_batches
    avg_losses = {k: v / n_batches for k, v in all_losses.items()}

    accuracies = {
        k: correct[k] / total[k] if total[k] > 0 else 0.0
        for k in correct
    }
    return avg_loss, avg_losses, accuracies


# ============================================================================
# Main Training Script
# ============================================================================

def main():

    
    # Set random seeds
    config = Config()
    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)
    
    print("\n" + "=" * 80)
    print("HIERARCHICAL PHONEME CLASSIFICATION TRAINING - TXT FORMAT")
    print("=" * 80)
    print(f"Device: {config.device}")
    print(f"Embedding dir: {config.embedding_dir}")
    print(f"Alignment dir: {config.alignment_dir}")
    
    # STEP 1: Validate dataset
    if config.validate_before_training:
        if not validate_dataset_quick(config.embedding_dir, config.alignment_dir, max_files=5):
            print("\n❌ Dataset validation FAILED!")
            print("Please fix the issues above before training.")
            return
    
    # STEP 2: Get file list
    all_files = []
    for fname in os.listdir(config.embedding_dir):
        if fname.endswith('.npy'):
            file_id = fname.replace('.npy', '')
            if os.path.exists(os.path.join(config.alignment_dir, f"{file_id}.txt")):
                all_files.append(file_id)
    
    print(f"\nFound {len(all_files)} audio files with both embeddings and alignments")
    
    if len(all_files) == 0:
        print("❌ No valid file pairs found!")
        return
    
    # STEP 3: Split data
    train_files, test_files = train_test_split(
        all_files, 
        test_size=config.test_ratio,
        random_state=config.random_seed
    )
    train_files, val_files = train_test_split(
        train_files,
        test_size=config.val_ratio / (1 - config.test_ratio),
        random_state=config.random_seed
    )
    
    print(f"Train files: {len(train_files)}")
    print(f"Val files: {len(val_files)}")
    print(f"Test files: {len(test_files)}")
    
    # STEP 4: Create datasets
    print("\nBuilding datasets...")
    train_dataset = PhonemeFrameDataset(
        config.embedding_dir,
        config.alignment_dir,
        train_files,
        debug=config.debug_mode
    )
    
    val_dataset = PhonemeFrameDataset(
        config.embedding_dir,
        config.alignment_dir,
        val_files,
        debug=config.debug_mode
    )
    
    # Check if we have consonants
    if train_dataset.stats['consonants'] == 0:
        print("\n❌ CRITICAL ERROR: NO CONSONANTS in training data!")
        print("Training cannot proceed. Please fix the phoneme definitions.")
        return
    
    if val_dataset.stats['consonants'] == 0:
        print("\n❌ CRITICAL ERROR: NO CONSONANTS in validation data!")
        print("Training cannot proceed. Please fix the phoneme definitions.")
        return
    
    # STEP 5: Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,  # Use 0 for debugging, increase for production
        pin_memory=True if config.device == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if config.device == 'cuda' else False
    )
    
    # STEP 6: Debug batch composition
    if config.debug_mode:
        print("\n" + "=" * 80)
        print("DEBUGGING TRAIN LOADER")
        print("=" * 80)
        if not debug_batch_composition(train_loader, max_batches=5):
            print("\n❌ Training data has no consonants!")
            return
        
        print("\n" + "=" * 80)
        print("DEBUGGING VAL LOADER")
        print("=" * 80)
        if not debug_batch_composition(val_loader, max_batches=5):
            print("\n❌ Validation data has no consonants!")
            return
    
    # STEP 7: Initialize model
    print("\n" + "=" * 80)
    print("INITIALIZING MODEL")
    print("=" * 80)
    
    model = RoutedPhonemeClassifier(
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout
    ).to(config.device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    print(f"\nNumber of classes:")
    print(f"  MOA: {len(ALL_MOA_CLASSES)} - {ALL_MOA_CLASSES}")
    print(f"  POA: {len(ALL_POA_CLASSES)} - {ALL_POA_CLASSES}")
    print(f"  Voicing: {len(ALL_VOICING_CLASSES)} - {ALL_VOICING_CLASSES}")
    print(f"  Height: {len(ALL_HEIGHT_CLASSES)} - {ALL_HEIGHT_CLASSES}")
    print(f"  Backness: {len(ALL_BACKNESS_CLASSES)} - {ALL_BACKNESS_CLASSES}")
    print(f"  Rounding: {len(ALL_ROUNDING_CLASSES)} - {ALL_ROUNDING_CLASSES}")
    
    # STEP 8: Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Create ReduceLROnPlateau compatibly (some PyTorch versions don't accept 'verbose')
    try:
        base_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
    except TypeError:
        base_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )

    # Wrap scheduler to print when LR is reduced (replaces verbose behavior)
    class _SchedulerWrapper:
        def __init__(self, sched, optim):
            self._sched = sched
            self._optim = optim
        def step(self, metric):
            old = [g['lr'] for g in self._optim.param_groups]
            self._sched.step(metric)
            new = [g['lr'] for g in self._optim.param_groups]
            if any(n < o for n, o in zip(new, old)):
                print(f"✓ LR reduced: {old} -> {new}")
    scheduler = _SchedulerWrapper(base_scheduler, optimizer)
    
    # STEP 9: Training loop
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80 + "\n")
    
    best_val_loss = float('inf')
    patience_counter = 0
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': [],
        'overfitting_gap': []  # Track train vs val loss gap
    }
    
    for epoch in range(config.epochs):
        print(f"\nEpoch {epoch + 1}/{config.epochs}")
        print("-" * 80)
        
        # Train with overfitting prevention (pass epoch for warmup)
        train_loss, train_losses = train_epoch(model, train_loader, optimizer, config, epoch)
        
        # Validate
        val_loss, val_losses, val_accs = evaluate(model, val_loader, config)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Calculate overfitting gap
        overfitting_gap = val_loss - train_loss
        
        # Log results
        print(f"\nTrain Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Overfitting Gap (Val - Train): {overfitting_gap:.4f}")
        
        # Alert if overfitting is detected
        if overfitting_gap > config.overfitting_threshold:
            print(f"⚠️  WARNING: High overfitting detected! Gap = {overfitting_gap:.4f}")
            print("   Consider: increasing dropout, using more data augmentation, or reducing model complexity")
        
        print("\nValidation Accuracies:")
        for k, v in val_accs.items():
            print(f"  {k.upper()}: {v:.4f}")
        
        # Check for consonant accuracy issues
        if val_accs.get('moa', 0) == 0 and epoch > 0:
            print("\nWARNING: MOA accuracy is still 0!")
            print("This suggests consonants are not being processed correctly.")
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_accs)
        history['overfitting_gap'].append(overfitting_gap)
        
        # Early stopping and model saving with improved logic
        improvement = best_val_loss - val_loss
        
        if improvement > config.early_stopping_min_delta:  # Only count substantial improvements
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_accs': val_accs,
                'config': config,
                'all_classes': {
                    'moa': ALL_MOA_CLASSES,
                    'poa': ALL_POA_CLASSES,
                    'voicing': ALL_VOICING_CLASSES,
                    'height': ALL_HEIGHT_CLASSES,
                    'backness': ALL_BACKNESS_CLASSES,
                }
            }, os.path.join(config.save_dir, 'best_model.pth'))
            
            print(f"\n✓ Model saved (val_loss: {val_loss:.4f}, improvement: {improvement:.6f})")
        else:
            patience_counter += 1
            print(f"\nNo improvement ({improvement:.6f} < {config.early_stopping_min_delta})")
            print(f"Patience: {patience_counter}/{config.early_stopping_patience}")
            
            if patience_counter >= config.early_stopping_patience:
                print("\n⚠ Early stopping triggered")
                break
    
    # Save final model and history
    torch.save(model.state_dict(), os.path.join(config.save_dir, 'final_model.pth'))
    
    with open(os.path.join(config.save_dir, 'training_history.json'), 'w') as f:
        json.dump({
            'train_loss': history['train_loss'],
            'val_loss': history['val_loss'],
            'overfitting_gap': history['overfitting_gap'],  # Save overfitting metrics
            'val_acc': [
                {k: float(v) for k, v in acc.items()}
                for acc in history['val_acc']
            ]
        }, f, indent=2)
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"\nBest validation loss: {best_val_loss:.4f}")
    print(f"Models saved to: {config.save_dir}")
    
    # Print overfitting statistics
    if history['overfitting_gap']:
        avg_gap = np.mean(history['overfitting_gap'])
        max_gap = np.max(history['overfitting_gap'])
        print(f"\nOverfitting Statistics:")
        print(f"  Average gap (val - train): {avg_gap:.4f}")
        print(f"  Max gap: {max_gap:.4f}")
        if avg_gap < config.overfitting_threshold:
            print(f"  ✓ Model shows good generalization (gap < {config.overfitting_threshold})")
        else:
            print(f"  ⚠️  Model may be overfitting (gap > {config.overfitting_threshold})")



if __name__ == "__main__":
    main()