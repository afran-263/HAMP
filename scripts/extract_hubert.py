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
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform.squeeze(0)


def extract_all_layers(model, feature_extractor, audio_path, device):
    """
    Returns a list of numpy arrays, one per hidden state layer.
    Index 0 = CNN output, 1..N = transformer layers.
    Each array has shape (T, D).
    """
    waveform = load_audio(audio_path)
    inputs = feature_extractor(
        waveform.numpy(), sampling_rate=16000, return_tensors="pt"
    )
    input_values = inputs.input_values.to(device)

    with torch.no_grad():
        outputs = model(input_values, output_hidden_states=True)

    # outputs.hidden_states is a tuple: (layer_0, layer_1, ..., layer_N)
    all_layers = [
        h.squeeze(0).cpu().numpy()   # (T, D)
        for h in outputs.hidden_states
    ]
    return all_layers


def main():
    parser = argparse.ArgumentParser(
        description="Extract HuBERT embeddings for every layer."
    )
    parser.add_argument('--audio_dir',   type=str, required=True,
                        help='Root directory of audio files')
    parser.add_argument('--output_dir',  type=str, required=True,
                        help='Root output directory; sub-folders layer_N are created automatically')
    parser.add_argument('--model_name',  type=str, default='facebook/hubert-base-ls960',
                        help='HuggingFace model name (default: hubert-base-ls960)')
    parser.add_argument('--layers',      type=int, nargs='+', default=None,
                        help='Which layer indices to save (default: all layers). '
                             'E.g. --layers 6 9 12')
    parser.add_argument('--ext',         type=str, default='.flac',
                        help='Audio file extension (default: .flac)')
    parser.add_argument('--skip_existing', action='store_true', default=True,
                        help='Skip utterances already extracted (default: True)')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Loading HuBERT model: {args.model_name}")

    model = HubertModel.from_pretrained(args.model_name).to(device)
    model.eval()
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.model_name)

    # Determine total number of layers from model config
    num_transformer_layers = model.config.num_hidden_layers   # 12 for base, 24 for large
    total_layers = num_transformer_layers + 1                  # +1 for CNN output (layer_0)

    # Decide which layers to save
    if args.layers is not None:
        layers_to_save = sorted(set(args.layers))
        invalid = [l for l in layers_to_save if l >= total_layers]
        if invalid:
            raise ValueError(
                f"Requested layers {invalid} are out of range. "
                f"Model has {total_layers} layers (0 to {total_layers - 1})."
            )
    else:
        layers_to_save = list(range(total_layers))

    print(f"Model has {total_layers} layers total (layer_0 = CNN, layer_1..{total_layers-1} = transformer)")
    print(f"Saving layers: {layers_to_save}")

    # Create one output sub-directory per layer
    layer_dirs = {}
    for l in layers_to_save:
        d = os.path.join(args.output_dir, f"layer_{l}")
        os.makedirs(d, exist_ok=True)
        layer_dirs[l] = d

    # Collect audio files
    audio_files = []
    for root, _, files in os.walk(args.audio_dir):
        for f in files:
            if f.endswith(args.ext):
                audio_files.append(os.path.join(root, f))

    print(f"Found {len(audio_files)} audio files\n")

    skipped = 0
    errors  = 0

    for audio_path in tqdm(audio_files, desc="Extracting embeddings"):
        utt_id = os.path.splitext(os.path.basename(audio_path))[0]

        # Check if all requested layers already exist for this utterance
        if args.skip_existing:
            already_done = all(
                os.path.exists(os.path.join(layer_dirs[l], utt_id + '.npy'))
                for l in layers_to_save
            )
            if already_done:
                skipped += 1
                continue

        try:
            all_layer_embs = extract_all_layers(
                model, feature_extractor, audio_path, device
            )

            for l in layers_to_save:
                out_path = os.path.join(layer_dirs[l], utt_id + '.npy')
                if args.skip_existing and os.path.exists(out_path):
                    continue
                np.save(out_path, all_layer_embs[l])

        except Exception as e:
            errors += 1
            tqdm.write(f"[ERROR] {audio_path}: {e}")

    print(f"\nDone.")
    print(f"  Processed : {len(audio_files) - skipped - errors}")
    print(f"  Skipped   : {skipped}  (already existed)")
    print(f"  Errors    : {errors}")
    print(f"\nEmbeddings saved under: {args.output_dir}")
    print("Sub-directories created:")
    for l in layers_to_save:
        n = len(os.listdir(layer_dirs[l]))
        print(f"  layer_{l:<3}  →  {n} files   ({layer_dirs[l]})")


if __name__ == '__main__':
    main()
