import os
import numpy as np
import torch
import json
from tqdm import tqdm

BASE_DIR = 'data/processed/'
RESOLUTIONS = [16, 32, 64, 128, 256]
OUTPUT_JSON = 'wavelet_stats.json'

def load_data(path):
    data = np.load(path)
    if isinstance(data, np.lib.npyio.NpzFile):
        data = data['data']
    return data  # shape (12, H, W)

def compute_stats_for_resolution(res):
    path = os.path.join(BASE_DIR, f'resolution_{res}')
    files = [f for f in os.listdir(path) if f.endswith('.npy') or f.endswith('.npz')]

    print(f"Processing resolution {res} with {len(files)} files...")

    num_channels = 12
    running_sum = torch.zeros(num_channels)
    running_sqsum = torch.zeros(num_channels)
    pixel_count = torch.zeros(num_channels)
    min_vals = torch.full((num_channels,), float('inf'))
    max_vals = torch.full((num_channels,), float('-inf'))

    for f in tqdm(files):
        try:
            data = load_data(os.path.join(path, f))  # (12, H, W)
            tensor = torch.from_numpy(data).float()  # shape (12, H, W)
        except Exception as e:
            print(f"Warning: skipping {f} (load error: {e})")
            continue

        if tensor.numel() == 0:
            print(f"Warning: skipping {f} (empty tensor)")
            continue

        for c in range(num_channels):
            ch = tensor[c]
            mask = torch.isfinite(ch)
            if not mask.any():
                continue

            valid = ch[mask]
            n = valid.numel()

            running_sum[c] += valid.sum()
            running_sqsum[c] += (valid ** 2).sum()
            pixel_count[c] += n
            min_vals[c] = min(min_vals[c], valid.min())
            max_vals[c] = max(max_vals[c], valid.max())

    # Safe mean/std computation
    mean = torch.where(pixel_count > 0, running_sum / pixel_count, torch.zeros_like(running_sum))
    var = torch.where(pixel_count > 0, (running_sqsum / pixel_count) - mean ** 2, torch.ones_like(running_sum))
    std = torch.sqrt(torch.clamp(var, min=1e-6))

    return {
        'mean': mean.tolist(),
        'std': std.tolist(),
        'min': min_vals.tolist(),
        'max': max_vals.tolist(),
    }

def main():
    all_stats = {}
    for res in RESOLUTIONS:
        stats = compute_stats_for_resolution(res)
        all_stats[f'resolution_{res}'] = stats

    # Save to JSON
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(all_stats, f, indent=4)
    print(f"\n✅ Stats saved to {OUTPUT_JSON}")

if __name__ == '__main__':
    main()
