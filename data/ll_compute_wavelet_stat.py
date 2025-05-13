import os
import numpy as np
import torch
import json
from tqdm import tqdm

LL_DIR = 'data/processed_cifar10_ll/resolution_16'
OUTPUT_JSON = 'll_stats.json'

def compute_ll_stats(directory):
    files = [f for f in os.listdir(directory) if f.endswith('.npy')]
    print(f"Processing {len(files)} LL files...")

    num_channels = 3
    running_sum = torch.zeros(num_channels)
    running_sqsum = torch.zeros(num_channels)
    pixel_count = torch.zeros(num_channels)
    min_vals = torch.full((num_channels,), float('inf'))
    max_vals = torch.full((num_channels,), float('-inf'))

    for fname in tqdm(files):
        path = os.path.join(directory, fname)
        data = np.load(path)  # (3, H, W)
        tensor = torch.from_numpy(data).float()

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

    mean = torch.where(pixel_count > 0, running_sum / pixel_count, torch.zeros_like(running_sum))
    var = torch.where(pixel_count > 0, (running_sqsum / pixel_count) - mean ** 2, torch.ones_like(running_sum))
    std = torch.sqrt(torch.clamp(var, min=1e-6))

    return {
        'mean': mean.tolist(),
        'std': std.tolist(),
        'min': min_vals.tolist(),
        'max': max_vals.tolist(),
    }

if __name__ == '__main__':
    stats = compute_ll_stats(LL_DIR)
    with open(OUTPUT_JSON, 'w') as f:
        json.dump({'resolution_16': stats}, f, indent=4)
    print(f"✅ LL stats saved to {OUTPUT_JSON}")
