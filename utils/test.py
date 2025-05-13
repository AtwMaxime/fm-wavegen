import os
import numpy as np
import pywt
from PIL import Image
import matplotlib.pyplot as plt

WAVELET = 'haar'
BASE_PATH = 'data/processed/'
FILENAME = '000000552386_16.npy'  # base filename, the suffix will change per resolution

def get_hf_components(npy_array):
    LH = np.stack([npy_array[3], npy_array[4], npy_array[5]], axis=-1)
    HL = np.stack([npy_array[6], npy_array[7], npy_array[8]], axis=-1)
    HH = np.stack([npy_array[9], npy_array[10], npy_array[11]], axis=-1)
    return LH, HL, HH

def get_ll(npy_array):
    return np.stack([npy_array[0], npy_array[1], npy_array[2]], axis=-1)

def reconstruct_from_wavelet(LL, LH, HL, HH):
    reconstructed = []
    for c in range(3):
        coeffs = (LL[:, :, c], (LH[:, :, c], HL[:, :, c], HH[:, :, c]))
        rec = pywt.idwt2(coeffs, WAVELET)
        reconstructed.append(rec)
    return np.stack(reconstructed, axis=-1)

def load_npy(filename, resolution):
    path = os.path.join(BASE_PATH, f'resolution_{resolution}', filename.replace('_16.npy', f'_{resolution}.npy'))
    return np.load(path)

def main():
    resolutions = [16, 32, 64, 128, 256, 512]

    # Step 1: load LL and HF from resolution_16
    data_16 = load_npy(FILENAME, 16)
    LL = get_ll(data_16)
    LH, HL, HH = get_hf_components(data_16)
    LL = reconstruct_from_wavelet(LL, LH, HL, HH)
    print(f"Reconstructed to resolution 32: {LL.shape}")

    # Step 2+: load only HF, apply IWT with previous LL
    for i in range(1, len(resolutions) - 1):
        res = resolutions[i]
        next_res = resolutions[i + 1]

        print(f"Reconstructing to resolution {next_res}")
        data = load_npy(FILENAME, res)
        LH, HL, HH = get_hf_components(data)
        print(f"LL: {LL.shape}, LH: {LH.shape}, HL: {HL.shape}, HH: {HH.shape}")
        LL = reconstruct_from_wavelet(LL, LH, HL, HH)
        print(f"Reconstructed to resolution {next_res}: {LL.shape}")

    # Clamp and convert final result to uint8
    reconstructed_img = np.clip(LL, 0, 255).astype(np.uint8)

    # Load original image
    image_id = FILENAME.replace('_16.npy', '.jpg')
    original_path = os.path.join(BASE_PATH, 'resized', image_id)
    original_img = Image.open(original_path).convert('RGB')
    original_array = np.array(original_img)

    # Plot side by side
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original_array)
    axes[0].set_title("Original (512×512)")
    axes[0].axis('off')

    axes[1].imshow(reconstructed_img)
    axes[1].set_title("Reconstructed")
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig("comparison.png")
    plt.show()

    from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity

    # Ensure same dtype and shape
    reconstructed_img = reconstructed_img.astype(np.uint8)
    original_array = original_array.astype(np.uint8)

    # Compute metrics
    mse = mean_squared_error(original_array, reconstructed_img)
    psnr = peak_signal_noise_ratio(original_array, reconstructed_img, data_range=255)
    ssim = structural_similarity(original_array, reconstructed_img, channel_axis=-1)

    print(f"MSE: {mse:.4f}")
    print(f"PSNR: {psnr:.2f} dB")
    print(f"SSIM: {ssim:.4f}")

    import cv2

    # --- Compute normalized error ---
    error_map = np.abs(original_array.astype(np.float32) - reconstructed_img.astype(np.float32)) / 255.0
    mean_error = error_map.mean(axis=-1)  # shape: (H, W)

    # --- Optional: log-scale for visibility ---
    log_mean_error = np.log1p(mean_error)  # log(1 + x) to preserve 0 and enhance small diffs

    # --- Show plain heatmap ---
    plt.figure(figsize=(6, 5), dpi=150)
    plt.imshow(log_mean_error, cmap='hot')
    plt.colorbar(label='Log Pixel-wise Error')
    plt.title("Reconstruction Error Heatmap (log scale)")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("error_heatmap_log.png")
    plt.show()

def main2():
    resolutions = [16, 32, 64, 128, 256, 512]
    input_dir = os.path.join(BASE_PATH, 'resolution_16')
    output_dir = os.path.join(BASE_PATH, 'reconstruct')
    os.makedirs(output_dir, exist_ok=True)

    npy_files = [f for f in os.listdir(input_dir) if f.endswith('.npy')]

    for i, filename in enumerate(npy_files):
        try:
            # Step 1: LL + HF from 16
            data = load_npy(filename, 16)
            LL = get_ll(data)
            LH, HL, HH = get_hf_components(data)
            LL = reconstruct_from_wavelet(LL, LH, HL, HH)

            # Step 2+: use HF from higher resolutions
            for j in range(1, len(resolutions) - 1):
                res = resolutions[j]
                data = load_npy(filename, res)
                LH, HL, HH = get_hf_components(data)
                LL = reconstruct_from_wavelet(LL, LH, HL, HH)

            # Final image: clamp and convert
            reconstructed_img = np.clip(LL, 0, 255).astype(np.uint8)
            save_path = os.path.join(output_dir, filename.replace('_16.npy', '.png'))
            Image.fromarray(reconstructed_img).save(save_path)

        except Exception as e:
            print(f"Failed to process {filename}: {e}")

from torch.utils.data import DataLoader
from data.hf_dataset import HFDataset, inverse_normalize
from utils.inverse_wavelet import batch_iwt
from utils.LL_to_image import to_image
from models.hf_unet import HFUNet
import torch

dataset = HFDataset('data/processed', resolution=16)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for ll, hf in loader:
    print("LL mean per channel:", ll.mean(dim=(0, 2, 3)))
    print("LL std per channel:", ll.std(dim=(0, 2, 3)))
    print("HF mean per channel:", hf.mean(dim=(0, 2, 3)))
    print("HF std per channel:", hf.std(dim=(0, 2, 3)))
    print("✅ LL shape:", ll.shape)  # (B, 3, H, W)
    print("✅ HF shape:", hf.shape)  # (B, 9, H, W)
    break  # only first batch

import matplotlib.pyplot as plt

# Take first sample from batch
ll_1 = ll[0].unsqueeze(0)  # (1, 3, H, W)
hf_1 = hf[0].unsqueeze(0)  # (1, 9, H, W)

# Inverse normalize
ll_inv = inverse_normalize(ll_1, dataset.ll_mean, dataset.ll_std)
hf_inv = inverse_normalize(hf_1, dataset.hf_mean, dataset.hf_std)

# Apply inverse wavelet transform
reconstructed = batch_iwt(ll_inv, hf_inv)  # (1, 3, 2H, 2W)

# Convert to image
img = to_image(reconstructed[0])  # PIL.Image

# Plot the final image
plt.figure(figsize=(5, 5))
plt.imshow(img)
plt.axis('off')
plt.title("Reconstructed Image from LL + HF")
plt.tight_layout()
plt.show()


model = HFUNet(
    in_channels=12,
    out_channels=12,
    cond_dim=256,
    block_out_channels=(128, 256, 512),
    layers_per_block=2,
    attention_head_dim=64,
).to("cuda")


model = HFUNet().cuda()

x = torch.randn(4, 12, 64, 64).cuda()
t = torch.randint(0, 1000, (4,)).cuda()
res = [64, 64, 64, 64]

out = model(x, t, res)
print(out.shape)  # (4, 12, 64, 64)


if __name__ == '__main__':
    main2()
