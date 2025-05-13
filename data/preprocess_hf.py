import os
import numpy as np
import pywt
from PIL import Image

# Define paths
RAW_DATA_DIR = 'data/raw/coco2017/train2017'
PROCESSED_DATA_BASE_DIR = 'data/processed/'
INITIAL_IMAGES_DIR = os.path.join(PROCESSED_DATA_BASE_DIR, 'resized/')

# Parameters
WAVELET = 'haar'  # Haar wavelet
IMAGE_SIZE = (512, 512)  # Initial image size after resizing
MIN_RESOLUTION = 16  # Minimum resolution to process

def process_image(image_path):
    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    width, height = img.size
    left = (width - 480) / 2
    top = (height - 480) / 2
    right = (width + 480) / 2
    bottom = (height + 480) / 2
    img = img.crop((left, top, right, bottom))
    img = img.resize(IMAGE_SIZE, Image.LANCZOS)
    img_array = np.array(img, dtype=np.float32)

    # Save initial image
    os.makedirs(INITIAL_IMAGES_DIR, exist_ok=True)
    initial_image_path = os.path.join(INITIAL_IMAGES_DIR, os.path.basename(image_path))
    img.save(initial_image_path, format='PNG')

    # Initialize LL components
    LL_R, LL_G, LL_B = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
    current_resolution = IMAGE_SIZE[0]

    while current_resolution > MIN_RESOLUTION:
        # Perform wavelet decomposition
        new_LL = []
        LHs, HLs, HHs = [], [], []

        for LL in [LL_R, LL_G, LL_B]:
            LL_out, (LH, HL, HH) = pywt.dwt2(LL, WAVELET)
            new_LL.append(LL_out)
            LHs.append(LH)
            HLs.append(HL)
            HHs.append(HH)

        # Save current LL and the HF components from this level
        components = new_LL + LHs + HLs + HHs
        resolution = current_resolution // 2  # because LL and HF are half-sized
        wavelet_image = np.stack(components, axis=0)

        output_dir = os.path.join(PROCESSED_DATA_BASE_DIR, f'resolution_{resolution}')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, os.path.basename(image_path).replace('.jpg', f'_{resolution}.npy'))
        np.save(output_path, wavelet_image.astype(np.float16))

        # Update for next iteration
        LL_R, LL_G, LL_B = new_LL
        current_resolution = resolution



from tqdm import trange

def main():
    # List all image files in the directory
    image_files = [f for f in os.listdir(RAW_DATA_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Initialize the progress bar
    for i in trange(len(image_files), desc="Processing images"):
        image_path = os.path.join(RAW_DATA_DIR, image_files[i])
        process_image(image_path)


if __name__ == '__main__':
    main()
