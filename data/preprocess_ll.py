import os
import pickle
import numpy as np
import pywt
from tqdm import trange

RAW_CIFAR_PATH = "data/raw/cifar-10-batches-py"
SAVE_DIR = "data/processed_cifar10_ll/resolution_16"
os.makedirs(SAVE_DIR, exist_ok=True)

def unpickle(file):
    """
    Loads and returns a Python object serialized in a pickle file.

    The function opens a specified binary file, deserializes its content
    using the `pickle.load` method, and returns the deserialized object.
    The function assumes the file provided contains a binary representation
    of a pickled Python object. It uses `encoding='bytes'` for compatibility
    during the unpickling process.

    :param file: Path to the binary pickle file.
    :type file: str
    :return: Deserialized Python object from the pickle file.
    :rtype: Any
    """
    with open(file, 'rb') as fo:
        return pickle.load(fo, encoding='bytes')

def extract_ll(image_array):
    """
    Extracts the low-low (LL) frequency components for each channel of the given
    image array using Discrete Wavelet Transform (DWT).

    This function processes a three-channel image data, computes the LL
    frequency components for each channel independently using the 'haar' wavelet,
    and returns a 3D array of LL components. The processed data is then converted
    to 16-bit floating-point format for compact representation.

    :param image_array: A 3D numpy array of shape (3, height, width) representing
        the RGB channels of an image.
    :return: A 3D numpy array of shape (3, new_height, new_width) containing
        the low-low (LL) frequency components extracted from each channel. The
        data is represented in 16-bit floating-point format.
    """
    LL_channels = []
    for c in range(3):
        LL, _ = pywt.dwt2(image_array[c], 'haar')
        LL_channels.append(LL)
    LL = np.stack(LL_channels, axis=0)
    return LL.astype(np.float16)

def process_batches():
    """
    Processes data batches to extract features and save them to a specified directory.

    This function processes CIFAR-like dataset batches, extracting features from each
    image using the specified `extract_ll` function and saves those features as .npy files.
    The processing iterates through a predefined set of batch files, reshaping each
    image's data, performing feature extraction, and incrementally saving the processed
    features in a sequential numerical order.

    :raises FileNotFoundError: If the file path for any batch file does not exist or is inaccessible.
    :raises KeyError: If the required data key is not present in the unpickled batch file.
    :raises IOError: If there is an issue saving the extracted features to the directory.

    :return: None
    """
    batch_files = [f"data_batch_{i}" for i in range(1, 6)] + ["test_batch"]
    counter = 0

    for batch_file in batch_files:
        data_dict = unpickle(os.path.join(RAW_CIFAR_PATH, batch_file))
        raw_images = data_dict[b'data']  # shape (10000, 3072)

        for i in trange(raw_images.shape[0], desc=f"Processing {batch_file}"):
            img = raw_images[i].reshape(3, 32, 32)
            LL = extract_ll(img)
            np.save(os.path.join(SAVE_DIR, f"{counter:05}.npy"), LL)
            counter += 1

if __name__ == "__main__":
    process_batches()
