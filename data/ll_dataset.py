import os
import numpy as np
import torch
from torch.utils.data import Dataset
import json

class LLDataset(Dataset):
    """
    Represents a dataset for loading and preprocessing low-level (LL) files stored in `.npy` format.

    The class provides functionality to load and preprocess LL data from a directory, with optional
    normalization using precomputed statistics and application of transformations. This class is
    particularly useful for scenarios where structured LL data needs to be fed into machine learning
    models or further processed.

    :ivar directory: Path to the directory containing `.npy` LL files.
    :type directory: str
    :ivar files: List of sorted `.npy` files found in the directory.
    :type files: list of str
    :ivar transform: Callable object or function to apply as a transformation to the tensor data. If set
        to None, no transformation will be applied.
    :type transform: callable or None
    :ivar normalize: Indicates whether normalization of the data using precomputed statistics (mean and
        standard deviation) is enabled.
    :type normalize: bool
    :ivar ll_mean: Tensor containing the mean values for each channel of the LL data. This is derived
        from the statistics provided in the `stats_path` file if normalization is enabled. If not
        initialized, its value will be None.
    :type ll_mean: torch.Tensor or None
    :ivar ll_std: Tensor containing the standard deviation values for each channel of the LL data. This
        is derived from the statistics provided in the `stats_path` file if normalization is enabled. If
        not initialized, its value will be None.
    :type ll_std: torch.Tensor or None
    """

    def __init__(self, directory, transform=None, normalize=True, stats_path=None):
        """
        Args:
            directory (str): Path to directory with .npy LL files (shape: 3x16x16).
            transform (callable): Optional transform to apply to the tensor.
            normalize (bool): Whether to normalize with precomputed stats.
            stats_path (str): Optional path to JSON file containing mean/std for LL channels.
        """
        self.directory = directory
        self.files = sorted(f for f in os.listdir(directory) if f.endswith('.npy'))
        self.transform = transform
        self.normalize = normalize


        if normalize:
            with open(stats_path, 'r') as f:
                stats = json.load(f)['resolution_16']
            self.mean = torch.tensor(stats['mean'])  # shape: (3,)
            self.std = torch.tensor(stats['std'])    # shape: (3,)
        else:
            self.mean = self.std = None

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.directory, self.files[idx])
        ll = np.load(path)  # shape: (3, 16, 16)
        ll = torch.from_numpy(ll).float()

        if self.normalize and self.mean is not None:
            ll = (ll - self.mean[:, None, None]) / (self.std[:, None, None] + 1e-6)

        if self.transform:
            ll = self.transform(ll)

        return ll

def inverse_normalize(tensor, mean, std):
    """
    Args:
        tensor: Tensor of shape (C, H, W) or (B, C, H, W)
        mean: Tensor of shape (C,)
        std: Tensor of shape (C,)
    Returns:
        Tensor rescaled to original range
    """
    mean = mean.view(1, -1, 1, 1) if tensor.dim() == 4 else mean.view(-1, 1, 1)
    std = std.view(1, -1, 1, 1) if tensor.dim() == 4 else std.view(-1, 1, 1)
    return tensor * std + mean
