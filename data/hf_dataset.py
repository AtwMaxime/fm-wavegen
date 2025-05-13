import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset

class HFDataset(Dataset):
    """
    Handles loading, normalization, and optional transformation of a dataset containing
    low-frequency (LL) and high-frequency (HF) components, stored in .npy or .npz files
    at different resolutions. Facilitates integration into deep learning pipelines by
    converting data to PyTorch tensors.

    The main purpose of this class is to provide a PyTorch-compatible dataset for training
    or evaluation of models that require multi-resolution data. The dataset supports data
    normalization using precomputed mean and standard deviation values per resolution.

    :ivar dir: Path to the dataset directory for a specific resolution.
    :type dir: str
    :ivar files: List of filenames (npy/npz) in the dataset directory.
    :type files: list
    :ivar transform: Optional transformation function to apply to the low-frequency (LL)
        component of the data.
    :type transform: callable or None
    :ivar to_tensor: Flag indicating whether the data should be converted to PyTorch tensors.
    :type to_tensor: bool
    :ivar ll_mean: Precomputed mean values for the low-frequency (LL) component.
    :type ll_mean: torch.Tensor
    :ivar ll_std: Precomputed standard deviation values for the low-frequency (LL) component.
    :type ll_std: torch.Tensor
    :ivar hf_mean: Precomputed mean values for the high-frequency (HF) component.
    :type hf_mean: torch.Tensor
    :ivar hf_std: Precomputed standard deviation values for the high-frequency (HF) component.
    :type hf_std: torch.Tensor
    """
    def __init__(self, base_dir, resolution, stats_path='wavelet_stats.json', transform=None, to_tensor=True):
        """
        Args:
            base_dir (str): Base path like 'data/processed'
            resolution (int): One of [16, 32, 64, 128, 256]
            stats_path (str): Path to JSON file with mean/std per resolution
            transform: Optional transform for LL
            to_tensor (bool): Convert numpy arrays to torch.Tensor
        """
        self.dir = os.path.join(base_dir, f'resolution_{resolution}')
        self.files = [f for f in os.listdir(self.dir) if f.endswith('.npy') or f.endswith('.npz')]
        self.transform = transform
        self.to_tensor = to_tensor

        # Load normalization stats for this resolution
        with open(stats_path, 'r') as f:
            stats = json.load(f)[f"resolution_{resolution}"]

        self.ll_mean = torch.tensor(stats['mean'][:3])
        self.ll_std = torch.tensor(stats['std'][:3])
        self.hf_mean = torch.tensor(stats['mean'][3:])
        self.hf_std = torch.tensor(stats['std'][3:])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.dir, self.files[idx])
        data = np.load(path)
        if isinstance(data, np.lib.npyio.NpzFile):
            data = data['data']

        LL = data[0:3]   # (3, H, W)
        HF = data[3:]    # (9, H, W)

        # To torch.Tensor
        LL = torch.from_numpy(LL).float()
        HF = torch.from_numpy(HF).float()

        # Normalize
        LL = (LL - self.ll_mean[:, None, None]) / (self.ll_std[:, None, None] + 1e-6)
        HF = (HF - self.hf_mean[:, None, None]) / (self.hf_std[:, None, None] + 1e-6)

        if self.transform:
            LL = self.transform(LL)

        return LL, HF

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
