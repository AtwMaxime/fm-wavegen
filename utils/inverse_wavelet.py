import torch
import pywt

def batch_iwt(LL, HF, wavelet='haar'):
    """
    Applies inverse DWT (IWT) channel-wise for a batch of images.

    Args:
        LL: Tensor (B, 3, H, W)
        HF: Tensor (B, 9, H, W) — ordered as [LH_R, LH_G, LH_B, HL_R, HL_G, HL_B, HH_R, HH_G, HH_B]
        wavelet: Wavelet name (default = 'haar')

    Returns:
        Tensor (B, 3, 2H, 2W)
    """
    B, C, H, W = LL.shape
    assert C == 3 and HF.shape[1] == 9

    recon = []
    for b in range(B):
        channels = []
        for c in range(3):
            coeffs = (
                LL[b, c].cpu().numpy(),
                (
                    HF[b, c].cpu().numpy(),       # LH
                    HF[b, c + 3].cpu().numpy(),   # HL
                    HF[b, c + 6].cpu().numpy(),   # HH
                )
            )
            rec = pywt.idwt2(coeffs, wavelet=wavelet)
            channels.append(torch.tensor(rec))
        img = torch.stack(channels, dim=0)  # (3, 2H, 2W)
        recon.append(img)

    return torch.stack(recon, dim=0)  # (B, 3, 2H, 2W)
