#!/usr/bin/env python3
import os
import torch
import torchdiffeq
from torchvision.transforms.functional import to_pil_image
from data.ll_dataset import LLDataset, inverse_normalize
from models.ll_unet import UNetModelWrapper

# Configuration (edit paths and parameters as needed)
DEVICE      = 'cuda'
CKPT_PATH   = 'models/checkpoints/ll_models/ll_model_step_195000.pt'
OUT_DIR     = 'outputs/ll_samples'
TOTAL_SAMPLES = 10000   # total number to generate
BATCH_SIZE    = 100     # generate this many at once
ODE_STEPS     = 25

MODEL_CONFIG = {
    'dim': (3, 16, 16),
    'num_channels': 128,
    'channel_mult': [2, 4],
    'num_res_blocks': 3,
    'num_heads': 6,
    'num_head_channels': 64,
    'attention_resolutions': "16,8",
    'dropout': 0
}


def rescale_tensor_to_255(x: torch.Tensor) -> torch.Tensor:
    x_rescaled = []
    for img in x:
        img_np = img.cpu().numpy()
        for c in range(img_np.shape[0]):
            ch = img_np[c]
            mn, mx = ch.min(), ch.max()
            if mx > mn:
                img_np[c] = 255 * (ch - mn) / (mx - mn)
            else:
                img_np[c] = 0
        x_rescaled.append(torch.tensor(img_np))
    return torch.stack(x_rescaled).clamp(0, 255).byte()


def load_model(ckpt_path: str, device: torch.device) -> torch.nn.Module:
    state = torch.load(ckpt_path, map_location=device)
    model = UNetModelWrapper(**MODEL_CONFIG).to(device)
    model.load_state_dict(state['ema_model'])
    model.eval()
    return model


def main():
    device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')

    # dataset stats for denormalization
    ds = LLDataset(
        "data/processed_cifar10_ll/resolution_16",
        normalize=True,
        stats_path="ll_stats.json"
    )

    model = load_model(CKPT_PATH, device)
    os.makedirs(OUT_DIR, exist_ok=True)

    t_span = torch.linspace(0, 1, ODE_STEPS, device=device)

    def fm_func(t, x):
        B = x.shape[0]
        return model(x=x, t=torch.full((B,), t.item(), device=device))

    generated = 0
    while generated < TOTAL_SAMPLES:
        cur_batch = min(BATCH_SIZE, TOTAL_SAMPLES - generated)
        noise = torch.randn(cur_batch, *MODEL_CONFIG['dim'], device=device)
        with torch.no_grad():
            traj = torchdiffeq.odeint(fm_func, noise, t_span, rtol=1e-4, atol=1e-4)
            out = traj[-1]
        # denormalize & rescale
        ll = inverse_normalize(out, ds.mean.to(device), ds.std.to(device))
        ll255 = rescale_tensor_to_255(ll)

        # save each sample
        for i, img in enumerate(ll255):
            pil = to_pil_image(img)
            idx = generated + i
            pil.save(os.path.join(OUT_DIR, f"sample_{idx:05d}.png"))
        generated += cur_batch
        print(f"Saved {generated}/{TOTAL_SAMPLES} samples")


if __name__ == '__main__':
    main()
