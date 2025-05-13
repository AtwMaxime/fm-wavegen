import torch
from torch.utils.data import DataLoader
from data.hf_dataset import HFDataset, inverse_normalize
from utils.inverse_wavelet import batch_iwt
from utils.LL_to_image import to_image
from models.hf_unet import HFUNet
from torchdiffeq import odeint
from torchvision.utils import make_grid, save_image
import os
import glob
import tqdm
import copy
from utils.fid import calculate_fid_between_folders


def find_latest_checkpoint(checkpoint_dir):
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "ckpt_step_*.pt"))
    if not checkpoint_files:
        raise ValueError(f"No checkpoints found in {checkpoint_dir}")
    latest = max(checkpoint_files, key=os.path.getctime)
    print(f"🔍 Found latest checkpoint: {latest}")
    return latest

def generate_images(model, dataloader, dataset, output_dir, target_samples=5000):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    count = 0
    device = next(model.parameters()).device

    for ll, hf in tqdm.tqdm(dataloader, desc="Generating images"):
        if count >= target_samples:
            break
        ll = ll.to(device)
        noise = torch.randn_like(hf).to(device)

        input = torch.cat([ll, noise], dim=1)
        t_span = torch.linspace(0, 1, 50).to(device)

        with torch.no_grad():
            traj = torchdiffeq.odeint(
                lambda t, x: model(t, x, resolution=torch.full((x.shape[0],), ll.shape[-1], device=x.device)),
                input, t_span, rtol=1e-4, atol=1e-4
            )
            xt = traj[-1][:, 3:]
            ll_inv = inverse_normalize(ll, dataset.ll_mean.cuda(), dataset.ll_std.cuda())
            hf_inv = inverse_normalize(xt, dataset.hf_mean.cuda(), dataset.hf_std.cuda())
            recon = batch_iwt(ll_inv, hf_inv)

            for img in recon:
                img_pil = to_image(img)
                img_pil.save(os.path.join(output_dir, f"{count:05}.png"))
                count += 1
                if count >= target_samples:
                    break
    print(f"✅ Generated {count} images into {output_dir}")

if __name__ == "__main__":

    for resolution in [16,32,64,128,256]:
        checkpoint_dir = f"models/checkpoints/hf_models_res{resolution}"
        output_dir = f"outputs/generated_res{resolution}"

        batch_size = 32
        if resolution == 64:
            batch_size = 16
        elif resolution == 128:
            batch_size = 8
        elif resolution == 256:
            batch_size = 4

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)

        dataset = HFDataset("data/processed", resolution=resolution)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)

        model = HFUNet(
            in_channels=12,
            out_channels=12,
            cond_dim=256,
            block_out_channels=(128, 256, 512),
            layers_per_block=2,
            attention_head_dim=64,
        ).cuda()

        checkpoint_path = find_latest_checkpoint(checkpoint_dir)
        ckpt = torch.load(checkpoint_path, map_location="cuda")
        model.load_state_dict(ckpt["ema_model"])

        generate_images(model, dataloader, dataset, output_dir, target_samples=5000)

        calculate_fid_between_folders(
            path_real=f"data/processed/resolution_{resolution}",
            path_fake=output_dir,
            output_txt="outputs/fid_results.txt"
        )
