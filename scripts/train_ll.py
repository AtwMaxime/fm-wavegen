import os
import copy
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher
from torch.cuda.amp import GradScaler, autocast
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image
from tqdm import trange
import wandb

import torch
import copy
import numpy as np
from tqdm import trange
from torch.nn.utils import clip_grad_norm_
from torchvision.utils import save_image
import torchdiffeq
from torchdyn.core import NeuralODE
from absl import app, flags
from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher
import tqdm
import wandb
wandb.login(key="deb9f76f43443b51940756901c13e8483c0b50c4")

from data.ll_dataset import LLDataset, inverse_normalize
from models.ll_unet import UNetModelWrapper

def rescale_tensor_to_255(x):
    # x: (B, C, H, W), float
    x_rescaled = []
    for img in x:
        img_np = img.cpu().numpy()
        for c in range(3):
            channel = img_np[c]
            min_val, max_val = channel.min(), channel.max()
            if max_val > min_val:
                img_np[c] = 255 * (channel - min_val) / (max_val - min_val)
            else:
                img_np[c] = 0  # uniform value
        x_rescaled.append(torch.tensor(img_np))
    return torch.stack(x_rescaled).clamp(0, 255).byte()





def infinite_dataloader(dataloader):
    while True:
        for batch in dataloader:
            yield batch

def update_ema(source_model, target_model, decay=0.9999):
    for ema_param, param in zip(target_model.parameters(), source_model.parameters()):
        ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)

def generate_samples(model, dataloader, dataset, savedir, step):
    os.makedirs(savedir, exist_ok=True)
    model.eval()
    ll_noise = next(dataloader).cuda()
    noise = torch.randn_like(ll_noise).cuda()

    t_span = torch.linspace(0, 1, 25).to(noise.device)


    with torch.no_grad():
        def fm_func(t, x):
            B = x.shape[0]
            return model(x=x, t=torch.full((B,), t.item(), device=x.device))

        traj = torchdiffeq.odeint(fm_func, noise, t_span, rtol=1e-4, atol=1e-4)
        out = traj[-1]
        ll_gen = inverse_normalize(out, dataset.mean.cuda(), dataset.std.cuda())
        ll_gen_rescaled = rescale_tensor_to_255(ll_gen)

        grid = make_grid(ll_gen_rescaled[:16], nrow=4)
        img = to_pil_image(grid)
        img.save(os.path.join(savedir, f"ll_samples_step_{step}.png"))
        wandb.log({f"ll_samples_step_{step}": wandb.Image(img)})

    model.train()

def train_ll():
    resolution = 16
    batch_size = 128
    total_steps = 200000
    save_every = 5000

    dataset = LLDataset(
        "data/processed_cifar10_ll/resolution_16",
        normalize=True,
        stats_path="ll_stats.json"
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)
    dataloader = infinite_dataloader(dataloader)

    model = UNetModelWrapper(
        dim=(3, 16, 16),   # (C, H, W) = (3, 89, 89)
        num_channels=128,  # Increase base channels for more capacity
        channel_mult=[2, 4],
        num_res_blocks=3,
        num_heads=6,
        num_head_channels=64,
        attention_resolutions="16,8",
        dropout=0
    ).cuda()
    ema_model = copy.deepcopy(model)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95))
    flow_matcher = ExactOptimalTransportConditionalFlowMatcher(sigma=0)
    scaler = GradScaler()

    wandb.init(
        project="ll-unet-wavelets",
        name="train_LL",
        config={
            "lr": 1e-4,
            "batch_size": batch_size,
            "steps": total_steps,
        }
    )

    loop = trange(total_steps, dynamic_ncols=True)
    for step in loop:
        model.train()
        optimizer.zero_grad()

        target = next(dataloader).cuda()
        noise = torch.randn_like(target)

        with autocast():
            t, xt, ut = flow_matcher.sample_location_and_conditional_flow(noise, target)
            vt = model(t=t, x=xt)
            loss = ((vt - ut) ** 2).mean()

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        update_ema(model, ema_model)

        loop.set_description(f"[Step {step}] Loss: {loss.item():.4f}")
        wandb.log({"loss": loss.item(), "step": step})

        if step % save_every == 0 or step == 0:
            torch.save({
                "model": model.state_dict(),
                "ema_model": ema_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step
            }, f"models/checkpoints/ll_models/ll_model_step_{step}.pt")

            generate_samples(ema_model, dataloader, dataset, "outputs/ll_samples", step)

if __name__ == "__main__":
    train_ll()