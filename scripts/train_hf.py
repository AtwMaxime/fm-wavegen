from torch.utils.data import DataLoader
from data.hf_dataset import HFDataset, inverse_normalize
from utils.inverse_wavelet import batch_iwt
from utils.LL_to_image import to_image
from models.hf_unet import HFUNet
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

import os

from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
from torchdiffeq import odeint


def generate_samples_conditional_batch(model, datalooper, dataset, savedir, step, net_name="normal", parallel=False, max_batch=8):
    """
    Generates and saves a grid of reconstructed real vs generated images (batch),
    and a HF filter comparison for a single sample.

    Args:
        model: HFUNet model with .forward(t, x, resolution).
        datalooper: infinite iterator yielding (LL, HF) batches.
        dataset: dataset instance (provides stats for unnormalization).
        savedir: where to save visual outputs.
        step: training step (used for naming).
        net_name: label ('ema', 'normal', etc.).
        parallel: if True, model.module is used.
        max_batch: number of images to show in grid.
    """
    os.makedirs(savedir, exist_ok=True)
    model.eval()
    model_ = model.module if parallel else model
    model_ = model_.cuda()

    ll, hf = next(datalooper)
    ll, hf = ll.cuda(), hf.cuda()
    batch_size = ll.size(0)
    B = min(batch_size, max_batch)

    noise = torch.randn_like(hf)
    input = torch.cat([ll, noise], dim=1)
    t_span = torch.linspace(0, 1, 50).to(ll.device)

    with torch.no_grad():
        traj = odeint(lambda t, x: model_(t, x, resolution=torch.full((x.shape[0],), ll.shape[-1], device=x.device)),
                      input, t_span, rtol=1e-4, atol=1e-4)
        xt = traj[-1][:, 3:]  # only generated HF

        # Inverse normalization
        ll_inv = inverse_normalize(ll, dataset.ll_mean.cuda(), dataset.ll_std.cuda())
        hf_real_inv = inverse_normalize(hf, dataset.hf_mean.cuda(), dataset.hf_std.cuda())
        hf_gen_inv = inverse_normalize(xt, dataset.hf_mean.cuda(), dataset.hf_std.cuda())

        # Reconstruct
        recon_real = batch_iwt(ll_inv[:B], hf_real_inv[:B])  # (B, 3, H, W)
        recon_gen = batch_iwt(ll_inv[:B], hf_gen_inv[:B])

        # === Save image grid ===
        both = torch.cat([recon_real, recon_gen], dim=0)  # (2B, 3, H, W)
        grid = make_grid(both, nrow=B)
        save_path = os.path.join(savedir, f"{net_name}_image_grid_step_{step}.png")
        to_pil_image(grid.byte()).save(save_path)

        # === Save HF filters for 1st sample only ===
        hf_titles = ["LH", "HL", "HH"]
        color_channels = ["R", "G", "B"]

        hf_real_1 = hf_real_inv[0].cpu()
        hf_gen_1 = hf_gen_inv[0].cpu()

        fig, axes = plt.subplots(3, 6, figsize=(18, 9))
        for i, ch in enumerate(color_channels):
            for j, hf_name in enumerate(hf_titles):
                axes[i, j].imshow(hf_real_1[i + j * 3], cmap='gray')
                axes[i, j].set_title(f"Real {hf_name} - {ch}")
                axes[i, j].axis("off")

                axes[i, j + 3].imshow(hf_gen_1[i + j * 3], cmap='gray')
                axes[i, j + 3].set_title(f"Generated {hf_name} - {ch}")
                axes[i, j + 3].axis("off")

        plt.tight_layout()
        hf_path = os.path.join(savedir, f"{net_name}_hf_filters_step_{step}.png")
        plt.savefig(hf_path)
        plt.close()

    model.train()

def freeze_early_blocks(model, num_blocks_to_freeze=1):
    """
    Freezes the first N down_blocks and the corresponding up_blocks of the HFUNet.
    """
    print(f"🔒 Freezing first {num_blocks_to_freeze} down and up blocks")

    for i in range(num_blocks_to_freeze):
        for p in model.unet.down_blocks[i].parameters():
            p.requires_grad = False
        for p in model.unet.up_blocks[-(i+1)].parameters():
            p.requires_grad = False

def infinite_dataloader(dataloader):
    """
    Generator function to endlessly yield batches of data from the provided
    dataloader in a cyclic manner. This is designed for iterating over
    datasets continuously.

    :param dataloader: Input dataloader to iterate over. It should be
        an iterable, such as PyTorch's DataLoader, that provides batches
        of data.
    :type dataloader: iterable
    :return: Yields individual batches from the dataloader endlessly.
    """
    while True:
        for batch in dataloader:
            yield batch

def update_ema(source_model, target_model, decay=0.9999):
    """
    Update the Exponential Moving Average (EMA) of parameters in the target model using the
    parameters of the source model.

    :param source_model: Source model containing the parameters to be averaged. It is the
        model whose parameter updates will influence the EMA target model.
    :type source_model: torch.nn.Module
    :param target_model: Target model whose parameters will maintain the exponentially
        decayed moving average of the source model's parameters.
    :type target_model: torch.nn.Module
    :param decay: Decay rate for updating the EMA parameters. It influences the
        exponential smoothing. The value should be between 0 and 1, where a higher value
        results in slower updates.
    :type decay: float
    :return: None
    """
    for ema_param, param in zip(target_model.parameters(), source_model.parameters()):
        ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)



def generate_samples_conditional_batch(model, datalooper, dataset, savedir, step, net_name="normal", parallel=False):
    """
    Generates and saves sample reconstructions (real vs generated) from the HF model.

    Args:
        model: HFUNet model (with .forward(t, x, resolution)).
        datalooper: infinite iterator yielding (LL, HF) batches.
        dataset: dataset instance (for stats).
        savedir: output directory.
        step: current training step (for naming).
        net_name: string identifier for the model (e.g., 'ema' or 'normal').
        parallel: if True, unwrap model.module
    """
    os.makedirs(savedir, exist_ok=True)
    model.eval()
    model_ = model.module if parallel else model
    model_ = model_.cuda()

    # === Get batch ===
    ll, hf = next(datalooper)
    ll, hf = ll.cuda(), hf.cuda()

    # === Sample noise & solve FM trajectory ===
    noise = torch.randn_like(hf)
    input = torch.cat([ll, noise], dim=1)
    t_span = torch.linspace(0, 1, 25).to(ll.device)

    with torch.no_grad():
        def fm_func(t, x):
            B = x.shape[0]
            return model_(
                x,
                torch.full((B,), t.item(), device=x.device),
                torch.full((B,), ll.shape[-1], device=x.device)
            )

        traj = odeint(fm_func, input, t_span, rtol=1e-4, atol=1e-4)
        xt = traj[-1][:, 3:]  # Keep generated HF only

        # === Inverse normalization ===
        ll_inv = inverse_normalize(ll, dataset.ll_mean.cuda(), dataset.ll_std.cuda())
        hf_real_inv = inverse_normalize(hf, dataset.hf_mean.cuda(), dataset.hf_std.cuda())
        hf_gen_inv = inverse_normalize(xt, dataset.hf_mean.cuda(), dataset.hf_std.cuda())

        # === Reconstruct images ===
        recon_real = batch_iwt(ll_inv, hf_real_inv)
        recon_gen = batch_iwt(ll_inv, hf_gen_inv)

        # === Convert to PIL images ===
        img_real = to_image(recon_real)
        img_gen = to_image(recon_gen)

        # === Save side-by-side real vs generated ===
        # Stack images
        real_tensors = torch.stack(
            [torch.tensor(np.array(img)).permute(2, 0, 1) for img in img_real[:8]])  # (8, 3, H, W)
        gen_tensors = torch.stack([torch.tensor(np.array(img)).permute(2, 0, 1) for img in img_gen[:8]])  # (8, 3, H, W)

        # Create a 2-row grid: real (top), generated (bottom)
        grid = make_grid(torch.cat([real_tensors, gen_tensors], dim=0), nrow=8)

        grid_img = to_pil_image(grid.byte())
        grid_img.save(os.path.join(savedir, f"{net_name}_comparison_step_{step}.png"))

        wandb.log({
            f"{net_name}_img_step_{step}": wandb.Image(grid_img, caption=f"{net_name} reconstruction @ step {step}")
        })

    model.train()


def train_resolution_conditioned_model(
    model,
    optimizer,
    flow_matcher,
    dataloader,
    total_steps,
    save_every=5000,
    save_dir="checkpoints",
    start_step=0,
    grad_clip=1.0,):
    """
    Train a resolution-conditioned model using a specified training loop.

    This function facilitates training a deep learning model conditioned on the
    input image resolution using paired low-level and high-frequency input data.
    Training involves gradient updates, loss computation through flow matching,
    and maintaining an exponential moving average (EMA) of the model parameters.
    The training process dynamically saves checkpoints and loss histories for
    monitoring and continuation purposes.

    :param model: The torch.nn.Module model to be trained.
    :param optimizer: The torch.optim.Optimizer used for training the model.
    :param flow_matcher: Object providing `sample_location_and_conditional_flow`
        method for flow matching computations.
    :param dataloader: DataLoader providing batches of paired low-level
        and high-frequency input data for training.
    :param total_steps: int. The total number of training steps to execute.
    :param resolution: int. Target resolution to condition the model on during training.
    :param save_every: int, optional. Frequency of saving checkpoints.
        Defaults to 5000.
    :param save_dir: str, optional. Directory to save model checkpoints and
        loss histories. Defaults to "checkpoints".
    :param start_step: int, optional. Step number to start training from,
        used for resuming training. Defaults to 0.
    :param grad_clip: float, optional. Maximum allowable value to clip gradient
        norm during training. Defaults to 1.0.
    :return: None
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    ema_model = copy.deepcopy(model)
    scaler = torch.cuda.amp.GradScaler()

    os.makedirs(save_dir, exist_ok=True)

    loss_history = []

    loop = trange(start_step, total_steps, dynamic_ncols=True)


    for step in loop:
        model.train()
        optimizer.zero_grad()

        # === Load batch ===
        ll, hf = next(dataloader)
        ll, hf = ll.to(device), hf.to(device)

        # === Build input and target ===
        noise = torch.randn_like(hf)
        model_input = torch.cat([ll, noise], dim=1)    # (B, 12, H, W)
        target = torch.cat([ll, hf], dim=1)            # (B, 12, H, W)

        # === Flow Matching ===
        with torch.cuda.amp.autocast():  # Automatically casts to float16 when safe
            t, xt, ut = flow_matcher.sample_location_and_conditional_flow(model_input, target)
            vt = model(t=t, x=xt, ll_image=ll)
            loss = ((vt - ut) ** 2).mean()

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        update_ema(model, ema_model)



        loop.set_description(f"[Step {step}] Loss: {loss.item():.4f}")
        loss_history.append(loss.item())

        wandb.log({
            "loss": loss.item(),
            "step": step,
        })

        if step % 1000 == 0:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    wandb.log({f"weights/{name}": wandb.Histogram(param.data.cpu())})
                    if param.grad is not None:
                        wandb.log({f"grads/{name}": wandb.Histogram(param.grad.data.cpu())})

        # === Save ===
        if (save_every and (step + 1) % save_every == 0) or step == 0 :
            ckpt = {
                "model": model.state_dict(),
                "ema_model": ema_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
            }

            torch.save(ckpt, os.path.join(save_dir, f"ckpt_step_{step}.pt"))
            np.save(os.path.join(save_dir, "loss_history.npy"), np.array(loss_history))
            generate_samples_conditional_batch(
                model=ema_model,
                datalooper=dataloader,
                dataset=dataset,
                savedir=save_dir,
                step=step,
                net_name="ema")

from models.hf_unet import HFUNet
from data.hf_dataset import HFDataset
from torch.utils.data import DataLoader
import torch.optim as optim

print("Creating model...")


# Model
model = HFUNet(
    in_channels=9,       # 9 = 3 HF bands × 3 channels
    out_channels=12,     # 9 HF + 3 LL passthrough
    block_out_channels=(128, 256, 512),
    layers_per_block=2,
).to("cuda")

print("Selecting Optimizer and Flow Matching Method...")

# Optimizer
optimizer = optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95))

# Flow Matching method
flow_matcher = ExactOptimalTransportConditionalFlowMatcher(sigma=0)

print("Setting training parameters...")

resolutions = [16, 32, 64,128,256]
steps_per_res = 50000
start_step = 0

wandb.init(
    project="hf-unet-wavelets",  # choose your own name here
    name=f"train_HF",
    config={
        "lr": 1e-4,
        "batch_size": 32,
        "model": "HFUNet",
        "steps_per_res": steps_per_res,
    }
)


print("Starting training...")

for res in resolutions:
    print(f"\n🔁 Training on resolution {res}x{res} for {steps_per_res} steps")

    if res >= 64:
        freeze_early_blocks(model, num_blocks_to_freeze=1)

    dataset = HFDataset("data/processed", resolution=res)
    batch_size = 32
    if res == 64 :
        batch_size = 16
    if res == 128:
        batch_size = 8
    elif res == 256:
        batch_size = 4

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)
    infinite_loader = infinite_dataloader(dataloader)


    train_resolution_conditioned_model(
        model=model,
        optimizer=optimizer,
        flow_matcher=flow_matcher,
        dataloader=infinite_loader,
        total_steps=start_step + steps_per_res,
        save_every=5000,
        save_dir=f"models/checkpoints/hf_models_res{res}",
        start_step=start_step,
    )

    start_step += steps_per_res
