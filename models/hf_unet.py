import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel


class ResolutionConditioning(nn.Module):
    def __init__(self, resolutions=(16, 32, 64, 128, 256), embed_dim=256):
        super().__init__()
        self.id_map = {res: i for i, res in enumerate(resolutions)}
        self.embed = nn.Embedding(len(resolutions), embed_dim)

    def forward(self, resolution_tensor):
        if not torch.is_tensor(resolution_tensor):
            resolution_tensor = torch.tensor(resolution_tensor, dtype=torch.long)

        resolution_tensor = resolution_tensor.to(next(self.parameters()).device)
        res_ids = torch.tensor([self.id_map[int(r)] for r in resolution_tensor.tolist()],
                               device=resolution_tensor.device)
        return self.embed(res_ids).unsqueeze(1)  # (B, 1, D)


class HFUNet(nn.Module):
    def __init__(
        self,
        in_channels=12,
        out_channels=12,
        cond_dim=256,
        block_out_channels=(128, 256, 512),
        layers_per_block=2,
        attention_head_dim=64,
        resolutions=(16, 32, 64, 128, 256),
    ):
        super().__init__()
        self.res_condition = ResolutionConditioning(resolutions=resolutions, embed_dim=cond_dim)

        self.unet = UNet2DConditionModel(
            in_channels=in_channels,
            out_channels=out_channels,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            down_block_types=("CrossAttnDownBlock2D",) * len(block_out_channels),
            up_block_types=("CrossAttnUpBlock2D",) * len(block_out_channels),
            mid_block_type="UNetMidBlock2DCrossAttn",
            only_cross_attention=False,  # Enables self + cross attention
            cross_attention_dim=cond_dim,
            attention_head_dim=attention_head_dim,
            norm_num_groups=32,
        )

        self._ensure_affine_groupnorm()

    def _ensure_affine_groupnorm(self):
        for m in self.unet.modules():
            if isinstance(m, nn.GroupNorm):
                if not hasattr(m, "weight") or m.weight is None:
                    m.weight = nn.Parameter(torch.ones(m.num_channels))
                if not hasattr(m, "bias") or m.bias is None:
                    m.bias = nn.Parameter(torch.zeros(m.num_channels))
                m.affine = True

    def forward(self, x, t, resolution):
        """
        Args:
            x: (B, in_channels, H, W)
            t: (B,) timestep
            resolution: list[int] or tensor[int] of size B
        """
        cond = self.res_condition(resolution)  # (B, 1, cond_dim)
        out = self.unet(x, t, encoder_hidden_states=cond).sample  # (B, out_channels, H, W)
        return out
