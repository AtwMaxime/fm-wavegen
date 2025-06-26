import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel

class HFUNet(nn.Module):
    def __init__(
        self,
        in_channels=9,              # HF noise channels
        cond_channels=3,            # LL RGB
        out_channels=12,            # 9 HF + 3 LL passthrough
        block_out_channels=(128, 128, 256),
        layers_per_block=2,
        norm_num_groups=32,
    ):
        super().__init__()

        self.unet = UNet2DConditionModel(
            in_channels=in_channels + cond_channels,
            out_channels=out_channels,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            down_block_types=("DownBlock2D",) * len(block_out_channels),
            up_block_types=("UpBlock2D",) * len(block_out_channels),
            mid_block_type="UNetMidBlock2D",
            only_cross_attention=False,
            norm_num_groups=norm_num_groups,
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

    def forward(self, x, t, ll_image):
        """
        Args:
            x: (B, 9, H, W)
            t: (B,)
            ll_image: (B, 3, H, W)
        """
        x_cat = torch.cat([x, ll_image], dim=1)  # (B, 12, H, W)

        # Provide dummy encoder_hidden_states to satisfy UNet2DConditionModel
        batch_size = x.shape[0]
        device = x.device
        dummy = torch.zeros((batch_size, 1, 1), device=device)  # (B, 1, D)

        return self.unet(x_cat, t, encoder_hidden_states=dummy).sample
