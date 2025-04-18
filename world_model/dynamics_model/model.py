from pathlib import Path

from dynamics_model.attention import ContinuousPositionBias, Transformer
from einops import pack, rearrange, repeat
from einops.layers.torch import Rearrange
import lpips
import torch
from torch import nn
import torch.nn.functional as F


def exists(val):
    return val is not None


def pair(val):
    ret = (val, val) if not isinstance(val, tuple) else val
    assert len(ret) == 2
    return ret


def check_for_nan(tensor):
    if torch.isnan(tensor).any():
        raise RuntimeError(f"NaN detected!")


class DynamicsModel(nn.Module):
    def __init__(
        self,
        *,
        dim,
        image_size,
        patch_size,
        spatial_depth,
        dim_head=64,
        heads=8,
        channels=3,
        action_dim=7,
        attn_dropout=0.0,
        ff_dropout=0.0,
        use_lpips_loss=True,
        lipips_loss_weight=1.0,
    ):
        super().__init__()

        self.image_size = pair(image_size)
        self.patch_size = pair(patch_size)
        patch_height, patch_width = self.patch_size
        image_height, image_width = self.image_size
        assert (image_height % patch_height) == 0 and (image_width % patch_width) == 0

        # Spatial Transformer
        self.spatial_rel_pos_bias = ContinuousPositionBias(dim=dim, heads=heads)
        spatial_transformer_kwargs = dict(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            causal=False,
            peg=False,
        )
        self.dec_spatial_transformer = Transformer(depth=spatial_depth, **spatial_transformer_kwargs)

        # Pixels to patch embedding
        self.to_patch_emb = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b h w (c p1 p2)", p1=patch_height, p2=patch_width),
            nn.LayerNorm(channels * patch_height * patch_width),
            nn.Linear(channels * patch_height * patch_width, dim - 1),
            nn.LayerNorm(dim - 1),
        )

        # Patch embedding to tokens
        self.to_pixels = nn.Sequential(
            nn.Linear(dim, channels * patch_width * patch_height),
            Rearrange("b h w (c p1 p2) -> b c (h p1) (w p2)", p1=patch_height, p2=patch_width),
        )

        # Action embedding
        num_patch_h, num_patch_w = self.patch_height_width
        self.to_action_emb = nn.Sequential(
            nn.Linear(action_dim, 4096),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, num_patch_h * num_patch_w),
            Rearrange("b (h w) -> b h w 1", h=num_patch_h, w=num_patch_w),
        )

        # Perceptual loss
        self.use_lpips_loss = use_lpips_loss
        self.lpips_loss_weight = lipips_loss_weight
        if self.use_lpips_loss:
            self.loss_fn_lpips = lpips.LPIPS(net="vgg").requires_grad_(False)
        else:
            self.loss_fn_lpips = None

    def state_dict(self, *args, **kwargs):
        return super().state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return super().load_state_dict(*args, **kwargs, strict=False)

    def load(self, path):
        path = Path(path)
        assert path.exists()
        pt = torch.load(str(path))
        pt = {k.replace("module.", "") if "module." in k else k: v for k, v in pt.items()}
        msg = self.load_state_dict(pt)
        print(msg)

    @property
    def patch_height_width(self):
        return self.image_size[0] // self.patch_size[0], self.image_size[1] // self.patch_size[1]

    def decode(
        self,
        tokens,
    ):
        b = tokens.shape[0]
        h, w = self.patch_height_width
        tokens = rearrange(tokens, "b h w d -> b 1 h w d")
        video_shape = tuple(tokens.shape[:-1])

        # decode - spatial
        tokens = rearrange(tokens, "b t h w d -> (b t) (h w) d")
        attn_bias = self.spatial_rel_pos_bias(h, w, device=tokens.device)
        tokens = self.dec_spatial_transformer(tokens, attn_bias=attn_bias, video_shape=video_shape)
        tokens = rearrange(tokens, "(b t) (h w) d -> (b t) h w d", b=b, h=h, w=w)  # t = 1, [b, h, w, d]
        recon_video = self.to_pixels(tokens)

        return recon_video

    def forward(
        self,
        image,
        action,
        future_image,
        step=0,
        return_recons_only=False,
    ):
        assert image.ndim == 4 and action.ndim == 2
        assert image.shape[0] == action.shape[0]
        b, c, *image_dims, device = *image.shape, image.device
        assert tuple(image_dims) == self.image_size

        tokens = self.to_patch_emb(image)
        action_tokens = self.to_action_emb(action)
        tokens = torch.cat([tokens, action_tokens], dim=-1)  # [b, h, w, d]

        recon_image = self.decode(tokens)  # [b, c, h, w]

        if return_recons_only:
            return recon_image

        recon_loss = F.l1_loss(future_image, recon_image)

        lpips_loss = 0.0
        if self.use_lpips_loss:
            lpips_loss = self.loss_fn_lpips.forward(2 * future_image - 1, 2 * recon_image - 1)
            lpips_loss = lpips_loss.mean()

        loss = recon_loss + self.lpips_loss_weight * lpips_loss

        log_dict = {
            "recon_loss": recon_loss.item(),
            "lpips_loss": lpips_loss.item(),
            "loss": loss.item(),
            "step": step,
        }
        check_for_nan(loss)
        return loss, log_dict, recon_image
