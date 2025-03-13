from pathlib import Path
import math

import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange, pack, repeat
from einops.layers.torch import Rearrange
import lpips
from torchvision.models.optical_flow import raft_large

from laq_model.attention import Transformer, ContinuousPositionBias
from laq_model.nsvq import NSVQ
from laq_model.vit import VisionTransformerEncoder, DINOv2Encoder


def exists(val):
    return val is not None


def pair(val):
    ret = (val, val) if not isinstance(val, tuple) else val
    assert len(ret) == 2
    return ret


def check_for_nan(tensor):
    if torch.isnan(tensor).any():
        raise RuntimeError(f"NaN detected!")


class LatentActionQuantizationVideo(nn.Module):
    def __init__(
        self,
        *,
        dim,
        quant_dim,
        codebook_size,
        image_size,
        patch_size,
        spatial_depth,
        temporal_depth,
        dim_head=64,
        heads=8,
        channels=3,
        attn_dropout=0.0,
        ff_dropout=0.0,
        code_seq_len=1,
        use_spatial_feat_condition=False,
        spatial_enc_type="dino_base",  # ["dino_base", "vit_base"]
        use_lpips_loss=True,
        lipips_loss_weight=1.0,
        use_flow_loss=True,
        flow_loss_weight=1.0,
        flow_loss_kickin_step=-1,
    ):
        """
        einstein notations:

        b - batch
        c - channels
        t - time
        d - feature dimension
        p1, p2, pt - image patch sizes and then temporal patch size
        """

        super().__init__()

        self.code_seq_len = code_seq_len
        self.image_size = pair(image_size)
        self.patch_size = pair(patch_size)
        patch_height, patch_width = self.patch_size

        self.spatial_rel_pos_bias = ContinuousPositionBias(dim=dim, heads=heads)

        image_height, image_width = self.image_size
        assert (image_height % patch_height) == 0 and (image_width % patch_width) == 0

        spatial_transformer_kwargs = dict(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            causal=False,
            peg=False,
        )

        # only temporal transformers have PEG and are causal

        temporal_causal_transformer_kwargs = dict(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            causal=True,
            peg=True,
            peg_causal=True,
        )

        # temporal_noncausal_transformer_kwargs = dict(
        #     dim=dim,
        #     dim_head=dim_head,
        #     heads=heads,
        #     attn_dropout=attn_dropout,
        #     ff_dropout=ff_dropout,
        #     causal=False,
        #     peg=True,
        #     peg_causal=False,
        # )

        spatial_enc_type = spatial_enc_type.split("_")
        vit_size = "_".join(spatial_enc_type[1:])
        self.spatial_enc_type = spatial_enc_type[0]
        if self.spatial_enc_type == "dino":
            self.enc_spatial_transformer = DINOv2Encoder(
                image_size=image_size,
                patch_size=patch_size,
                vit_size=vit_size,
            )
        elif self.spatial_enc_type == "vit":
            self.enc_spatial_transformer = VisionTransformerEncoder(
                image_size=image_size,
                patch_size=patch_size,
                vit_size=vit_size,
            )
        else:
            raise ValueError("Invalid spatial_enc_type. Choose 'dino' or 'vit'.")
        self.enc_temporal_transformer = Transformer(depth=temporal_depth, **temporal_causal_transformer_kwargs)

        self.vq = NSVQ(
            dim=dim,
            num_embeddings=codebook_size,
            embedding_dim=quant_dim,
            # device="cuda",  # default device is cpu
            code_seq_len=code_seq_len,
            patch_size=patch_size,
            image_size=image_size,
        )

        self.use_spatial_feat_condition = use_spatial_feat_condition
        self.dec_spatial_transformer = Transformer(depth=spatial_depth, **spatial_transformer_kwargs)
        self.dec_temporal_transformer = Transformer(depth=temporal_depth, **temporal_causal_transformer_kwargs)
        self.to_pixels_rest_frame = nn.Sequential(
            nn.Linear(dim, channels * patch_width * patch_height),
            Rearrange("b t h w (c p1 p2) -> b c t (h p1) (w p2)", p1=patch_height, p2=patch_width),
        )
        self.combine_action_spatial_tokens = nn.Sequential(
            nn.Linear(2 * dim, 4096), nn.LeakyReLU(0.1), nn.Linear(4096, dim)
        )

        # perceptual loss
        self.use_lpips_loss = use_lpips_loss
        self.lpips_loss_weight = lipips_loss_weight
        if self.use_lpips_loss:
            self.loss_fn_lpips = lpips.LPIPS(net="vgg").requires_grad_(False)
        else:
            self.loss_fn_lpips = None

        self.use_flow_loss = use_flow_loss
        self.flow_loss_weight = flow_loss_weight
        self.flow_loss_kickin_step = flow_loss_kickin_step
        if self.use_flow_loss:
            self.flow_model = raft_large(pretrained=True, progress=False).eval().requires_grad_(False)

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

    def decode_from_codebook_indices(self, indices):
        codes = self.vq.codebook[indices]

        return self.decode(codes)

    @property
    def patch_height_width(self):
        return self.image_size[0] // self.patch_size[0], self.image_size[1] // self.patch_size[1]

    def encode(self, tokens):
        b = tokens.shape[0]
        h, w = self.patch_height_width

        tokens = rearrange(tokens, "b t c h w -> (b t) c h w")

        video_tokens, tokens = self.enc_spatial_transformer(tokens)

        video_tokens = rearrange(video_tokens, "(b t) (h w) d -> b t h w d", b=b, h=h, w=w)
        tokens = rearrange(tokens, "(b t) (h w) d -> b t h w d", b=b, h=h, w=w)
        video_shape = tuple(tokens.shape[:-1])

        tokens = rearrange(tokens, "b t h w d -> (b h w) t d")

        tokens = self.enc_temporal_transformer(tokens, video_shape=video_shape)

        tokens = rearrange(tokens, "(b h w) t d -> b t h w d", b=b, h=h, w=w)

        first_video_tokens = video_tokens[:, :1]
        rest_video_tokens = video_tokens[:, 1:]
        first_tokens = tokens[:, :1]
        rest_tokens = tokens[:, 1:]

        return first_tokens, rest_tokens, first_video_tokens, rest_video_tokens

    def decode(
        self,
        tokens,
    ):
        b = tokens.shape[0]
        h, w = self.patch_height_width

        if tokens.ndim == 3:
            tokens = rearrange(tokens, "b (t h w) d -> b t h w d", h=h, w=w)

        video_shape = tuple(tokens.shape[:-1])

        # decode - temporal

        tokens = rearrange(tokens, "b t h w d -> (b h w) t d")

        tokens = self.dec_temporal_transformer(tokens, video_shape=video_shape)

        tokens = rearrange(tokens, "(b h w) t d -> b t h w d", b=b, h=h, w=w)

        # decode - spatial

        tokens = rearrange(tokens, "b t h w d -> (b t) (h w) d")

        attn_bias = self.spatial_rel_pos_bias(h, w, device=tokens.device)

        tokens = self.dec_spatial_transformer(tokens, attn_bias=attn_bias, video_shape=video_shape)

        tokens = rearrange(tokens, "(b t) (h w) d -> b t h w d", b=b, h=h, w=w)

        rest_frames_tokens = tokens

        recon_video = self.to_pixels_rest_frame(rest_frames_tokens)

        return recon_video

    def forward(
        self,
        video,
        step=0,
        mask=None,
        return_recons_only=False,
        return_only_codebook_ids=False,
        inference_mode=False
    ):
        if inference_mode:
            self.inference(video, step=step, mask=mask, return_only_codebook_ids=return_only_codebook_ids)

        assert video.ndim in {4, 5}

        is_image = video.ndim == 4

        if is_image:
            video = rearrange(video, "b c h w -> b c 1 h w")
            assert not exists(mask)

        b, c, f, *image_dims, device = *video.shape, video.device

        assert tuple(image_dims) == self.image_size
        assert not exists(mask) or mask.shape[-1] == f

        first_frame, rest_frames = video[:, :, :1], video[:, :, 1:]

        tokens = rearrange(video, "b c t h w -> b t c h w")

        first_tokens, last_tokens, first_frame_tokens, rest_frame_tokens = self.encode(tokens)
        shape = first_tokens.shape
        *_, h, w, _ = shape

        first_tokens, first_packed_fhw_shape = pack([first_tokens], "b * d")
        last_tokens, last_packed_fhw_shape = pack([last_tokens], "b * d")

        vq_mask = None
        if exists(mask):
            vq_mask = mask[:, 1:]
        # self.lookup_free_quantization = False
        # vq_kwargs = dict(mask = vq_mask) if not self.lookup_free_quantization else dict()

        # TODO: perplexity and codebook_usage should be fixed when using mask
        tokens, perplexity, codebook_usage, indices, num_unique_indices = self.vq(
            first_tokens, last_tokens, mask_last=vq_mask, codebook_training_only=False, condn_first=False
        )

        if ((step % 10 == 0 and step < 100)  or (step % 100 == 0 and step < 1000) or (step % 500 == 0 and step < 5000)) and step != 0:
            print(f"update codebook {step}")
            # self.vq.replace_unused_codebooks(b * (f - 1))
            self.vq.replace_unused_codebooks_sync(b * (f - 1))

        if return_only_codebook_ids:
            return indices

        if math.sqrt(self.code_seq_len) % 1 == 0:  # "code_seq_len should be square number"
            action_h = int(math.sqrt(self.code_seq_len))
            action_w = int(math.sqrt(self.code_seq_len))
        elif self.code_seq_len == 2:
            action_h = 2
            action_w = 1
        else:
            ## error
            print("code_seq_len should be square number or defined as 2")
            return

        # condition on first T-1 tokens, predict last T-1 tokens
        if self.use_spatial_feat_condition:
            condn_tokens_first = rearrange(first_tokens, "b (t h w) d -> b t h w d", h=h, w=w)
            condn_tokens_last = rearrange(last_tokens, "b (t h w) d -> b t h w d", h=h, w=w)
            condn_tokens = torch.cat([condn_tokens_first, condn_tokens_last], dim=1)
            condn_tokens = condn_tokens[:, :-1]
        else:
            condn_tokens_first = first_frame_tokens  # .detach() #+ tokens  [b,t,h,w,d]
            condn_tokens_last = rest_frame_tokens
            condn_tokens = torch.cat([condn_tokens_first, condn_tokens_last], dim=1)
            condn_tokens = condn_tokens[:, :-1]
        # upsample tokens to match the spatial size of the condn_tokens
        tokens = rearrange(tokens, "b (t h w) d -> (b t) d h w", h=action_h, w=action_w)
        scale_factor = (h // action_h, w // action_w)
        tokens = F.interpolate(tokens, scale_factor=scale_factor, mode="nearest")
        tokens = rearrange(tokens, "(b t) d h w -> b t h w d", b=b, h=h, w=w)
        concat_tokens = torch.cat([condn_tokens, tokens], dim=-1)  # [b,t,h,w,2d]
        concat_tokens = self.combine_action_spatial_tokens(concat_tokens)
        recon_video = self.decode(concat_tokens)  # [b c t h w]

        video = rest_frames
        if exists(mask):
            mask = mask[:, 1:]

        if return_recons_only:
            return recon_video

        if exists(mask):
            # variable lengthed video / images training
            recon_loss = F.l1_loss(video, recon_video, reduction="none")
            recon_loss = recon_loss[repeat(mask, "b t -> b c t", c=c)]
            recon_loss = recon_loss.mean()
        else:
            recon_loss = F.l1_loss(video, recon_video)

        video_flattened = rearrange(video, "b c t h w -> (b t) c h w")
        recon_video_flattened = rearrange(recon_video, "b c t h w -> (b t) c h w")

        lpips_loss = 0.0
        if self.use_lpips_loss:
            lpips_loss = self.loss_fn_lpips.forward(2 * video_flattened - 1, 2 * recon_video_flattened - 1)
            if exists(mask):
                lpips_loss = lpips_loss.squeeze()[rearrange(mask, "b t -> (b t)")]
            lpips_loss = lpips_loss.mean()

        flow_loss = 0.0
        if self.use_flow_loss:
            flow_loss = self.flow_loss_fn(video, recon_video, mask)

        loss = recon_loss + self.lpips_loss_weight * lpips_loss
        if step >= self.flow_loss_kickin_step:
            loss += self.flow_loss_weight * flow_loss
        log_dict = {
            "recon_loss": recon_loss.item(),
            "lpips_loss": lpips_loss.item(),
            "flow_loss": flow_loss.item(),
            "loss": loss.item(),
            "perplexity": perplexity,
            "num_unique_indices": num_unique_indices,
            "step": step,
        }
        check_for_nan(loss)
        return loss, log_dict

    def inference(self, video, step=0, mask=None, return_only_codebook_ids=False, user_action_token_num=None):

        assert video.ndim in {4, 5}

        is_image = video.ndim == 4

        if is_image:
            video = rearrange(video, "b c h w -> b c 1 h w")
            assert not exists(mask)

        b, c, f, *image_dims, device = *video.shape, video.device

        assert tuple(image_dims) == self.image_size
        assert not exists(mask) or mask.shape[-1] == f

        first_frame, rest_frames = video[:, :, :1], video[:, :, 1:]

        tokens = rearrange(video, "b c t h w -> b t c h w")

        first_tokens, last_tokens, first_frame_tokens, rest_frame_tokens = self.encode(tokens)
        shape = first_tokens.shape
        *_, h, w, _ = shape

        # quantize
        first_tokens, first_packed_fhw_shape = pack([first_tokens], "b * d")
        last_tokens, last_packed_fhw_shape = pack([last_tokens], "b * d")

        if user_action_token_num is not None:
            raise NotImplementedError("user_action_token_num is not implemented yet")
            tokens, indices = self.vq.inference(
                first_tokens, last_tokens, user_action_token_num=user_action_token_num, condn_first=False
            )
        else:
            tokens, indices = self.vq.inference(first_tokens, last_tokens, condn_first=False)
            # tokens, indices = self.vq(first_tokens, last_tokens, condn_first=False, inference=True)

        if return_only_codebook_ids:
            return indices

        if math.sqrt(self.code_seq_len) % 1 == 0:  # "code_seq_len should be square number"
            action_h = int(math.sqrt(self.code_seq_len))
            action_w = int(math.sqrt(self.code_seq_len))
        elif self.code_seq_len == 2:
            action_h = 2
            action_w = 1
        else:
            print("code_seq_len should be square number or defined as 2")
            return

        tokens = rearrange(tokens, "b (t h w) d -> b t h w d", h=action_h, w=action_w)
        if self.use_spatial_feat_condition:
            concat_tokens = rearrange(first_tokens, "b (t h w) d -> b t h w d", h=h, w=w)
        else:
            concat_tokens = first_frame_tokens  # .detach() #+ tokens
        recon_video = self.decode(concat_tokens, actions=tokens)
        returned_recon = rearrange(recon_video, "b c 1 h w -> b c h w")
        video = rest_frames

        return returned_recon

    def get_flow(self, vid0, vid1):
        unflattened_vid0 = rearrange(vid0, "b c t h w -> (b t) c h w")
        unflattened_vid1 = rearrange(vid1, "b c t h w -> (b t) c h w")
        unflattend_flow = self.flow_model(
            2 * unflattened_vid0 - 1,
            2 * unflattened_vid1 - 1
        )[-1]  # get flow from the last raft iter
        flow = rearrange(unflattend_flow, "(b t) c h w -> b c t h w", b=vid0.shape[0])
        return flow

    def flow_loss_fn(self, video, recon_video, mask=None):
        flow_prev_gt = self.get_flow(video[:, :, 1:], video[:, :, :-1])
        flow_next_gt = self.get_flow(video[:, :, :-1], video[:, :, 1:])

        flow_prev_recon = self.get_flow(recon_video[:, :, 1:], recon_video[:, :, :-1])
        flow_next_recon = self.get_flow(recon_video[:, :, :-1], recon_video[:, :, 1:])

        if exists(mask):
            loss = F.l1_loss(flow_prev_gt, flow_prev_recon, reduction="none") + F.l1_loss(
                flow_next_gt, flow_next_recon, reduction="none"
            )
            flow_mask = self.calculate_flow_loss_mask(mask)
            loss = loss[repeat(flow_mask, "b t -> b c t", c=2)]
            loss = loss.mean()
        else:
            loss = F.l1_loss(flow_prev_gt, flow_prev_recon) + F.l1_loss(flow_next_gt, flow_next_recon)
        loss = 0.5 * loss

        return loss

    def calculate_flow_loss_mask(self, mask):
        flow_mask = torch.logical_and(mask[:, 1:], mask[:, :-1])
        return flow_mask
