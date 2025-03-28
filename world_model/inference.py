import torch
from torch.utils.data import DataLoader
from dynamics_model import DynamicsModel
from dynamics_model.data import DynamicsModelDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
laq = DynamicsModel(
    dim=768,
    image_size=224,
    patch_size=14,
    spatial_depth=8,  # 8
    dim_head=64,
    heads=16,
    use_lpips_loss=True,
)
pretrain_ckpt = "results/dyna1_bridge_4f4c35d2/vae.pt"
ckpt = torch.load(pretrain_ckpt, map_location="cpu")["model"]
msg = laq.load_state_dict(ckpt)
print(msg)
laq = laq.to(device)
laq.eval()

val_ds = DynamicsModelDataset(["simpler"], 256, mode="val")
val_dl = DataLoader(val_ds, batch_size=1, num_workers=4, shuffle=False)

for idx, val_sample in enumerate(val_dl):
    valid_img, valid_action, valid_future_img = val_sample

    valid_img, valid_action, valid_future_img = (
        valid_img.to(device),
        valid_action.to(device),
        valid_future_img.to(device),
    )

    recons = laq(valid_img, valid_action, valid_future_img, return_recons_only=True)