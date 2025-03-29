import os

from tqdm import tqdm
os.environ["TORCH_HOME"] = "/data/user_data/jasonl6/sandeep"
import torch
from torch.utils.data import DataLoader
from dynamics_model import DynamicsModel
from dynamics_model.data import DynamicsModelDataset
from evaluate_utils import *



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
pretrain_ckpt = "/home/jasonl6/sandeep/SimplerEnv-SITCOM/world_model/results/dyna1_simpl_ft_1/vae.pt"
ckpt = torch.load(pretrain_ckpt, map_location="cpu")["model"]
msg = laq.load_state_dict(ckpt)
print(msg)
laq = laq.to(device)
laq.eval()

val_ds = DynamicsModelDataset(["simpler"], image_size=224, mode="val")
val_dl = DataLoader(val_ds, batch_size=96, num_workers=4, shuffle=False)

evaluator = OpticalFlowEvaluator(device=device, image_size=(3, 224, 224))
batch_evaluator = BatchEvaluator(evaluator, device=device)
total_of_loss = 0.0
total_fid_sum = 0.0
num_batches = 0

for idx, val_sample in tqdm(enumerate(val_dl)):
    valid_img, valid_action, valid_future_img = val_sample

    valid_img, valid_action, valid_future_img = (
        valid_img.to(device),
        valid_action.to(device),
        valid_future_img.to(device),
    )

    recons = laq(valid_img, valid_action, valid_future_img, return_recons_only=True)
    # Evaluate dataloader
        
    # Move to device
    base_batch = valid_img.to(device)
    pred_batch = recons.to(device)
    gt_batch = valid_future_img.to(device)
    
    # Evaluate batch
    batch_results = batch_evaluator.evaluate_batch(base_batch, pred_batch, gt_batch)
    
    # Accumulate metrics
    total_of_loss += batch_results["optical_flow_loss"]
    total_fid_sum += batch_results["fid_score"]
    print(f'{idx}  {batch_results["fid_score"]}')
    num_batches += 1
    
# Compute average metrics
avg_of_loss = total_of_loss / num_batches
avg_batch_fid = total_fid_sum / num_batches 

print(f"avg_optical_flow_loss: {avg_of_loss}")
print(f"Average Batch FID: {avg_batch_fid}")
print(f'Global FID: {batch_results["fid_score"]}')

# avg_optical_flow_loss: 0.9921017951435513
# Average Batch FID: 15.052823066711426
# Global FID: 11.156933784484863

# avg_optical_flow_loss: 1.6658014986250136
# Average Batch FID: 21.807578404744465
# Global FID: 17.02658462524414