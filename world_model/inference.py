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
pretrain_ckpt = "results/dyna1_bridge_4f4c35d2/vae.pt"
ckpt = torch.load(pretrain_ckpt, map_location="cpu")["model"]
msg = laq.load_state_dict(ckpt)
print(msg)
laq = laq.to(device)
laq.eval()

val_ds = DynamicsModelDataset(["simpler"], 256, mode="val")
val_dl = DataLoader(val_ds, batch_size=32, num_workers=4, shuffle=False)

evaluator = OpticalFlowEvaluator(device=device, image_size=(256, 256))
batch_evaluator = BatchEvaluator(evaluator, device=device)
total_of_loss = 0.0
total_fid_sum = 0.0
num_batches = 0

for idx, val_sample in enumerate(val_dl):
    valid_img, valid_action, valid_future_img = val_sample

    valid_img, valid_action, valid_future_img = (
        valid_img.to(device),
        valid_action.to(device),
        valid_future_img.to(device),
    )

    recons = laq(valid_img, valid_action, valid_future_img, return_recons_only=True)
    # Evaluate dataloader

        
    # Reset FID calculator to use it for global FID calculation
    # This way we can accumulate features without storing all images
    evaluator.fid_calculator.reset()
        
    # Move to device
    base_batch = base_batch.to(device)
    pred_batch = pred_batch.to(device)
    gt_batch = gt_batch.to(device)
    
    # Evaluate batch
    batch_results = batch_evaluator.evaluate_batch(base_batch, pred_batch, gt_batch)
    
    # Accumulate metrics
    total_of_loss += batch_results["optical_flow_loss"]
    total_fid_sum += batch_results["fid_score"]
    num_batches += 1
    
    # Prepare images for FID calculation
    if pred_batch.min() < 0:
        pred_batch = (pred_batch + 1) / 2
    if gt_batch.min() < 0:
        gt_batch = (gt_batch + 1) / 2
        
    # Convert to uint8 for FID calculation
    pred_batch_uint8 = (pred_batch * 255).to(torch.uint8)
    gt_batch_uint8 = (gt_batch * 255).to(torch.uint8)
    
    # Update global FID calculator
    batch_evaluator.evaluator.fid_calculator.update(pred_batch_uint8, real=False)
    batch_evaluator.evaluator.fid_calculator.update(gt_batch_uint8, real=True)
        
    # Compute global FID
    global_fid = batch_evaluator.evaluator.fid_calculator.compute()
    
# Compute average metrics
avg_of_loss = total_of_loss / num_batches
avg_batch_fid = total_fid_sum / num_batches 

print(f"avg_optical_flow_loss: {avg_of_loss}")
print(f"Average Batch FID: {avg_batch_fid}")
print(f"Global FID: {global_fid}")
