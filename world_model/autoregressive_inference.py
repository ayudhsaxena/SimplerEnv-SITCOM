import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import json
from typing import List, Dict, Tuple, Optional, Any

from tqdm import tqdm
import torch
from dynamics_model import DynamicsModel
from dynamics_model.data import DynamicsModelAutoregressiveDataset
from evaluate_utils import * 
from torchvision import transforms as T
from PIL import Image
import torchvision.utils as vutils 
import torchvision

import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_autoregressive_dataloader(
    folders: List[str],
    image_size: int = 224,
    future_frames: int = 5,
    mode: str = "val",
    batch_size: int = 4,
    shuffle: bool = False,
    num_workers: int = 4
) -> DataLoader:
    """
    Create a DataLoader for autoregressive trajectory data
    
    Args:
        folders (List[str]): List of folders containing the datasets
        image_size (int): Size of the images
        future_frames (int): Number of consecutive frames to include in each sample
        mode (str): Mode to use for dataset loading ('train', 'val', 'trainval')
        batch_size (int): Batch size for the dataloader
        shuffle (bool): Whether to shuffle the data
        num_workers (int): Number of worker processes for data loading
        
    Returns:
        DataLoader: PyTorch DataLoader for trajectory data
    """
    dataset = DynamicsModelAutoregressiveDataset(
        folder=folders,
        image_size=image_size,
        mode=mode,
        window_size=future_frames,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader


# Load model
laq = DynamicsModel(
    dim=768,
    image_size=224,
    patch_size=14,
    spatial_depth=8,
    dim_head=64,
    heads=16,
    use_lpips_loss=True,
)

NUM_FUTURE_STEPS = 10
# Create dataloader with the DynamicsModelAutoregressiveDataset
val_dl = create_autoregressive_dataloader(
    folders=["simpler"],  # Use folder name that corresponds to simpler dataset
    image_size=224,
    future_frames=NUM_FUTURE_STEPS,
    mode="val",
    shuffle=False
)

# Load model checkpoint
pretrain_ckpt = "/data/user_data/ayudhs/random/multimodal/SimplerEnv-SITCOM/results/dyna1_simpl_ft_1/vae.pt"
ckpt = torch.load(pretrain_ckpt, map_location="cpu")["model"]
msg = laq.load_state_dict(ckpt)
print(msg)
laq = laq.to(device)
laq.eval()

# Initialize evaluators
batch_evaluator = [BatchEvaluator(OpticalFlowEvaluator(device=device, image_size=(3, 224, 224)), device=device) for _ in range(NUM_FUTURE_STEPS-1)]
total_of_loss = [0.0 for _ in range(NUM_FUTURE_STEPS-1)]
total_fid_sum = [0.0 for _ in range(NUM_FUTURE_STEPS-1)]
num_batches = 0

save_path = "/data/user_data/ayudhs/random/multimodal/SimplerEnv-SITCOM/results/autoregressive_results_val"
os.makedirs(save_path, exist_ok=True)

for idx, val_sample in tqdm(enumerate(val_dl)):
    # Unpack data from the autoregressive dataset format
    window_imgs, window_actions, window_next_imgs = val_sample
    
    # Move to device
    window_imgs = window_imgs.to(device)       # [B, window_size-1, C, H, W]
    window_actions = window_actions.to(device) # [B, window_size-1, action_dim]
    window_next_imgs = window_next_imgs.to(device) # [B, window_size-1, C, H, W]
    
    # Get the number of future iterations from the data
    future_iters = window_actions.shape[1]
    
    # Start with the first frame
    curr_img_batch = window_imgs[:, 0]
    recons_img_list = []
    
    # Perform autoregressive prediction
    for i in range(future_iters):
        # Get ground truth and action for this step
        gt_img_batch = window_next_imgs[:, i]
        action_batch = window_actions[:, i]
        
        # Generate predicted next frame
        recons_img_batch = laq(curr_img_batch, action_batch, gt_img_batch, return_recons_only=True)
        
        # Evaluate the prediction
        # For optical flow evaluation, we need the initial frame, the predicted frame, and the ground truth
        batch_results = batch_evaluator[i].evaluate_batch(window_imgs[:, 0], recons_img_batch, gt_img_batch)
        
        # Save the first sample's reconstruction for visualization
        recons_img_list.append(recons_img_batch[0])
        
        # Accumulate metrics
        total_of_loss[i] += batch_results["optical_flow_loss"]
        total_fid_sum[i] += batch_results["fid_score"]
        
        # Use the predicted frame as input for next step
        curr_img_batch = recons_img_batch
    
    # Create a grid of images for visualization
    # Concatenate ground truth images with reconstructed images
    grid = torchvision.utils.make_grid(
        torch.cat((window_next_imgs[0], torch.stack(recons_img_list, dim=0)), dim=0),
        nrow=window_next_imgs.shape[1],
        padding=2,
    )
    
    # Save the grid
    vutils.save_image(grid, os.path.join(save_path, f"comparison{idx}.png"))
    
    num_batches += 1
    
    # Plot the evaluation metrics
    if idx % 5 == 0:  # Plot every 5 batches to reduce computation
        # Set Seaborn style for aesthetics
        sns.set_style("whitegrid")
        
        # Compute average metrics
        avg_of_loss = [elem / num_batches for elem in total_of_loss]
        avg_batch_fid = [elem / num_batches for elem in total_fid_sum] 
        global_fid = [elem.evaluator.fid_calculator.compute().cpu() for elem in batch_evaluator]
        
        # Plot avg_of_loss and save
        plt.figure(figsize=(8, 5))
        plt.plot(avg_of_loss, marker='o', linestyle='-', color='royalblue', label='Avg Optical Loss')
        plt.xlabel("Future step")
        plt.ylabel("Optical Loss")
        plt.title("Average Optical Loss Over Epochs")
        plt.legend()
        plt.savefig(os.path.join(save_path, "avg_of_loss_plot.png"), dpi=300)
        plt.close()

        # Plot avg_batch_fid and save
        plt.figure(figsize=(8, 5))
        plt.plot(avg_batch_fid, marker='s', linestyle='--', color='darkorange', label='Avg Batch FID')
        plt.xlabel("Future step")
        plt.ylabel("Batch FID")
        plt.title("Average Batch FID Over Epochs")
        plt.legend()
        plt.savefig(os.path.join(save_path, "avg_batch_fid_plot.png"), dpi=300)
        plt.close()

        # Plot global_fid and save
        plt.figure(figsize=(8, 5))
        plt.plot(global_fid, marker='d', linestyle='-', color='green', label='Global FID')
        plt.xlabel("Future step")
        plt.ylabel("Global FID")
        plt.title("Global FID Over Epochs")
        plt.legend()
        plt.savefig(os.path.join(save_path, "global_fid_plot.png"), dpi=300)
        plt.close()

