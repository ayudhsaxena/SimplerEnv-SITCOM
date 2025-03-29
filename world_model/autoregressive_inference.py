import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import os
import json
from typing import List, Dict, Tuple, Optional, Any

import os

from tqdm import tqdm
os.environ["TORCH_HOME"] = "/data/user_data/jasonl6/sandeep"
import torch
from torch.utils.data import DataLoader
from dynamics_model import DynamicsModel
from dynamics_model.data import DynamicsModelDataset
from evaluate_utils import * 
from torchvision import transforms as T
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TrajectoryDataset(Dataset):
    def __init__(self, json_path: str, root_dir: str, future_frames: int = 5, image_size : int = 224):
        """
        Args:
            json_path (str): Path to the JSON file with trajectory data
            root_dir (str): Directory where images are stored
            future_frames (int): Number of consecutive frames to include in each sample
        """
        self.image_size = (image_size, image_size) if isinstance(image_size, int) else image_size
        
        self.root_dir = root_dir
        self.future_frames = future_frames
        
        # Load and organize trajectory data
        with open(json_path) as json_file:
            json_list = json.load(json_file)
        
        # Group by episode
        episode_dict = {}
        for elem in json_list:
            tmp_id = elem["id"].split("/")
            episode_id = tmp_id[0] + tmp_id[1]
            
            if episode_id not in episode_dict:
                episode_dict[episode_id] = [elem]
            else:
                episode_dict[episode_id].append(elem)
        
        # Create a list of valid samples (start indices for each window of future_frames)
        self.samples = []
        for episode_id, rollout in episode_dict.items():
            # Only consider this episode if it has at least future_frames steps
            if len(rollout) >= self.future_frames:
                for i in range(len(rollout) - self.future_frames + 1):
                    self.samples.append((episode_id, rollout[i:i+self.future_frames]))
    
        self.transform = T.Compose(
            [
                T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                T.Resize(self.image_size),
                T.ToTensor(),
            ]
        )
        # Use bridge stats for all datasets
        self.action_mean = torch.tensor([
            0.00023341893393080682,
            0.0001300494186580181,
            -0.0001276240509469062,
            -0.00015565630747005343,
            -0.0004039352061226964,
            0.00023557755048386753,
            0.5764579772949219
        ], dtype=torch.float32).reshape(-1)
        self.action_std = torch.tensor([
            0.009765958413481712,
            0.01368918176740408,
            0.012667348608374596,
            0.02853415347635746,
            0.03063797391951084,
            0.07691441476345062,
            0.4973689615726471
        ], dtype=torch.float32).reshape(-1)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample containing future_frames consecutive frames and future_frames-1 actions
        """
        episode_id, trajectory_slice = self.samples[idx]
        
        # Process the slice of trajectory
        frames = []
        actions = []
        
        for i, step in enumerate(trajectory_slice):
            # Load image and convert to tensor
            image_path = os.path.join(self.root_dir, step["image"])
            img = Image.open(image_path)
            img = self.transform(img)
            frames.append(img)
            
            # Get action except for the last frame (as we need frames to predict actions)
            if i < len(trajectory_slice) - 1:
                action = torch.tensor(step["conversations"][1]["raw_actions"], dtype=torch.float32).reshape(-1)
                action = (action - self.action_mean) / self.action_std
                actions.append(action)
        
        # Stack tensors
        frames_tensor = torch.stack(frames)
        actions_tensor = torch.stack(actions) if actions else torch.empty(0)
        
        return {
            'frames': frames_tensor,  # Shape: [future_frames, C, H, W]
            'actions': actions_tensor,  # Shape: [future_frames-1, action_dim]
            'episode_id': episode_id
        }


def create_trajectory_dataloader(
    json_path: str,
    root_dir: str,
    future_frames: int = 5,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4
) -> DataLoader:
    """
    Create a DataLoader for trajectory data
    
    Args:
        json_path (str): Path to the JSON file with trajectory data
        root_dir (str): Directory where images are stored
        future_frames (int): Number of consecutive frames to include in each sample
        batch_size (int): Batch size for the dataloader
        shuffle (bool): Whether to shuffle the data
        num_workers (int): Number of worker processes for data loading
        
    Returns:
        DataLoader: PyTorch DataLoader for trajectory data
    """
    dataset = TrajectoryDataset(
        json_path=json_path,
        root_dir=root_dir,
        future_frames=future_frames
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader



laq = DynamicsModel(
    dim=768,
    image_size=224,
    patch_size=14,
    spatial_depth=8,  # 8
    dim_head=64,
    heads=16,
    use_lpips_loss=True,
)

NUM_FUTURE_STEPS = 5
val_dl = create_trajectory_dataloader(json_path= "/home/scratch/hshah2/aggregate_v3/simpler_v3.jsonl", root_dir= "/home/scratch/hshah2/aggregate_v3", future_frames=NUM_FUTURE_STEPS)

breakpoint()

pretrain_ckpt = "/home/jasonl6/sandeep/SimplerEnv-SITCOM/world_model/results/dyna1_simpl_ft_1/vae.pt"
ckpt = torch.load(pretrain_ckpt, map_location="cpu")["model"]
msg = laq.load_state_dict(ckpt)
print(msg)
laq = laq.to(device)
laq.eval()

batch_evaluator = [BatchEvaluator(OpticalFlowEvaluator(device=device, image_size=(3, 224, 224)), device=device) for _ in range(NUM_FUTURE_STEPS-1)]
total_of_loss = [0.0 for _ in range(NUM_FUTURE_STEPS-1)]
total_fid_sum = [0.0 for _ in range(NUM_FUTURE_STEPS-1)]
num_batches = 0

for idx, val_sample in tqdm(enumerate(val_dl)):
    images = val_sample["frames"].to(device) # B x K x C x H x W
    actions = val_sample["actions"].to(device) # B x (K-1) x A
    future_iters = actions.shape[1]
    curr_img_batch = images[:,0]
    for i in range(1, future_iters):
        gt_img_batch = images[:, i]
        action_batch = actions[:, i-1]
        recons_img_batch = laq(curr_img_batch, action_batch, gt_img_batch, return_recons_only=True)
        batch_results = batch_evaluator[i-1].evaluate_batch(images[:, 0], recons_img_batch, gt_img_batch)
    
        # Accumulate metrics
        total_of_loss[i-1] += batch_results["optical_flow_loss"]
        total_fid_sum[i-1] += batch_results["fid_score"]
    
    num_batches += 1
    
# Compute average metrics
avg_of_loss = [elem / num_batches for elem in total_of_loss]
avg_batch_fid = [elem / num_batches for elem in total_fid_sum] 
global_fid = [elem.evaluator.fid_calculator.compute() for elem in batch_evaluator]
print(f'Global FID: {batch_results["fid_score"]}')


import matplotlib.pyplot as plt
import seaborn as sns
# Set Seaborn style for aesthetics
sns.set_style("whitegrid")

# Plot avg_of_loss and save
plt.figure(figsize=(8, 5))
plt.plot(avg_of_loss, marker='o', linestyle='-', color='royalblue', label='Avg Optical Loss')
plt.xlabel("Epochs")
plt.ylabel("Optical Loss")
plt.title("Average Optical Loss Over Epochs")
plt.legend()
plt.savefig("avg_of_loss_plot.png", dpi=300)
plt.close()

# Plot avg_batch_fid and save
plt.figure(figsize=(8, 5))
plt.plot(avg_batch_fid, marker='s', linestyle='--', color='darkorange', label='Avg Batch FID')
plt.xlabel("Epochs")
plt.ylabel("Batch FID")
plt.title("Average Batch FID Over Epochs")
plt.legend()
plt.savefig("avg_batch_fid_plot.png", dpi=300)
plt.close()

# Plot global_fid and save
plt.figure(figsize=(8, 5))
plt.plot(global_fid, marker='d', linestyle='-', color='green', label='Global FID')
plt.xlabel("Epochs")
plt.ylabel("Global FID")
plt.title("Global FID Over Epochs")
plt.legend()
plt.savefig("global_fid_plot.png", dpi=300)
plt.close()
