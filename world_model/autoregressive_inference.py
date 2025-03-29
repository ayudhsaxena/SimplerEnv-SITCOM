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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TrajectoryDataset(Dataset):
    def __init__(self, json_path: str, root_dir: str, future_frames: int = 5):
        """
        Args:
            json_path (str): Path to the JSON file with trajectory data
            root_dir (str): Directory where images are stored
            future_frames (int): Number of consecutive frames to include in each sample
        """
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
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            # Convert to RGB and normalize to [0, 1]
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
            image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # [C, H, W]
            frames.append(image_tensor)
            
            # Get action except for the last frame (as we need frames to predict actions)
            if i < len(trajectory_slice) - 1:
                action = np.array(step["conversations"][1]["raw_actions"], dtype=np.float32)
                actions.append(torch.tensor(action, dtype=torch.float32))
        
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

# Example usage:
if __name__ == "__main__":
    # Create dataloader with 5 future frames
    dataloader = create_trajectory_dataloader(
        json_path="simpler_v3.jsonl",
        root_dir="ROOT_DIR",
        future_frames=5,
        batch_size=16
    )
    
    # Example iteration
    for batch in dataloader:
        frames = batch['frames']  # [B, future_frames, C, H, W]
        actions = batch['actions']  # [B, future_frames-1, action_dim]
        
        print(f"Frames shape: {frames.shape}")
        print(f"Actions shape: {actions.shape}")
        
        # Here you would use these to train an autoregressive model
        break