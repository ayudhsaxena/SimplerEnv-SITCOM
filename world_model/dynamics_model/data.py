from functools import partial
import json
import os
from pathlib import Path
import random
from typing import List

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader as PytorchDataLoader
from torch.utils.data import Dataset
from torchvision import transforms as T
from tqdm import tqdm

num_trajs_oxe = {
    "fractal": {"trainval": 70000, "val": 7000},
    "bridge": {"trainval": 25460, "val": 2546},
    "kuka": {"trainval": 70000, "val": 7000},
}

num_trajs_ft = {
    "simpler": {"trainval": 80, "val": 20},
    "bridge": {"trainval": 25460, "val": 2546},
}

robot_data_root = "/data/hf_cache/OpenX/fractal"
simpler_data_root = "/data/user_data/ayudhs/random/multimodal/aggregate_v3"


def exists(val):
    return val is not None


def identity(t, *args, **kwargs):
    return t


def pair(val):
    return val if isinstance(val, tuple) else (val, val)


def img_seq_to_tensor(path: str, transform=T.ToTensor(), max_frames=5):
    img_list = os.listdir(path)
    img_list = sorted(img_list)
    # pick random sequence of max_frames
    if len(img_list) <= max_frames:
        start_frame = 0
        end_frame = start_frame + len(img_list)
    else:
        start_frame = random.randint(0, len(img_list) - max_frames)
        end_frame = start_frame + max_frames
    img_list = img_list[start_frame:end_frame]
    frames = [Image.open(os.path.join(path, img)) for img in img_list]
    frames_torch = tuple(map(transform, frames))
    frames_torch = torch.stack(frames_torch, dim=1)
    return frames_torch


# video dataset
def process_oxe_videos(dataset_name: str, mode: str):
    if dataset_name == "fractal":
        data_root = Path(robot_data_root) / dataset_name / "processed"
    elif dataset_name == "bridge":
        data_root = Path(robot_data_root) / dataset_name / "processed"
    elif dataset_name == "kuka":
        data_root = Path(robot_data_root) / dataset_name / "processed"
    else:
        raise ValueError(f"Dataset {dataset_name} not found")
    num_trajs_to_sample = num_trajs_oxe[dataset_name][mode]
    print("Fetching data from:", data_root, "mode:", mode)
    video_paths = [str(pth) for pth in data_root.iterdir() if pth.is_dir()]
    if mode == "trainval":
        seed = 42
    elif mode == "val":
        seed = 43
    else:
        seed = 44
    local_random = random.Random(seed)
    local_random.shuffle(video_paths)
    video_paths = video_paths[:num_trajs_to_sample]
    data = []
    for vid_path in tqdm(video_paths):
        images_path = os.path.join(vid_path, "images")
        actions_file = os.path.join(vid_path, "actions.npy")
        if os.path.exists(images_path) and os.path.exists(actions_file):
            images = sorted(os.listdir(images_path))
            actions = np.load(actions_file)
            if len(images) != len(actions):
                raise ValueError(f"Mismatch in {vid_path}: {len(images)} images vs {len(actions)} actions.")
            for idx in range(len(images) - 1):
                data.append((os.path.join(images_path, images[idx]), actions[idx], os.path.join(images_path, images[idx + 1])))                
    return data


def process_ssv2_videos(root_dir: Path, mode: str) -> List[Path]:
    if mode == "train":
        label_paths = [root_dir / "labels" / "train.json"]
    elif mode == "val":
        label_paths = [root_dir / "labels" / "validation.json"]
    elif mode == "trainval":
        label_paths = [root_dir / "labels" / "train.json", root_dir / "labels" / "validation.json"]
    elif mode == "test":
        label_paths = [root_dir / "labels" / "test.json"]
    elif mode == "all":
        label_paths = [
            root_dir / "labels" / "train.json",
            root_dir / "labels" / "validation.json",
            root_dir / "labels" / "test.json",
        ]

    paths: List[Path] = []
    print("Fetching data from:", root_dir)
    for label_path in label_paths:
        with open(label_path, "r") as f:
            data = json.load(f)
        for ent in data:
            vid_name = ent["id"] + ".webm"
            paths.append(root_dir / "20bn-something-something-v2" / vid_name)
    return paths


def prcess_simpler_videos(dataset_name: str, mode: str):
    data_root = Path(simpler_data_root)
    env_dict: dict[str, list[str]] = {}
    for env in data_root.iterdir():
        if env.is_dir():
            env_dict[str(env)] = []
    
    for env in env_dict:
        env_dir = data_root / env
        for traj in env_dir.iterdir():
            if not traj.is_dir():
                continue
            env_dict[env].append(str(traj))
    
    local_random = random.Random(42)
    for env in env_dict:
        local_random.shuffle(env_dict[env])
        assert len(env_dict[env]) == sum([x // len(env_dict) for x in num_trajs_ft[dataset_name].values()])
        num_trajs = num_trajs_ft[dataset_name]["trainval"] // len(env_dict)
        if mode == "trainval":
            env_dict[env] = env_dict[env][:num_trajs]
        elif mode == "val":
            env_dict[env] = env_dict[env][num_trajs:]
    data = []
    for env in env_dict:
        env_dir = data_root / env
        for traj in env_dict[env]:
            traj_dir = env_dir / traj
            images_path = str(traj_dir / "images")
            actions_file = str(traj_dir / "actions.npy")
            actions = np.load(actions_file)
            images = sorted(os.listdir(images_path))
            for idx in range(len(images) - 1):
                 data.append((os.path.join(images_path, images[idx]), actions[idx], os.path.join(images_path, images[idx + 1])))
    return data

def process_simpler_videos_autoregressive(dataset_name: str, mode: str, window_size: int = 5):
    data_root = Path(simpler_data_root)
    env_dict: dict[str, list[str]] = {}
    
    for env in data_root.iterdir():
        if env.is_dir():
            env_dict[str(env)] = []
    
    for env in env_dict:
        env_dir = data_root / env
        for traj in env_dir.iterdir():
            if not traj.is_dir():
                continue
            env_dict[env].append(str(traj))
    
    local_random = random.Random(42)
    for env in env_dict:
        local_random.shuffle(env_dict[env])
        assert len(env_dict[env]) == sum([x // len(env_dict) for x in num_trajs_ft[dataset_name].values()])
        num_trajs = num_trajs_ft[dataset_name]["trainval"] // len(env_dict)
        if mode == "trainval":
            env_dict[env] = env_dict[env][:num_trajs]
        elif mode == "val":
            env_dict[env] = env_dict[env][num_trajs:]
    
    data = []
    for env in env_dict:
        env_dir = data_root / env
        for traj in env_dict[env]:
            traj_dir = env_dir / traj
            images_path = str(traj_dir / "images")
            actions_file = str(traj_dir / "actions.npy")
            actions = np.load(actions_file)
            images = sorted(os.listdir(images_path))
            
            # Ensure we have enough frames to create a window of the specified size
            for start_idx in range(len(images) - window_size + 1):
                window_data = []
                
                # Create a tuple for each frame in the window
                for offset in range(window_size - 1):
                    idx = start_idx + offset
                    window_data.append((
                        os.path.join(images_path, images[idx]),
                        actions[idx],
                        os.path.join(images_path, images[idx + 1])
                    ))
                
                # Add the sequence of tuples to our dataset
                data.append(window_data)
    
    return data
            

class DynamicsModelDataset(Dataset):
    def __init__(
        self,
        folder: List[str],
        image_size: int,
        mode: str = "train",
    ):
        super().__init__()

        self.folder = folder
        self.image_size = (image_size, image_size) if isinstance(image_size, int) else image_size
        self.data_list = []
        for data_dir in folder:
            if "something-something-v2" in data_dir:
                # self.video_list.extend(process_ssv2_videos(Path(f"{data_dir}"), mode))
                raise NotImplementedError("Something-something-v2 dataset not implemented yet")
            if "fractal" in data_dir:
                self.data_list.extend(process_oxe_videos("fractal", mode))
            if "bridge" in data_dir:
                self.data_list.extend(process_oxe_videos("bridge", mode))
            if "kuka" in data_dir:
                self.data_list.extend(process_oxe_videos("kuka", mode))
            if "simpler" in data_dir:
                self.data_list.extend(prcess_simpler_videos("simpler", mode))
            if "ego4d" in data_dir:
                raise NotImplementedError("ego4d dataset not implemented yet")

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

        # shuffle the video list
        local_random = random.Random(42)
        local_random.shuffle(self.data_list)
        print(f"Found {len(self.data_list)} samples in {mode} mode!")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        try:
            img_path, action, next_img_path = self.data_list[index]
            img = Image.open(img_path)
            next_img = Image.open(next_img_path)
            img = self.transform(img)
            next_img = self.transform(next_img)
            action = torch.tensor(action, dtype=torch.float32).reshape(-1)
            action = (action - self.action_mean) / self.action_std
            return img, action, next_img
        except:
            print("error", index, self.data_list[index])
            if index < self.__len__() - 1:
                return self.__getitem__(index + 1)
            else:
                return self.__getitem__(random.randint(0, self.__len__() - 1))

class DynamicsModelAutoregressiveDataset(Dataset):
    def __init__(
        self,
        folder: List[str],
        image_size: int,
        mode: str = "train",
        window_size: int = 5,
    ):
        super().__init__()

        self.folder = folder
        self.image_size = (image_size, image_size) if isinstance(image_size, int) else image_size
        self.data_list = []
        for data_dir in folder:
            if "something-something-v2" in data_dir:
                # self.video_list.extend(process_ssv2_videos(Path(f"{data_dir}"), mode))
                raise NotImplementedError("Something-something-v2 dataset not implemented yet")
            if "fractal" in data_dir:
                self.data_list.extend(process_oxe_videos("fractal", mode))
            if "bridge" in data_dir:
                self.data_list.extend(process_oxe_videos("bridge", mode))
            if "kuka" in data_dir:
                self.data_list.extend(process_oxe_videos("kuka", mode))
            if "simpler" in data_dir:
                self.data_list.extend(process_simpler_videos_autoregressive("simpler", mode, window_size=window_size))
            if "ego4d" in data_dir:
                raise NotImplementedError("ego4d dataset not implemented yet")

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

        # shuffle the video list
        local_random = random.Random(42)
        local_random.shuffle(self.data_list)
        print(f"Found {len(self.data_list)} samples in {mode} mode!")

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        try:
            # Now self.data_list[index] is a list of tuples for the window
            window_data = self.data_list[index]
            
            # Process all frames in the window
            window_imgs = []
            window_actions = []
            window_next_imgs = []
            
            for img_path, action, next_img_path in window_data:
                img = Image.open(img_path)
                next_img = Image.open(next_img_path)
                
                img = self.transform(img)
                next_img = self.transform(next_img)
                
                action = torch.tensor(action, dtype=torch.float32).reshape(-1)
                action = (action - self.action_mean) / self.action_std
                
                window_imgs.append(img)
                window_actions.append(action)
                window_next_imgs.append(next_img)
            
            # Stack all images and actions along a new dimension
            window_imgs = torch.stack(window_imgs, dim=0)  # [window_size-1, C, H, W]
            window_actions = torch.stack(window_actions, dim=0)  # [window_size-1, action_dim]
            window_next_imgs = torch.stack(window_next_imgs, dim=0)  # [window_size-1, C, H, W]
            
            return window_imgs, window_actions, window_next_imgs
        
        except Exception as e:
            print(f"Error at index {index}: {e}")
            if index < self.__len__() - 1:
                return self.__getitem__(index + 1)
            else:
                return self.__getitem__(random.randint(0, self.__len__() - 1))


if __name__ == "__main__":
    # test video dataset
    dataset = DynamicsModelDataset(["simpler"], 256, mode="trainval")
    t1 = dataset[4]
    import ipdb

    ipdb.set_trace()
    # for i in range(len(dataset)):
    #     t1 = dataset[i]
