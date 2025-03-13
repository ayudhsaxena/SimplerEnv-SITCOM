from functools import partial
import json
from pathlib import Path
from typing import List
from PIL import Image

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader as PytorchDataLoader

from torchvision import transforms as T

import os
import random


num_trajs_oxe = {
    "fractal": {"trainval": 70000, "val": 7000},
    "bridge": {"trainval": 25460, "val": 2546},
    "kuka": {"trainval": 70000, "val": 7000},
}

robot_data_root = "/data/datasets/hf_cache/OpenX/"


def exists(val):
    return val is not None

def identity(t, *args, **kwargs):
    return t

def pair(val):
    return val if isinstance(val, tuple) else (val, val)

'''
This is the dataset class for Sthv2 dataset.
The dataset is a list of folders, each folder contains a sequence of frames.
You have to change the dataset class to fit your dataset for custom training.
'''

class ImageVideoDataset(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        offset=5,
    ):
        super().__init__()
        
        self.folder = folder
        self.folder_list = os.listdir(folder)
        self.image_size = image_size
      
        self.offset = offset

        self.transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize(image_size),
            T.ToTensor(),
        ])


    def __len__(self):
        return len(self.folder_list) ## length of folder list is not exact number of frames; TODO: change this to actual number of frames
    
    def __getitem__(self, index):
        try :
            offset = self.offset
            
            folder = self.folder_list[index]
            img_list = os.listdir(os.path.join(self.folder, folder))

            img_list = sorted(img_list, key=lambda x: int(x.split('.')[0][4:]))
            ## pick random frame 
            first_frame_idx = random.randint(0, len(img_list)-1)
            first_frame_idx = min(first_frame_idx, len(img_list)-1)
            second_frame_idx = min(first_frame_idx + offset, len(img_list)-1)
            
            first_path = os.path.join(self.folder, folder, img_list[first_frame_idx])
            second_path = os.path.join(self.folder, folder, img_list[second_frame_idx])
                    
            img = Image.open(first_path)
            next_img = Image.open(second_path)
            
            transform_img = self.transform(img).unsqueeze(1)
            next_transform_img = self.transform(next_img).unsqueeze(1)
            
            cat_img = torch.cat([transform_img, next_transform_img], dim=1)
            return cat_img
        except :
            print("error", index)
            if index < self.__len__() - 1:
                return self.__getitem__(index + 1)
            else:
                return self.__getitem__(random.randint(0, self.__len__() - 1))


def video_to_tensor(
    path: str,              # Path of the video to be imported
    offset: int = 1,        # Number of frames to skip after first
    transform = T.ToTensor(),       # Transform to be applied to each frame
) -> torch.Tensor:          # shape (1, channels, frames, height, width)

    video = cv2.VideoCapture(path)
    # frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    ## pick random frame 
    # first_frame_idx = random.randint(0, frame_count - 1)
    # first_frame_idx = min(first_frame_idx, frame_count - 1)
    # second_frame_idx = min(first_frame_idx + offset, frame_count - 1)
    # frame_indices = [first_frame_idx, second_frame_idx]

    frames = []
    check = True

    # for fidx in frame_indices:
    #     video.set(cv2.CAP_PROP_POS_FRAMES, fidx)
    #     check, frame = video.read()
    #     if not check:
    #         print(path, frame_indices, frame_count)
    #         break
    #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     frames.append(Image.fromarray(frame))
    while check:
        check, frame = video.read()
        if not check:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame))

    frame_count = len(frames)
    first_frame_idx = random.randint(0, frame_count - 1)
    first_frame_idx = min(first_frame_idx, frame_count - 1)
    second_frame_idx = min(first_frame_idx + offset, frame_count - 1)
    frame_indices = [first_frame_idx, second_frame_idx]
    frames_sampled = [frames[i] for i in frame_indices]
    frames_torch = tuple(map(transform, frames_sampled))
    frames_torch = torch.stack(frames_torch, dim = 1)

    return frames_torch


def video_chunk_to_tensor(
    path: str,              # Path of the video to be imported
    offset: int = 1,        # Number of frames to skip after first
    transform = T.ToTensor(),       # Transform to be applied to each frame
    max_frames = 5,
) -> torch.Tensor:          # shape (1, channels, frames, height, width)

    video = cv2.VideoCapture(path)

    frames = []
    check = True

    while check:
        check, frame = video.read()
        if not check:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame))
    
    frame_count = len(frames)
    max_cont_frames = max_frames * offset
    if frame_count < max_cont_frames:
        start_frame = random.randint(0, frame_count % offset)
        end_frame = frame_count
        frames_sampled = [frames[i] for i in range(start_frame, end_frame, offset)]
    else:
        start_frame = random.randint(0, frame_count - max_cont_frames)
        end_frame = start_frame + max_cont_frames
        frames_sampled = [frames[i] for i in range(start_frame, end_frame, offset)]

    frames_torch = tuple(map(transform, frames_sampled))
    frames_torch = torch.stack(frames_torch, dim = 1)
    
    return frames_torch


def img_seq_to_tensor(path: str, transform = T.ToTensor(), max_frames = 5):
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
    frames_torch = torch.stack(frames_torch, dim = 1)
    return frames_torch


# video dataset
def process_oxe_videos(dataset_name: str, mode: str):
    if dataset_name == 'fractal':
        data_root = Path(robot_data_root) / dataset_name / "processed"
    elif dataset_name == 'bridge':
        data_root = Path(robot_data_root) / dataset_name / "processed"
    elif dataset_name == 'kuka':
        data_root = Path(robot_data_root) / dataset_name / "processed"
    else:
        raise ValueError(f"Dataset {dataset_name} not found")
    num_trajs_to_sample = num_trajs_oxe[dataset_name][mode]
    print("Fetching data from:", data_root)
    video_paths = [str(pth) for pth in data_root.iterdir() if pth.is_dir()]
    local_random = random.Random(42)
    local_random.shuffle(video_paths)
    video_paths = video_paths[:num_trajs_to_sample]
    return video_paths


def process_ssv2_videos(root_dir: Path, mode: str) -> List[Path]:
    if mode == 'train':
        label_paths = [root_dir / 'labels' / 'train.json']
    elif mode == 'val':
        label_paths = [root_dir / 'labels' / 'validation.json']
    elif mode == 'trainval':
        label_paths = [
            root_dir / 'labels' / 'train.json',
            root_dir / 'labels' / 'validation.json'
        ]
    elif mode == 'test':
        label_paths = [root_dir / 'labels' / 'test.json']
    elif mode == 'all':
        label_paths = [
            root_dir / 'labels' / 'train.json',
            root_dir / 'labels' / 'validation.json',
            root_dir / 'labels' / 'test.json'
        ]

    paths: List[Path] = []
    print("Fetching data from:", root_dir)
    for label_path in label_paths:
        with open(label_path, 'r') as f:
            data = json.load(f)
        for ent in data:
            vid_name = ent['id'] + '.webm'
            paths.append(root_dir / '20bn-something-something-v2' / vid_name)
    return paths


class VideoDataset(Dataset):
    def __init__(
        self,
        folder: List[str],
        image_size: int,
        mode: str = 'train',
        offset: int = 5,
    ):
        super().__init__()
        
        self.folder = folder
        self.image_size = (image_size, image_size) if isinstance(image_size, int) else image_size     
        self.offset = offset
        self.video_list = []
        for data_dir in folder:
            if "something-something-v2" in data_dir:
                self.video_list.extend(process_ssv2_videos(Path(f'{data_dir}'), mode))
            if "ego4d" in folder:
                raise NotImplementedError("ego4d dataset not implemented yet")        

        self.transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize(self.image_size),
            T.ToTensor(),
        ])

        # functions to transform video path to tensor
        self.video_to_tensor = partial(video_to_tensor, offset=offset, transform = self.transform)

        print(f"Found {len(self.video_list)} videos in {mode} mode!")


    def __len__(self):
        return len(self.video_list) ## length of folder list is not exact number of frames; TODO: change this to actual number of frames
    
    def __getitem__(self, index):
        try :
            video_path = self.video_list[index]
            video_tensor = self.video_to_tensor(video_path)
            return video_tensor
        except :
            print("error", index)
            if index < self.__len__() - 1:
                return self.__getitem__(index + 1)
            else:
                return self.__getitem__(random.randint(0, self.__len__() - 1))


class VideoDataset2(Dataset):
    def __init__(
        self,
        folder: List[str],
        image_size: int,
        mode: str = 'train',
        offset: int = 8,
        max_frames = 5,
    ):
        super().__init__()
        
        self.folder = folder
        self.image_size = (image_size, image_size) if isinstance(image_size, int) else image_size     
        self.offset = offset
        self.max_frames = max_frames
        self.video_list = []
        for data_dir in folder:
            if "something-something-v2" in data_dir:
                self.video_list.extend(process_ssv2_videos(Path(f'{data_dir}'), mode))
            if "ego4d" in folder:
                raise NotImplementedError("ego4d dataset not implemented yet")        

        self.transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize(self.image_size),
            T.ToTensor(),
        ])

        # functions to transform video path to tensor
        self.video_to_tensor = partial(video_chunk_to_tensor, offset=offset, transform = self.transform, max_frames = max_frames)

        print(f"Found {len(self.video_list)} videos in {mode} mode!")


    def __len__(self):
        return len(self.video_list) ## length of folder list is not exact number of frames; TODO: change this to actual number of frames
    
    def __getitem__(self, index):
        try :
            video_path = self.video_list[index]
            video_tensor = self.video_to_tensor(video_path)
            # create mask
            mask = torch.zeros(self.max_frames, dtype=torch.bool)
            mask[:video_tensor.shape[1]] = True
            video_tensor = torch.nn.functional.pad(
                video_tensor, (0, 0, 0, 0, 0, self.max_frames - video_tensor.shape[1])
            )
            return video_tensor, mask
        except :
            print("error", index, self.video_list[index])
            if index < self.__len__() - 1:
                return self.__getitem__(index + 1)
            else:
                return self.__getitem__(random.randint(0, self.__len__() - 1))


class VideoDatasetCotrain(Dataset):
    def __init__(
        self,
        folder: List[str],
        image_size: int,
        mode: str = 'train',
        offset: int = 8,
        max_frames = 5,
    ):
        super().__init__()
        
        self.folder = folder
        self.image_size = (image_size, image_size) if isinstance(image_size, int) else image_size     
        self.offset = offset
        self.max_frames = max_frames
        self.video_list = []
        for data_dir in folder:
            if "something-something-v2" in data_dir:
                self.video_list.extend(process_ssv2_videos(Path(f'{data_dir}'), mode))
            if "fractal" in data_dir:
                self.video_list.extend(process_oxe_videos("fractal", mode))
            if "bridge" in data_dir:
                self.video_list.extend(process_oxe_videos("bridge", mode))
            if "kuka" in data_dir:
                self.video_list.extend(process_oxe_videos("kuka", mode))
            if "ego4d" in data_dir:
                raise NotImplementedError("ego4d dataset not implemented yet")        

        self.transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize(self.image_size),
            T.ToTensor(),
        ])

        # functions to transform video path to tensor
        self.video_to_tensor = partial(video_chunk_to_tensor, offset=offset, transform = self.transform, max_frames = max_frames)
        self.img_seq_to_tensor = partial(img_seq_to_tensor, transform = self.transform, max_frames = max_frames)

        # shuffle the video list
        local_random = random.Random(42)
        local_random.shuffle(self.video_list)
        print(f"Found {len(self.video_list)} videos in {mode} mode!")


    def __len__(self):
        return len(self.video_list) ## length of folder list is not exact number of frames; TODO: change this to actual number of frames
    
    def __getitem__(self, index):
        try :
            video_path = self.video_list[index]
            if "something-something-v2" in str(video_path):
                video_tensor = self.video_to_tensor(video_path)
            else:
                video_path = os.path.join(video_path, "images")
                video_tensor = self.img_seq_to_tensor(video_path)
            # create mask
            mask = torch.zeros(self.max_frames, dtype=torch.bool)
            mask[:video_tensor.shape[1]] = True
            video_tensor = torch.nn.functional.pad(
                video_tensor, (0, 0, 0, 0, 0, self.max_frames - video_tensor.shape[1])
            )
            return video_tensor, mask
        except :
            print("error", index, self.video_list[index])
            if index < self.__len__() - 1:
                return self.__getitem__(index + 1)
            else:
                return self.__getitem__(random.randint(0, self.__len__() - 1))

if __name__ == "__main__":
    # test video dataset
    dataset = VideoDatasetCotrain(["kuka"], 256, mode='val')
    t1 = dataset[337]
    import ipdb; ipdb.set_trace()
    # for i in range(len(dataset)):
    #     t1 = dataset[i]