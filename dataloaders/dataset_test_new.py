import torch
import torch.utils.data
import numpy as np
import cv2
from skimage import io
import os
import os.path as osp
import yaml

class OmniDepthDataset(torch.utils.data.Dataset):
    '''PyTorch dataset module for efficient loading'''
    def __init__(self,
                 root_path,
                 yaml_path,

                 path_file_source="/media/doc-lab/HDCL-UT/outdoor_dataset/carla_captures/eqr_rgbd/Town03_05/left",
                 path_file_target="/media/doc-lab/HDCL-UT/outdoor_dataset/carla_captures/eqr_rgbd/Town03_05/right"):
        
        self.root_path = root_path
        self.yaml_path = yaml_path
        self.path_file_source = path_file_source 
        self.path_file_target = path_file_target 
        self.image_list = self.make_list(self.yaml_path)
    def make_list(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"YAML file not found: {path}")
        with open(path, 'r') as f1:
            data = yaml.safe_load(f1)
        clean_list = [x for x in data if x.endswith('_rgb.png')]
        
        print(f"Loaded {len(clean_list)} RGB images from YAML (Filtered out {len(data) - len(clean_list)} other files)")
        return clean_list
    
    def __getitem__(self, idx):
        rgb_name = self.image_list[idx]
        prefix = rgb_name.replace('_rgb.png', '_depth')
        depth_name = prefix + '.npy'
        
        if os.path.isabs(self.path_file_source):
            left_dir = self.path_file_source
            right_dir = self.path_file_target
        else:
            left_dir = osp.join(self.root_path, self.path_file_source)
            right_dir = osp.join(self.root_path, self.path_file_target)
        left_rgb_path = osp.join(left_dir, rgb_name)
        left_depth_path = osp.join(left_dir, depth_name)
        right_rgb_path = osp.join(right_dir, rgb_name)
        right_depth_path = osp.join(right_dir, depth_name)
        left_rgb, left_depth = self.readRGBPano(left_rgb_path, left_depth_path)
        right_rgb, right_depth = self.readRGBPano(right_rgb_path, right_depth_path)
        fish_data = [left_rgb, right_rgb, left_depth, right_depth]
        fish_data[0] = torch.from_numpy(fish_data[0].transpose(2, 0, 1).copy()).float()
        fish_data[1] = torch.from_numpy(fish_data[1].transpose(2, 0, 1).copy()).float()
        # Depth: (H, W) -> (1, H, W)
        fish_data[2] = torch.from_numpy(fish_data[2][None, :, :].copy()).float()
        fish_data[3] = torch.from_numpy(fish_data[3][None, :, :].copy()).float()
        
        return fish_data
    
    def __len__(self):
        return len(self.image_list)
    
    def readRGBPano(self, rgb_path, depth_path):

        if not os.path.exists(rgb_path):
            raise FileNotFoundError(f"RGB missing: {rgb_path}")
        rgb = io.imread(rgb_path).astype(np.float32) / 255.
        
        if not os.path.exists(depth_path):
             if depth_path.endswith('.npy'):
                 fallback_path = depth_path.replace('.npy', '.png')
                 if os.path.exists(fallback_path):
                     depth_path = fallback_path
                 else:
                     raise FileNotFoundError(f"Depth missing: {depth_path}")
        
        if depth_path.endswith('.npy'):

            depth = np.load(depth_path).astype(np.float32)
        else:

            depth_ori = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if depth_ori is None:
                raise ValueError(f"Failed to read depth PNG: {depth_path}")
            depth = depth_ori.astype(np.float32) * 0.001 
        
        max_depth = 80.0
        depth[depth > max_depth] = max_depth
        depth = np.nan_to_num(depth, posinf=max_depth, neginf=0.0)
        
        return rgb, depth
