import h5py
import numpy as np
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from astra_wrapper import ASTRAOperator

class HDF5SeqDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.file = None
        with h5py.File(path, 'r') as f:
            self.length = len(f)
            

    def __len__(self):
        return self.length
    

    def __getitem__(self, idx):
        if self.file is None:
            self.file = h5py.File(self.path, 'r')
        sample = self.file[f'sample_{idx}']

        return {
            'sinogram_rays': torch.from_numpy(sample['sinogram_rays'][:]).float(),
            'image_GT': torch.from_numpy(sample['image_GT'][:]).float(),
            'baseline': torch.from_numpy(sample['baseline'][:]).float(),
            'img_idx': idx
        }
    
    

class Ellipses_SeqData(Dataset):
    def __init__(
        self, 
        images, 
        device_idx=0, 
        n_detectors=32, 
        n_angles=180, 
        I0=1e4, 
    ):
        self.images = images
        self.length = len(images)
        
        img_size = images.shape[-1]
     
        self.astra_op = ASTRAOperator(
            device_idx=device_idx, 
            img_size=img_size, 
            n_detectors=n_detectors, 
            n_angles=n_angles,
            I0=I0
        )
        
        self.phi_grid = torch.from_numpy(np.repeat(np.arange(n_angles), n_detectors)).float() # angle indices: repeat positions for each angle
        self.s_grid = torch.from_numpy(np.tile(np.arange(n_detectors), n_angles)).float() # detector indices: cycle through positions
        
        ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        topk_dir = os.path.join(ROOT, 'topk')
        topk = torch.load(
            topk_dir + '/astra_topk_' + str(img_size) + '.pt', 
            map_location='cpu', 
            weights_only=True
        )
            
        self.ray_indices = topk['ray_indices'].int()
        self.ray_weights = topk['ray_weights']

        
    def __len__(self):
        return self.length
    
    
    # ----- SORTED RAY SEQUENCES FROM PROJECTION DATA WITH GEOMETRICAL FEATURES ----- 
    def pixel_ray_sequences(self, sinogram):
        # sort ray sequences with respect to angle & offset structure
        sort_key = self.phi_grid[self.ray_indices] * (self.s_grid.max() + 1) + self.s_grid[self.ray_indices]
        sort_idx = torch.argsort(sort_key, dim=1)
        sorted_ray_indices = torch.gather(self.ray_indices, dim=1, index=sort_idx)
        
        values = sinogram.flatten()
        ray_values = values[sorted_ray_indices]  # projection values: (n_pixels, k)
        phi = self.phi_grid[sorted_ray_indices]  # projection angles: (n_pixels, k)
        s = self.s_grid[sorted_ray_indices]      # offsets: (n_pixels, k)
        weights = torch.gather(self.ray_weights, dim=1, index=sort_idx) # ray-pixel intersection weights: (n_pixels, k)

        return torch.stack([ray_values, phi, s, weights], dim=-1)  # shape: (n_pixels, k, 4)


    def __getitem__(self, idx):
        # GT image
        image = self.images[idx]

        # noisy sinogram from groundtruth 
        sinogram = self.astra_op.forward(image, device='cpu', add_noise=True)

        # ray sequences
        sinogram_rays = self.pixel_ray_sequences(sinogram)

        # baseline: FBP
        baseline_img = self.astra_op.filtered_back_projection(sinogram, device='cpu')

        return {
            'sinogram_rays': sinogram_rays.float(),
            'image_GT': torch.from_numpy(image).float(),
            'baseline': baseline_img.float(),
            'img_idx': idx
        }
    
    
    
class LoDoPaB_SeqData(Dataset):
    def __init__(
        self, 
        files, 
        device_idx=0, 
        img_size=32, 
        n_detectors=32, 
        n_angles=180, 
        I0=1e4
    ):
        self.files = files
        self.img_size = img_size
        
        self.cumulative = []
        total_length = 0
        if self.files is not None:
            for f in self.files:
                with h5py.File(f, 'r') as h:
                    length = len(h['data'])
                total_length += length
                self.cumulative.append(total_length)
        self.length = total_length
        
        self.astra_op = ASTRAOperator(
            device_idx=device_idx, 
            img_size=img_size, 
            n_detectors=n_detectors, 
            n_angles=n_angles, 
            I0=I0
        )
        
        self.phi_grid = torch.from_numpy(np.repeat(np.arange(n_angles), n_detectors)).float() # angle indices: repeat positions for each angle
        self.s_grid = torch.from_numpy(np.tile(np.arange(n_detectors), n_angles)).float() # detector indices: cycle through positions
        
        ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        topk_dir = os.path.join(ROOT, 'topk')
        topk = torch.load(
            topk_dir + '/astra_topk_' + str(img_size) + '.pt', 
            map_location='cpu', 
            weights_only=True
        )

        self.ray_indices = topk['ray_indices'].int()
        self.ray_weights = topk['ray_weights']
        
        
    def __len__(self):
        return self.length
    
    
    # ----- SORTED RAY SEQUENCES FROM PROJECTION DATA WITH GEOMETRICAL FEATURES ----- 
    def pixel_ray_sequences(self, sinogram):
        # sort ray sequences with respect to angle & offset structure
        sort_key = self.phi_grid[self.ray_indices] * (self.s_grid.max() + 1) + self.s_grid[self.ray_indices]
        sort_idx = torch.argsort(sort_key, dim=1)
        sorted_ray_indices = torch.gather(self.ray_indices, dim=1, index=sort_idx)
        
        values = sinogram.flatten()
        ray_values = values[sorted_ray_indices]  # projection values: (n_pixels, k)
        phi = self.phi_grid[sorted_ray_indices]  # projection angles: (n_pixels, k)
        s = self.s_grid[sorted_ray_indices]      # offsets: (n_pixels, k)
        weights = torch.gather(self.ray_weights, dim=1, index=sort_idx) # ray-pixel intersection weights: (n_pixels, k)

        return torch.stack([ray_values, phi, s, weights], dim=-1)  # shape: (n_pixels, k, 4)
    
    
    def __getitem__(self, idx):
        # ----- CREATE SEQUENCE DATA WITH LODOPAB GROUNDTRUTH -----
        # index mapping
        f_idx = 0
        while idx >= self.cumulative[f_idx]:
            f_idx += 1   
        if f_idx == 0:
            local_idx = idx
        else:
            local_idx = idx - self.cumulative[f_idx - 1]
            
        # load GT data
        with h5py.File(self.files[f_idx], 'r') as f:
            image_GT = f['data'][local_idx]
        image_GT = torch.tensor(image_GT).float()

        # resize GT data
        if self.img_size != 362: # original LoDoPaB image resolution: 362x362 pixels
            image_GT = F.interpolate(
                image_GT.unsqueeze(0).unsqueeze(0), 
                size=(self.img_size, self.img_size), 
                mode='bilinear'
            ).squeeze(0).squeeze(0)

        # noisy sinogram from groundtruth 
        sinogram = self.astra_op.forward(image_GT, device='cpu', add_noise=True)

        # ray sequences
        sinogram_rays = self.pixel_ray_sequences(sinogram)

        # baseline: FBP 
        baseline = self.astra_op.filtered_back_projection(sinogram, device='cpu')
            
        return {
            'sinogram_rays': sinogram_rays,
            'image_GT': image_GT,
            'baseline': baseline,
            'img_idx': idx
        }