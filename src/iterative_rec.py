import numpy as np
import os
import torch
import torch.nn as nn
from astra_wrapper import ASTRAOperator
from rec_model import MambaReconstructor

class IterativeReconstructor(nn.Module):
    def __init__(
        self, 
        device_idx=0, 
        img_size=32, 
        n_detectors=32, 
        n_angles=180,
        n_iterations=1, 
        alpha=1.0, 
        chunk_size=1024, 
        d_model=32, 
        headdim=8,
        aggregation='mean_pooling'
    ):
        super().__init__()
        
        self.n_iterations = n_iterations
        self.alpha = alpha

        self.astra_op = ASTRAOperator(
            device_idx=device_idx, 
            img_size=img_size, 
            n_detectors=n_detectors, 
            n_angles=n_angles
        )
        
        self.mamba = MambaReconstructor(
            img_size=img_size, 
            chunk_size=chunk_size, 
            d_model=d_model, 
            headdim=headdim, 
            aggregation=aggregation
        )

        phi_grid = np.repeat(np.arange(n_angles), n_detectors) # angle indices: repeat positions for each angle
        s_grid = np.tile(np.arange(n_detectors), n_angles) # detector indices: cycle through positions
        
        ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        topk_dir = os.path.join(ROOT, 'topk')
        topk = torch.load(
            topk_dir + '/astra_topk_' + str(img_size) + '.pt', 
            map_location='cpu', 
            weights_only=True
        )

        self.register_buffer('phi_grid', torch.from_numpy(phi_grid).float())
        self.register_buffer('s_grid', torch.from_numpy(s_grid).float())
        self.register_buffer('ray_indices', topk['ray_indices'].int()) 
        self.register_buffer('ray_weights', topk['ray_weights']) 


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

    # ----- LEARNED ITERATIVE RECONSTRUCTION -----
    def forward(self, x0, y):
        x = x0     
        ATy = self.mamba(y) # reconstruction from projection measurements (sinogram y)

        for _ in range(self.n_iterations):
            y_pred = []
            for i in range(x.shape[0]):
                x_i = x[i]
                y_i = self.astra_op.forward(x_i, device=x.device) # sinogram from current estimate
                y_pred.append(self.pixel_ray_sequences(y_i))
            y_pred = torch.stack(y_pred)

            correction = self.mamba(y_pred) - ATy            
            x = x - self.alpha * correction # update the reconstruction estimate
            #x = x - 1.0 / float(self.n_iterations) * correction

        return x