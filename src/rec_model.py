import torch
import torch.nn as nn
from mamba_ssm import Mamba2
from torch.utils.checkpoint import checkpoint

class MambaReconstructor(nn.Module):
    def __init__(self, img_size=32, chunk_size=1024, d_model=32, headdim=8, aggregation='mean_pooling'):
        super().__init__()
        self.img_size = img_size
        self.chunk_size = chunk_size
        self.aggregation = aggregation
        
        self.input_proj = nn.Linear(4, d_model)

        self.mamba = Mamba2(
            d_model=d_model, # internal feature representation
            d_state=16, # internal state representation
            d_conv=4, # convolutional features
            expand=2, # expansion factor
            headdim=headdim # head dimension
        )

        self.norm = nn.LayerNorm(d_model) 
        
        # sequence representation -> pixel value
        self.output_MLP = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )
        
        
    def block(self, x):
        x = self.input_proj(x)
        x = self.norm(x)
        x = self.mamba(x)
        
        return x
            

    # ----- IMAGE RECONSTRUCTION FROM RAY SEQUENCES -----        
    def forward(self, rays):
        B, n_pixels, seq_length, n_features = rays.shape # input: ray sequence
        full_pred = torch.zeros(B, n_pixels, device=rays.device, dtype=rays.dtype)
        
        # pixelwise reconstruction in pixel chunks
        for start in range(0, n_pixels, self.chunk_size):
            end = min(n_pixels, start + self.chunk_size)
            chunk_pixels = end - start
            rays_chunk = rays[:, start:end].reshape(B * chunk_pixels, seq_length, n_features)
            
            if self.aggregation == 'mean_pooling':
                h = checkpoint(self.block, rays_chunk, use_reentrant=False).mean(dim=1) # mean pooling
            else:
                h = checkpoint(self.block, rays_chunk, use_reentrant=False)[:, -1, :] # final token representation (last hidden)
            # final sequence representation: (B * chunk_pixels, seq_length, d_model) -> (B * chunk_pixels, d_model)
            
            chunk_pred = self.output_MLP(h).squeeze(-1).view(B, chunk_pixels)
            full_pred[:, start:end] = chunk_pred

        return full_pred.view(B, self.img_size, self.img_size)