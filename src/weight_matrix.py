import astra
import numpy as np
from astra_wrapper import ASTRAOperator

class ASTRAWeightMatrix:
    def __init__(self, device_idx=0, img_size=32, n_detectors=32, n_angles=180):
        self.astra_op = ASTRAOperator(
            device_idx=device_idx, 
            img_size=img_size, 
            n_detectors=n_detectors, 
            n_angles=n_angles
        )
        
        self.img_size = img_size
        

    def build_matrix(self):
        n_pixels = self.img_size * self.img_size

        W = np.zeros((
            n_pixels,
            self.astra_op.proj_geom['DetectorCount'] * len(self.astra_op.proj_geom['ProjectionAngles']) # sinogram size
        ))

        for pixel_idx in range(n_pixels):
            img = np.zeros((self.img_size, self.img_size), dtype=np.float32)
            i = pixel_idx // self.img_size # complete rows
            j = pixel_idx % self.img_size # position within row
            img[i, j] = 1.0 # basis image
            sinogram = self.astra_op.forward(img, device='cpu') # forward operator applied on basis image
            W[pixel_idx] = sinogram.cpu().flatten() # row: intersection weights from all rays

        return W