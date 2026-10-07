import astra
import numpy as np
import torch

class ASTRAOperator:
    def __init__(
        self, 
        device_idx=0, 
        img_size=32, 
        n_detectors=32, 
        n_angles=180, 
        I0=1e4
    ):
        self.device_idx = device_idx
        det_spacing = np.sqrt(img_size**2 + img_size**2) / n_detectors # detector plane -> image diagonal
        angles = np.linspace(0, np.pi, n_angles, endpoint=False)

        # ASTRA volume geometry
        self.vol_geom = astra.create_vol_geom(img_size, img_size)

        # ASTRA projection geometry
        self.proj_geom = astra.create_proj_geom(
            'parallel', 
            det_spacing, 
            n_detectors, 
            angles
        )
        
        astra.set_gpu_index(device_idx)

        # create projector
        self.projector_id = astra.create_projector(
            'cuda', 
            self.proj_geom, 
            self.vol_geom
        )
        
        self.I0 = I0

    
    def cleanup(self):
        if hasattr(self, 'projector_id'):
            astra.projector.delete(self.projector_id)
            self.projector_id = None


    # ----- CT FORWARD: image -> sinogram -----        
    def forward(self, x, device, add_noise=False):
        if torch.is_tensor(x):
            x = x.detach().cpu().numpy()
        x = np.asarray(x, dtype=np.float32)


        vol_id = astra.data2d.create('-vol', self.vol_geom, x) # create volume object
        sino_id = astra.create_sino(vol_id, self.projector_id) # compute forward projection
        if isinstance(sino_id, (tuple, list)):
            sino_id = sino_id[0]
        sino_id = int(sino_id)

        sino = astra.data2d.get(sino_id)

        if add_noise and np.max(sino) > 0: # astra add_noise_to_sino fails on empty sinograms
            sino = astra.functions.add_noise_to_sino(sino, self.I0, seed=0) # Poisson noise model
            
        astra.data2d.delete(vol_id)
        astra.data2d.delete(sino_id)

        return torch.from_numpy(sino).float().to(device)
    
    
    # ----- FILTERED BACK-PROJECTION: sinogram -> reconstruction -----
    def filtered_back_projection(self, sino, device):
        if torch.is_tensor(sino):
            sino = sino.detach().cpu().numpy()

        sino = np.asarray(sino, dtype=np.float32)

        sino_id = astra.data2d.create('-sino', self.proj_geom, sino) # create sinogram (projection data) object
        rec_id = astra.data2d.create('-vol', self.vol_geom) # compute reconstruction

        # reconstruction algorithm configurations
        cfg = astra.astra_dict('FBP_CUDA')
        cfg['option'] = {'GPUIndex': self.device_idx}
        cfg['ReconstructionDataId'] = rec_id
        cfg['ProjectionDataId'] = sino_id

        alg_id = astra.algorithm.create(cfg) # create FBP algorithm
        astra.algorithm.run(alg_id)

        rec = astra.data2d.get(rec_id)

        astra.algorithm.delete(alg_id)
        astra.data2d.delete(sino_id)
        astra.data2d.delete(rec_id)

        return torch.from_numpy(rec).float().to(device)