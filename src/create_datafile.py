import argparse
import h5py
import numpy as np
import os
import torch
import wget
import zipfile
from glob import glob
from tqdm import tqdm
from ellipsesGen.EllipsesDataset import EllipsesDataset
from sequence_dataset import Ellipses_SeqData, LoDoPaB_SeqData

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_dir = os.path.join(ROOT, 'datasets')
lodopab_dir = os.path.join(ROOT, 'lodopab_validation')
ellipses_dir = os.path.join(ROOT, 'ellipses_dataset')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_size', type=int, default=32)
    #parser.add_argument('--n_detectors', type=int, default=32)
    parser.add_argument('--n_angles', type=int, default=180)
    parser.add_argument('--I0', type=float, default=1e4, help='noise level')
    parser.add_argument('--dataset_gt', type=str, default='ellipses', help='ellipses or lodopab')
    
    return parser.parse_args()


def create_ellipses_data(
    dataset_dir, 
    ellipses_dir, 
    img_size=32,
    n_detectors=32,
    n_angles=180,
    I0=1e4
):
    # ----- GENERATE SYNTHETIC GROUND-TRUTH IMAGES -----
    n_images = 10000
    if not os.path.exists(ellipses_dir + '_' + str(img_size) + 'px' + str(n_images) + '.hdf5'):
        print('Generating ellipses...')
        images_GT = EllipsesDataset(
            image_count=n_images,
            image_size=img_size,
            ellipses_per_image=5,
            binary_output=False,
            min_excentricity=0.975,
            max_excentricity=1.0,
            ellipse_scales=(0.01, 0.06),
            ellipse_intensities=(0.1, 1.0),
            normalize_intensities=True,
            seed=42,
            smooth=False
        )
        images_GT.save_to_file(ellipses_dir + '_' + str(img_size) + 'px' + str(n_images) + '.hdf5')
    else:
        print('Loading ellipses from file...')
        images_GT = EllipsesDataset.from_file(ellipses_dir + '_' + str(img_size) + 'px' + str(n_images) + '.hdf5')
    images_GT = np.squeeze(torch.stack([images_GT[i]['input'] for i in range(len(images_GT))], dim=0).numpy(), axis=1)
    #print(images_GT.shape, images_GT.dtype)
    
    # ----- CREATE DATASET WITH RAY SEQUENCES -----
    dataset = Ellipses_SeqData(
        images_GT, 
        device_idx=torch.cuda.current_device(), 
        n_detectors=n_detectors, 
        n_angles=n_angles,
        I0=I0,
    )
    
    with h5py.File(dataset_dir + '/ellipsesSeqData_' + str(img_size) + 'px' + str(int(I0)) + '.hdf5', 'w') as f:
        # ----- METADATA -----
        f.attrs['img_size'] = img_size
        f.attrs['n_detectors'] = n_detectors
        f.attrs['n_angles'] = n_angles
        f.attrs['I0'] = I0
        f.attrs['description'] = 'Ellipses CT dataset with ray sequences and FBP baseline'
        
        # ----- DATA -----
        for i in tqdm(range(len(dataset)), desc='Create dataset file'):
            sample = dataset[i]
            grp = f.create_group(f'sample_{i}')
            grp.create_dataset('sinogram_rays', data=sample['sinogram_rays'].cpu().numpy(), compression='gzip')
            grp.create_dataset('image_GT', data=sample['image_GT'].cpu().numpy(), compression='gzip')
            grp.create_dataset('baseline', data=sample['baseline'].cpu().numpy(), compression='gzip')
    

def create_lodopab_data(
    dataset_dir, 
    lodopab_dir, 
    img_size=32, 
    n_detectors=32, 
    n_angles=180, 
    I0=1e4
):
    # ----- DOWNLOAD LODOPAB-CT VALIDATION SET -----
    os.makedirs(lodopab_dir, exist_ok=True)
    zip_path = lodopab_dir + '/ground_truth_validation.zip'
    if not os.path.exists(zip_path):
        print('Download groundtruth validation zip...')
        wget.download('https://zenodo.org/records/3384092/files/ground_truth_validation.zip', zip_path)
    
    files = sorted(glob(lodopab_dir + '/ground_truth_validation_*.hdf5'))
    if len(files) > 0:
        print(f'LoDoPaB validation folder: {len(files)} groundtruth files')
    else: 
        print('Extract groundtruth files...')
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(lodopab_dir)
        files = sorted(glob(lodopab_dir + '/ground_truth_validation_*.hdf5'))
        print(f'Finished Extraction -> LoDoPaB validation folder: {len(files)} groundtruth files')
    
    # ----- CREATE DATASET WITH RAY SEQUENCES FROM LODOPAB GROUND-TRUTH -----
    dataset = LoDoPaB_SeqData(
        files, 
        device_idx=torch.cuda.current_device(), 
        img_size=img_size, 
        n_detectors=n_detectors,
        n_angles=n_angles,
        I0=I0
    )
    
    with h5py.File(dataset_dir + '/lodopabSeqData_' + str(img_size) + 'px' + str(int(I0)) + '.hdf5', 'w') as f:
        # ----- METADATA -----
        f.attrs['img_size'] = img_size
        f.attrs['n_detectors'] = n_detectors
        f.attrs['n_angles'] = n_angles
        f.attrs['I0'] = I0
        f.attrs['description'] = 'LoDoPaB CT dataset with ray sequences and FBP baseline'
        
        # ----- DATA -----
        for i in tqdm(range(len(dataset)), desc='Create dataset file'):
            sample = dataset[i]
            grp = f.create_group(f'sample_{i}')
            grp.create_dataset('sinogram_rays', data=sample['sinogram_rays'].cpu().numpy(), compression='gzip')
            grp.create_dataset('image_GT', data=sample['image_GT'].cpu().numpy(), compression='gzip')
            grp.create_dataset('baseline', data=sample['baseline'].cpu().numpy(), compression='gzip')
    
    
def main():
    args = get_args()
    n_detectors = args.img_size
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'Using device: {device}')
    
    os.makedirs(dataset_dir, exist_ok=True)
    
    if args.dataset_gt=='ellipses':
        create_ellipses_data(
            dataset_dir, 
            ellipses_dir, 
            img_size=args.img_size,
            n_detectors=n_detectors,
            n_angles=args.n_angles,
            I0=args.I0
        )
    else:
        create_lodopab_data(
            dataset_dir,
            lodopab_dir,
            img_size=args.img_size,
            n_detectors=n_detectors,
            n_angles=args.n_angles,
            I0=args.I0
        )
    
    
if __name__ == '__main__':
    main()