import argparse
import os
import torch
from weight_matrix import ASTRAWeightMatrix

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
topk_dir = os.path.join(ROOT, 'topk')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_size', type=int, default=32)
    #parser.add_argument('--n_detectors', type=int, default=32)
    parser.add_argument('--n_angles', type=int, default=180)
    parser.add_argument('--k', type=int, default=260, help='rays per pixel (sequence length L)')
    
    return parser.parse_args()


def main():
    args = get_args()
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'Using device: {device}') 
    
    n_detectors = args.img_size

    W = ASTRAWeightMatrix(
        device_idx=torch.cuda.current_device(), 
        img_size=args.img_size, 
        n_detectors=n_detectors, 
        n_angles=args.n_angles
    ).build_matrix()
    W = torch.from_numpy(W).float()

    # for each pixel (row): top k (k=260) weight values and corresponding indices
    ray_weights, ray_indices = torch.topk(W, args.k, dim=1)
    # shape: (n_pixels, k)
    
    del W

    os.makedirs(topk_dir, exist_ok=True)
    torch.save({
        'ray_weights': ray_weights, 
        'ray_indices': ray_indices
    }, topk_dir + '/astra_topk_' + str(args.img_size) + '.pt')

    
if __name__ == '__main__':
    main()  