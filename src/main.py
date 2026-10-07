import argparse
import numpy as np
import os
import random
import torch
import torch.distributed as dist
import torch.nn as nn
from pytorch_msssim import SSIM
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import random_split, DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from iterative_rec import IterativeReconstructor
from sequence_dataset import HDF5SeqDataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
checkpoint_dir = os.path.join(ROOT, 'checkpoints')
dataset_dir = os.path.join(ROOT, 'datasets')

def get_args():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--img_size', type=int, default=32)
    #parser.add_argument('--n_detectors', type=int, default=32)
    parser.add_argument('--n_angles', type=int, default=180)
    parser.add_argument('--chunk_size', type=int, default=1024)
    parser.add_argument('--n_epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--train_test_split', type=float, default=0.8)
    parser.add_argument('--n_iterations', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=1.0, help='alpha value for learned correction per iteration')
    parser.add_argument('--d_model', type=int, default=32, help='model dimension -> d_inner = d_model * expand')
    parser.add_argument('--headdim', type=int, default=8, help='head dimension -> nheads = d_inner // headdim')
    parser.add_argument('--aggregation', default='mean_pooling', help='aggregation strategy after Mamba block: mean_pooling or last_hidden')
    parser.add_argument('--resume_training', default=False, help='resume training from checkpoint')
    parser.add_argument('--model_file', type=str, default='checkpoint.pt', help='path to checkpoint (*.pt)')
    parser.add_argument('--dataset_file', type=str, default='ellipsesSeqData_32px10000.hdf5', help='path to dataset file (*.hdf5)')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    
    return parser.parse_args()

def load_dataset(
    dataset_file,  
    device_idx=0, 
    train_test_split=0.8, 
    seed=0,
):
    if device_idx == 0:
        print('Loading dataset from file...')
    
    dataset = HDF5SeqDataset(path=os.path.join(dataset_dir, dataset_file))  
    
    sample = dataset[0]
    if device_idx == 0:
        print(f'Ray sequences shape: {sample["sinogram_rays"].shape}')
        print(f'Ground Truth shape: {sample["image_GT"].shape}')
        print(f'Baseline Reconstruction shape: {sample["baseline"].shape}')
        print(f'Sample Size: {len(dataset)}')
        print(f'Pixel value range of Ground Truth: min={sample["image_GT"].min():.8f}, max={sample["image_GT"].max():.8f}')
    img_size = sample["image_GT"].shape[-1]

    train_size = int(train_test_split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(
        dataset, 
        [train_size, test_size], 
        generator=torch.Generator().manual_seed(seed)
    )

    return train_dataset, test_dataset, img_size


def eval(model, dataloader, device, loss_fn, ssim_module, epoch):
    model.eval()

    baseline_total_loss = 0.0
    baseline_total_ssim = 0.0
    final_total_loss = 0.0
    final_total_ssim = 0.0
    
    gt_images = []
    baseline_images = []
    prediction_images = []
    
    loader = (tqdm(dataloader, desc='Evaluation') if device.index == 0 else dataloader)

    with torch.no_grad():
        for batch in loader:
            y = batch['sinogram_rays'].to(device)
            x_gt = batch['image_GT'].to(device)
            x = batch['baseline'].to(device) # baseline: filtered backprojection

            with torch.autocast(device_type=device.type, dtype=torch.float16):
                x_pred = model(x, y)
            
            gt_images.append(x_gt)
            baseline_images.append(x)
            prediction_images.append(x_pred)

            baseline_total_loss += loss_fn(x, x_gt).item()
            baseline_total_ssim += ssim_module(x.unsqueeze(1), x_gt.unsqueeze(1)).item()
            final_total_loss += loss_fn(x_pred, x_gt).item()
            final_total_ssim += ssim_module(x_pred.unsqueeze(1), x_gt.unsqueeze(1)).item()
            
    baseline_loss = torch.tensor(baseline_total_loss/len(dataloader), device=device)
    baseline_ssim = torch.tensor(baseline_total_ssim/len(dataloader), device=device)
    final_loss = torch.tensor(final_total_loss/len(dataloader), device=device)
    final_ssim = torch.tensor(final_total_ssim/len(dataloader), device=device)

    return gt_images, baseline_images, prediction_images, baseline_loss, final_loss, baseline_ssim, final_ssim


def training(model, dataloader, device, optimizer, loss_fn, scaler, epoch, n_epochs):   
    model.train()
    
    total_loss = 0.0
        
    if device.index == 0:
        print(f'\033[92mEpoch {epoch+1}/{n_epochs}\033[0m')
        loader = tqdm(dataloader, desc='Training') 
    else:
        loader = dataloader

    for batch_idx, batch in enumerate(loader):
        y = batch['sinogram_rays'].to(device)
        x_gt = batch['image_GT'].to(device)
        x = batch['baseline'].to(device) # baseline: filtered backprojection

        optimizer.zero_grad()

        with torch.autocast(device_type='cuda', dtype=torch.float16):
            x_pred = model(x, y)
            loss = loss_fn(x_pred, x_gt)

        #loss.backward()
        scaler.scale(loss).backward()

        #optimizer.step()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    train_loss = torch.tensor(total_loss/len(dataloader), device=device)

    return train_loss


def main():
    args = get_args()
    
    torch.manual_seed(args.seed) # ensure reproducable initialization
    #torch.use_deterministic_algorithms(True, warn_only=True)
    
    # ----- DISTRIBUTED TRAINING SETUP -----
    dist.init_process_group(backend='nccl') # initialize process group
    local_rank = int(os.environ['LOCAL_RANK']) # local rank of the process (GPU index)
    torch.cuda.set_device(local_rank) 
    device = torch.device(f'cuda:{local_rank}') if torch.cuda.is_available() else torch.device('cpu')
    print(f'Rank {local_rank} using device: {device}')
    
    if local_rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        
    # ----- LOAD DATASET -----
    train_dataset, test_dataset, img_size = load_dataset(
        args.dataset_file,
        device_idx=local_rank, 
        train_test_split=args.train_test_split,
        seed=args.seed
    )
    n_detectors = img_size
    
    world_size = dist.get_world_size()
    
    # distribute train dataset across processes
    train_sampler = DistributedSampler(
        train_dataset, 
        num_replicas=world_size, 
        rank=local_rank, 
        shuffle=False,
        #seed=args.seed,
        #drop_last=True
    )
    
    # distribute test dataset across processes
    test_sampler = DistributedSampler(
        test_dataset, 
        num_replicas=world_size, 
        rank=local_rank, 
        shuffle=False,
        drop_last=False
    )
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        sampler=train_sampler
    )
    
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        sampler=test_sampler
    )
    
    # ----- MODEL INITIALIZATION -----
    model = IterativeReconstructor(
        device_idx=local_rank, 
        img_size=img_size,
        n_detectors=n_detectors,
        n_angles=args.n_angles,  
        n_iterations=args.n_iterations, 
        alpha=args.alpha,
        chunk_size=args.chunk_size,
        d_model=args.d_model,
        headdim=args.headdim,
        aggregation=args.aggregation   
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    loss_fn = nn.MSELoss()
    ssim_module = SSIM(data_range=1.0, size_average=True, channel=1)
    
    # ----- LOAD MODEL CHECKPOINT IF AVAILABLE -----
    if os.path.exists(os.path.join(checkpoint_dir, args.model_file)) and args.resume_training:
        model_checkpoint = torch.load(
            os.path.join(checkpoint_dir, args.model_file), 
            map_location='cpu', 
            weights_only=True
        )
        model.load_state_dict(model_checkpoint['model_state_dict'])
        optimizer.load_state_dict(model_checkpoint['optimizer_state_dict'])
        start_epoch = model_checkpoint['epoch'] + 1
        if local_rank == 0: # use main process to load metric lists
            last_eval_loss = model_checkpoint['last_eval_loss']
            train_loss_values = model_checkpoint['train_loss_values']
            eval_loss_values = model_checkpoint['eval_loss_values']
            eval_ssim_values = model_checkpoint['eval_ssim_values']
            baseline_loss_values = model_checkpoint['baseline_loss_values']
            baseline_ssim_values = model_checkpoint['baseline_ssim_values']
        
        model.to(device)
        
        for state in optimizer.state.values():
            for i, v in state.items():
                if torch.is_tensor(v):
                    state[i] = v.to(device)
                    
        del model_checkpoint
        torch.cuda.empty_cache()

    else:
        model.to(device)
        
        start_epoch = 0
        if local_rank == 0:
            last_eval_loss = 0.0
            train_loss_values = []
            eval_loss_values = []
            eval_ssim_values = []
            baseline_loss_values = []
            baseline_ssim_values = []
            
    model = DDP(model, device_ids=[local_rank]) # wrap the model with DistributedDataParallel
        
    dist.barrier() # synchronize processes before continuing
    
    # ----- EVALUATE BEFORE TRAINING -----
    if not args.resume_training:
        gt_images, baseline_images, prediction_images, eval_baseline_loss, eval_final_loss, eval_baseline_ssim, eval_final_ssim = eval(
            model, 
            test_dataloader, 
            device, 
            loss_fn, 
            ssim_module, 
            epoch=-1
        )
        
        # aggregate across all processes
        dist.all_reduce(eval_baseline_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(eval_baseline_ssim, op=dist.ReduceOp.SUM)
        dist.all_reduce(eval_final_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(eval_final_ssim, op=dist.ReduceOp.SUM)
        eval_baseline_loss = eval_baseline_loss.item() / world_size
        eval_baseline_ssim = eval_baseline_ssim.item() / world_size
        eval_final_loss = eval_final_loss.item() / world_size
        eval_final_ssim = eval_final_ssim.item() / world_size
        
        if local_rank == 0:
            last_eval_loss = eval_final_loss
            print('\033[92mEpoch 0 (before training)\033[0m')
            print(f' -> Evaluation Loss: {eval_baseline_loss:.8f} (baseline) / {eval_final_loss:.8f} (final)') 
            print(f' -> SSIM (Evaluation): {eval_baseline_ssim:.8f} (baseline) / {eval_final_ssim:.8f} (final)')
            
            train_loss_values.append(None)
            baseline_loss_values.append(eval_baseline_loss)
            baseline_ssim_values.append(eval_baseline_ssim)
            eval_loss_values.append(eval_final_loss)
            eval_ssim_values.append(eval_final_ssim)
            
            # save model state and metric lists
            torch.save({
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': -1,
                'last_eval_loss': last_eval_loss,
                'train_loss_values': train_loss_values,
                'baseline_loss_values': baseline_loss_values,
                'baseline_ssim_values': baseline_ssim_values,
                'eval_loss_values': eval_loss_values,
                'eval_ssim_values': eval_ssim_values,
                'gt_images': gt_images,
                'baseline_images': baseline_images,
                'prediction_images': prediction_images,
            }, os.path.join(checkpoint_dir, args.model_file))
        
    dist.barrier() # synchronize processes before continuing
    
    scaler = torch.amp.GradScaler('cuda')
    
    # ----- TRAINING LOOP -----
    for epoch in range(start_epoch, args.n_epochs):
        torch.cuda.reset_peak_memory_stats()
        
        # ----- TRAINING -----
        train_loss = training(
            model, 
            train_dataloader, 
            device, 
            optimizer, 
            loss_fn, 
            scaler, 
            epoch, 
            args.n_epochs
        )
        #print(f' -> Rank {local_rank} / Train Loss: {train_loss:.8f}') 
        
        # aggregate across all processes
        dist.all_reduce(train_loss, op=dist.ReduceOp.SUM)
        train_loss = train_loss.item() / world_size
        if local_rank == 0:
            print(f' -> Train Loss: {train_loss:.8f}')

        dist.barrier() # synchronize processes before continuing
            
        # ----- EVALUATION AND CHECKPOINTING -----
        gt_images, baseline_images, prediction_images, eval_baseline_loss, eval_final_loss, eval_baseline_ssim, eval_final_ssim = eval(
            model, 
            test_dataloader, 
            device, 
            loss_fn, 
            ssim_module, 
            epoch
        )
        
        # aggregate across all processes
        dist.all_reduce(eval_baseline_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(eval_final_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(eval_baseline_ssim, op=dist.ReduceOp.SUM)
        dist.all_reduce(eval_final_ssim, op=dist.ReduceOp.SUM)
        eval_baseline_loss = eval_baseline_loss.item() / world_size
        eval_baseline_ssim = eval_baseline_ssim.item() / world_size
        eval_final_loss = eval_final_loss.item() / world_size
        eval_final_ssim = eval_final_ssim.item() / world_size

        if local_rank == 0:
            print(f' -> Evaluation Loss: {eval_baseline_loss:.8f} (baseline) / {eval_final_loss:.8f} (final)') 
            print(f' -> SSIM (Evaluation): {eval_baseline_ssim:.8f} (baseline) / {eval_final_ssim:.8f} (final)')
            if last_eval_loss > 0 and eval_final_loss > last_eval_loss:
                print('\033[93mWarning: Evaluation Loss increased!\033[0m')
            last_eval_loss = eval_final_loss 
            
            train_loss_values.append(train_loss)
            baseline_loss_values.append(eval_baseline_loss)
            baseline_ssim_values.append(eval_baseline_ssim)
            eval_loss_values.append(eval_final_loss)
            eval_ssim_values.append(eval_final_ssim)
            
            # save model state and metric lists
            torch.save({
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'last_eval_loss': last_eval_loss,
                'train_loss_values': train_loss_values,
                'baseline_loss_values': baseline_loss_values,
                'baseline_ssim_values': baseline_ssim_values,
                'eval_loss_values': eval_loss_values,
                'eval_ssim_values': eval_ssim_values,
                'gt_images': gt_images,
                'baseline_images': baseline_images,
                'prediction_images': prediction_images,
            }, os.path.join(checkpoint_dir, args.model_file))
        
        dist.barrier() 
        
        scheduler.step()
        
        memory = torch.cuda.max_memory_allocated(local_rank)/1024**3
        memories = [None] * world_size if local_rank == 0 else None
        dist.gather_object(memory, memories, dst=0)
        if local_rank == 0:
            print('Memory Allocation:')
            print(' | '.join([f'GPU {i}: {mem:.2f} GB' for i, mem in enumerate(memories)]))
        
    if hasattr(model.module, 'astra_op'):
        model.module.astra_op.cleanup()
  

if __name__ == '__main__':
    try:
        main()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group() 