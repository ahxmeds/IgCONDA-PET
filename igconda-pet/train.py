#%%
import os
import time
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import sys
from monai.utils import set_determinism
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import argparse
import pandas as pd 
from generative.inferers import DiffusionInferer
from generative.networks.nets.diffusion_model_unet import DiffusionModelUNet
from generative.networks.schedulers import DDIMScheduler
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist 
from get_data import get_train_valid_datasets
torch.multiprocessing.set_sharing_strategy("file_system")
from monai.data import DataLoader, CacheDataset
torch.backends.cudnn.benchmark = True
WORKING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(WORKING_DIR)
from config import RESULTSDIR
from utils.utils import pad_zeros_at_front, str2bool
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
#%%
def ddp_setup():
    dist.init_process_group(backend='nccl', init_method="env://")

set_determinism(42)

def main_worker(models_dir, logs_dir, args):
    # init_process_group
    ddp_setup() 
    # get local rank on the GPU
    local_rank = int(dist.get_rank())
    if local_rank == 0:
        print(f"The models will be saved in {models_dir}")
        print(f"The training/validation logs will be saved in {logs_dir}")

    dataset_train, dataset_valid = get_train_valid_datasets(args.cache_rate)
    print(f'Total number of images: {dataset_train}')

    sampler_train = DistributedSampler(dataset=dataset_train, shuffle=True)
    sampler_valid = DistributedSampler(dataset=dataset_valid, shuffle=False)
    
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=sampler_train,
        num_workers=args.num_workers
    )
    dataloader_valid = DataLoader(
        dataset_valid,
        batch_size=args.batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=sampler_valid,
        num_workers=args.num_workers
    )
    
    trainlog_fpath = os.path.join(logs_dir, f'trainlog_gpu{local_rank}.csv')
    validlog_fpath = os.path.join(logs_dir, f'validlog_gpu{local_rank}.csv')
    
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    
    embedding_dimension = 64
    model = DiffusionModelUNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        num_channels=(64, 64, 64),
        attention_levels=(args.attn_layer1, args.attn_layer2, args.attn_layer3),
        num_res_blocks=1,
        num_head_channels=16,
        with_conditioning=True,
        cross_attention_dim=embedding_dimension,
    ).to(device)
    embed = torch.nn.Embedding(num_embeddings=3, embedding_dim=embedding_dimension, padding_idx=0).to(device)
    scheduler = DDIMScheduler(num_train_timesteps=1000)
    
    start_epoch = 0
    
    model = DDP(model, device_ids=[device])
    embed = DDP(embed, device_ids=[device])

    optimizer = torch.optim.Adam(params=list(model.parameters()) + list(embed.parameters()), lr=1e-5)
    inferer = DiffusionInferer(scheduler)
    
    condition_dropout = 0.15
    n_epochs = args.epochs
    val_interval = args.val_interval
    
    train_epoch_loss_list = [] 
    valid_epoch_loss_list = []

    scaler = GradScaler()
    experiment_start_time = time.time()
    
    for epoch in range(n_epochs):
        epoch_start_time = time.time()
        model.train()
        epoch_loss = 0
        sampler_train.set_epoch(epoch)
        progress_bar = tqdm(enumerate(dataloader_train), total=len(dataloader_train), ncols=80)
        progress_bar.set_description(f"Epoch {start_epoch + epoch + 1}")
        for step, batch in progress_bar:
            images, classes = batch['PT'].to(device), batch['Label'].to(device)
            # images, classes = batch[0].to(device), batch[1].to(device)
            
            classes = classes * (torch.rand_like(classes) > condition_dropout)
            class_embedding = embed(classes.long().to(device)).unsqueeze(1)
            optimizer.zero_grad(set_to_none=True)
            timesteps = torch.randint(0, 1000, (len(images),)).to(device)

            with autocast('cuda'):
                noise = torch.randn_like(images).to(device)
                noise_pred = inferer(inputs=images, diffusion_model=model, noise=noise, timesteps=timesteps, condition=class_embedding)
                loss = F.mse_loss(noise_pred.float(), noise.float())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            progress_bar.set_postfix({f"GPU[{local_rank}]: Train Loss": round(epoch_loss / (step + 1), 6)})
        
        train_epoch_loss_list.append((epoch_loss / (step + 1)))
        trainlog_df = pd.DataFrame(train_epoch_loss_list, columns=['Loss'])
        trainlog_df.to_csv(trainlog_fpath, index=False)

        
        if (epoch + 1) % val_interval == 0:
            progress_bar = tqdm(enumerate(dataloader_valid), total=len(dataloader_valid), ncols=80)
            progress_bar.set_description(f"Epoch {start_epoch + epoch + 1}")
            model.eval()
            val_epoch_loss = 0
            for step, batch in progress_bar:
                # images, classes = batch['PT'].to(device), batch['Label'].to(device)
                images, classes = batch['PT'].to(device), batch['Label'].to(device)
                # cross attention expects shape [batch size, sequence length, channels]
                class_embedding = embed(classes.long().to(device)).unsqueeze(1)
                timesteps = torch.randint(0, 1000, (len(images),)).to(device)
                with torch.no_grad():
                    with autocast('cuda'):
                        noise = torch.randn_like(images).to(device)
                        noise_pred = inferer(
                            inputs=images,
                            diffusion_model=model,
                            noise=noise,
                            timesteps=timesteps,
                            condition=class_embedding,
                        )
                        val_loss = F.mse_loss(noise_pred.float(), noise.float())
                
                val_epoch_loss += val_loss.item()
                progress_bar.set_postfix({f"GPU[{local_rank}]: Valid Loss": round(val_epoch_loss / (step + 1), 6)})
                
            valid_epoch_loss_list.append(val_epoch_loss / (step + 1))
            validlog_df = pd.DataFrame(valid_epoch_loss_list, columns=['Loss'])
            validlog_df.to_csv(validlog_fpath, index=False)

            saved_dict = {
                'model_state_dict': model.module.state_dict(),
                'embed_state_dict': embed.module.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }

            # Save the dictionary to a .pth file
            checkpoint_fpath = os.path.join(models_dir, f'chkpt_ep{pad_zeros_at_front(start_epoch+epoch+1, 5)}.pth')
            torch.save(saved_dict, checkpoint_fpath)

        epoch_end_time = (time.time() - epoch_start_time)
        print(f"[GPU:{local_rank}]: Epoch {start_epoch + epoch + 1} time: {round(epoch_end_time,2)} sec")
        
    experiment_end_time = (time.time() - experiment_start_time)/(60)
    print(f"[GPU:{local_rank}]: Total time: {round(experiment_end_time,2)} min")    
    print('Destroying process')
    dist.destroy_process_group()
    print('Destroyed process')
  
#%%
def main(args):
    os.environ['OMP_NUM_THREADS'] = '6'
    experiment_code = args.experiment 
    RESULTS_FOLDER = os.path.join(RESULTSDIR, experiment_code)   
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    
    save_models_dir = os.path.join(RESULTS_FOLDER, 'models')
    os.makedirs(save_models_dir, exist_ok=True)
    
    save_logs_dir = os.path.join(RESULTS_FOLDER,'logs')
    os.makedirs(save_logs_dir, exist_ok=True)
    
    main_worker(save_models_dir, save_logs_dir, args)
    
    

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='IgCONDA-PET: Counterfactual Diffusion with Implicit Guidance for PET Anomaly Detection - Training phase')
    parser.add_argument('--experiment', type=str, default='exp0', metavar='exp',
                        help='experiment identifier')
    parser.add_argument('--attn-layer1', type=str2bool, default=False, metavar='attn1',
                        help='whether to put attention mechanism in layer 1 (default=False)')
    parser.add_argument('--attn-layer2', type=str2bool, default=True, metavar='attn2',
                        help='whether to put attention mechanism in layer 2 (default=True)')
    parser.add_argument('--attn-layer3', type=str2bool, default=True, metavar='attn3',
                        help='whether to put attention mechanism in layer 3 (default=True)')
    parser.add_argument('--epochs', type=int, default=400, metavar='epochs',
                        help='number of epochs to train (default=400)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='bs',
                        help='mini-batchsize for training/validation (default=64)')
    parser.add_argument('--num-workers', type=int, default=4, metavar='nw',
                        help='num_workers for train and validation dataloaders (default=4)')
    parser.add_argument('--cache-rate', type=float, default=1, metavar='cr',
                        help='cache_rate for CacheDataset from MONAI (default=1)')
    parser.add_argument('--val-interval', type=int, default=10, metavar='val-interval',
                        help='epochs interval for which validation will be performed (default=10)')
    args = parser.parse_args()
    
    main(args)
    
    

  
