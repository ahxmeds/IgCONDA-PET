#%%
import argparse
import time
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import sys
from monai.data import CacheDataset, DataLoader
from tqdm import tqdm
from generative.networks.nets.diffusion_model_unet import DiffusionModelUNet
from generative.networks.schedulers.ddim import DDIMScheduler
torch.multiprocessing.set_sharing_strategy("file_system")
import time 
import pandas as pd 
from joblib import Parallel, delayed
WORKING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(WORKING_DIR)
from utils.utils import str2bool, pad_zeros_at_front
from get_data import get_test_dataset
from config import RESULTSDIR

#%%
def process_batch(batch, local_rank, model, embed, scheduler, total_timesteps, device, guidance_scale, noise_level, test_preds_dir, test_visuals_dir):
    start_ = time.time()
    torch.cuda.empty_cache()
    images = batch['PT'].to(device)
    gts = batch['GT'].to(device)
    batch_size = images.shape[0]
    
    fnames = [os.path.basename(batch['GT_meta_dict']['filename_or_obj'][i])[:-10] for i in range(batch_size)]
    impaths = [os.path.join(test_preds_dir, f'{fname}.npy') for fname in fnames]
    vizpaths = [os.path.join(test_visuals_dir, f'{fname}.png')  for fname in fnames]
    PTs = images[:, 0].clone().detach().cpu().numpy()
    GTs = gts[:, 0].clone().detach().cpu().numpy()

    model.eval()

    current_img = images.to(device)
    scheduler.set_timesteps(num_inference_steps=total_timesteps)

    ## Enconding step
    # Encoding via class conditioning using an unconditional model 
    # (Notice the conditioning variable c=0)
    scheduler.clip_sample = False
    conditioning = torch.zeros(1).long().to(device)
    class_embedding = embed(conditioning).unsqueeze(dim=1)
    class_embedding_batch = class_embedding.repeat(batch_size, 1, 1)
    progress_bar = tqdm(range(noise_level), desc=f'Noising batch [GPU{local_rank}]')
    for i in progress_bar:  # noising process for args.noise_level steps
        t = i
        # with autocast('cuda'):
        with torch.no_grad():
            t_batch = torch.Tensor([t]*batch_size).to(current_img.device)
            model_output = model(current_img, timesteps=t_batch, context=class_embedding_batch)
        current_img, _ = scheduler.reversed_step(model_output, t, current_img)
        progress_bar.set_postfix({"timestep input": t})

    latent_image = current_img
    LTs = latent_image[:, 0].clone().detach().cpu().numpy() # latent image after args.noise_level steps of noise encoding

    ## Deconding step
    # Decoding via class conditioning using both conditional (c=2 where 2:unhealthy class) and unconditional models (c=0)
    # After this we employ, implicit guidance for healthy counterfactual generation  
    conditioning = torch.cat([torch.zeros(1).long(), torch.ones(1).long()], dim=0).to(device)
    class_embedding = embed(conditioning).unsqueeze(dim=1)
    class_embedding_batch = torch.zeros(2*batch_size, 1, 64).to(device)
    class_embedding_batch[0:batch_size] = class_embedding[0].unsqueeze(dim=1).repeat(batch_size, 1, 1)
    class_embedding_batch[batch_size:2*batch_size] = class_embedding[1].unsqueeze(dim=1).repeat(batch_size, 1, 1)
    # class_embedding_batch = class_embedding.repeat(1, batch_size, 1)
    progress_bar = tqdm(range(noise_level), desc=f'Denoising batch [GPU{local_rank}]')
    for i in progress_bar:  # denoising process for args.noise_level steps
        t = noise_level - i
        current_img_double = torch.cat([current_img] * 2)
        # with autocast('cuda'):
        with torch.no_grad():
            t_batch = torch.Tensor([t, t]*batch_size).to(current_img.device)
            model_output = model(current_img_double, timesteps=t_batch, context=class_embedding_batch)
        noise_pred_uncond, noise_pred_text = model_output.chunk(2)
        # the equation below is called implicit or classifier-free guidance
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        current_img, _ = scheduler.step(noise_pred, t, current_img)
        progress_bar.set_postfix({"timestep input": t})
        torch.cuda.empty_cache()
    
    # healthy counterfactual generated after noise encoding and decoding via implicit guidance
    HLs = current_img[:, 0].clone().detach().cpu().numpy() 

    # anomaly map generation
    ANs = abs(PTs - HLs)
    
    def save_images_and_plots(i):
    # for i in range(batch_size):
        pt, gt, lt, hl, an = PTs[i], GTs[i], LTs[i], HLs[i], ANs[i]
        image_stack = np.zeros((5, 64, 64))
        image_stack[0, ...] = pt
        image_stack[1, ...] = gt
        image_stack[2, ...] = lt
        image_stack[3, ...] = hl
        image_stack[4, ...] = an
        np.save(impaths[i], image_stack)
    
        # image plotting
        fig, ax = plt.subplots(1, 5, figsize=(10, 30))
        fig.patch.set_facecolor('white')
        fig.patch.set_alpha(1)
        im0 = ax[0].imshow(np.rot90(pt), vmin=0, vmax=1, cmap="nipy_spectral")
        ax[0].set_title("Unhealthy\nPET slice")
        ax[0].axis('off')
        im1 = ax[1].imshow(np.rot90(lt), vmin=0, vmax=1, cmap="gray")
        ax[1].set_title("Latent\nimage")
        ax[1].axis('off')
        im2 = ax[2].imshow(np.rot90(hl), vmin=0, vmax=1, cmap="nipy_spectral")
        ax[2].set_title("Healthy\nreconstruction")
        ax[2].axis('off')
        im3 = ax[3].imshow(np.rot90(an), cmap="inferno")
        ax[3].set_title("Anomaly\nmap")
        ax[3].axis('off')
        im4 = ax[4].imshow(np.rot90(gt), cmap="gray")
        ax[4].set_title("Ground\ntruth")
        ax[4].axis('off')
        
        ims = [im0, im1, im2, im3, im4]
        for kk in range(5):
            fig.colorbar(ims[kk], ax=ax[kk], orientation='horizontal', pad=0.01)
        plt.subplots_adjust(wspace=0.1)

        fig.savefig(vizpaths[i], dpi=200, bbox_inches='tight')
        plt.close('all')
    
    Parallel(n_jobs=8)(delayed(save_images_and_plots)(i) for i in range(batch_size))

    print(f'Finished inference on batch [GPU{local_rank}]: {fnames}')
    elapsed_ = time.time() - start_
    print(f'Time for this batch [GPU{local_rank}]: {elapsed_/60:2f} min')
    return HLs

#%%
def main(args):
    start = time.time() 
    device_id = args.device_id

    results_dir = os.path.join(RESULTSDIR, f'{args.experiment}')
    model_dir = os.path.join(results_dir, 'models')
    logs_dir = os.path.join(results_dir, 'logs')
    validlog_fpath = os.path.join(logs_dir, 'validlog_gpu0.csv')
    validlog_df = pd.read_csv(validlog_fpath)
    val_interval = args.val_interval
    best_epoch = val_interval*(1 + np.argmin(validlog_df['Loss'])) 
    print(f'Best epoch: {best_epoch}')
    best_checkpoint_fpath = os.path.join(model_dir, f'chkpt_ep{pad_zeros_at_front(best_epoch, 5)}.pth')
    
    guidance_scale = args.guidance_scale
    noise_level = args.noise_level
   
    test_preds_dir = os.path.join(results_dir, f'test_preds_w={guidance_scale}_D={noise_level}')
    test_visuals_dir = os.path.join(results_dir, f'test_visuals_w={guidance_scale}_D={noise_level}')
    os.makedirs(test_preds_dir, exist_ok=True)
    os.makedirs(test_visuals_dir, exist_ok=True)
    
    device = torch.device(f"cuda:{device_id}")
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
    scheduler = DDIMScheduler(num_train_timesteps=1000).to(device)
    
    total_timesteps = 1000

    best_checkpoint = torch.load(best_checkpoint_fpath, map_location=device, weights_only=True)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    embed.load_state_dict(best_checkpoint['embed_state_dict'])
    scheduler.load_state_dict(best_checkpoint['scheduler_state_dict'])

    datalist, data_transforms = get_test_dataset(label='2')
    dataset = CacheDataset(datalist, transform=data_transforms, cache_rate=1)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4, drop_last=False, persistent_workers=True)

    
    def process_a_batch(batch):
        process_batch(batch, device_id, model, embed, scheduler, total_timesteps, device, guidance_scale, noise_level, test_preds_dir, test_visuals_dir)

    # Parallel processing: running args.num_workers jobs in parallel. You can change args.num_workers flag to higher values 
    # depending on the availability of GPU memory due to parallelization.  
    Parallel(n_jobs=1, backend="loky")(delayed(process_a_batch)(batch) for batch in dataloader)

    elapsed = time.time() - start 
    print(f'Time taken: {elapsed/(60*60)} hrs')



if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='IgCONDA-PET: Counterfactual Diffusion with Implicit Guidance for PET Anomaly Detection - Test phase')
    parser.add_argument('--experiment', type=str, default='exp0', metavar='exp',
                        help='experiment identifier')
    parser.add_argument('--device-id', type=int, default=0, metavar='devid',
                        help='device id (default=0)')
    parser.add_argument('--attn-layer1', type=str2bool, default=False, metavar='attn1',
                        help='whether to put attention mechanism in layer 1 (default=False)')
    parser.add_argument('--attn-layer2', type=str2bool, default=True, metavar='attn2',
                        help='whether to put attention mechanism in layer 2 (default=True)')
    parser.add_argument('--attn-layer3', type=str2bool, default=True, metavar='attn3',
                        help='whether to put attention mechanism in layer 3 (default=True)')
    parser.add_argument('--guidance-scale', type=float, default=3.0, metavar='w',
                        help='Guidance scale for performing implicit guidance (default=3.0)')
    parser.add_argument('--noise-level', type=int, default=400, metavar='D',
                        help='number of noising and denoising steps for inference (default=400)')
    parser.add_argument('--num-workers', type=int, default=4, metavar='nw',
                        help='num_workers for train and validation dataloaders (default=4)')
    parser.add_argument('--cache-rate', type=float, default=1, metavar='cr',
                        help='cache_rate for CacheDataset from MONAI (default=1)')
    parser.add_argument('--val-interval', type=int, default=10, metavar='val-interval',
                        help='epochs interval for which validation will be performed (default=10)')
    args = parser.parse_args()
    
    main(args)
# %%
