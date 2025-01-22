#%%
import pandas as pd 
import os 
from glob import glob 
import numpy as np 
import matplotlib.pyplot as plt 
import SimpleITK as sitk 
from monai import transforms 
from monai.data import Dataset, CacheDataset, DataLoader
from joblib import Parallel, delayed
# %%
def read_image_array(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(path)).T
#%%
dataset = 'hecktor'
data_orig_dir = '/data/blobfuse/default/hecktor_star_shape_loss_results/data/raw' # change this
ptpaths = sorted(glob(os.path.join(data_orig_dir, 'images', '*PT.nii.gz')))
gtpaths = sorted(glob(os.path.join(data_orig_dir, 'labels', '*.nii.gz')))
datalist = [{'PT': ptpath, 'GT': gtpath} for ptpath, gtpath in zip(ptpaths, gtpaths)]
#%%
save_main_dir = '/data/blobfuse/default/diffusion2d_wsad/data/'
dataformat = 'ptgt'
mod_keys = ['CT', 'PT', 'GT'] if dataformat == 'ctptgt' else ['PT', 'GT']
autopet_spacing = (2.0364201068878174, 2.0364201068878174, 3.0)
resampling_mode = ['bilinear', 'bilinear', 'nearest'] if dataformat == 'ctptgt' else ['bilinear', 'nearest']
processed_images_dir = os.path.join(save_main_dir, dataset, 'images')
processed_labels_dir = os.path.join(save_main_dir, dataset, 'labels')
os.makedirs(processed_images_dir, exist_ok=True)
os.makedirs(processed_labels_dir, exist_ok=True)
#%%
data_transforms = transforms.Compose(
        [
            transforms.LoadImaged(keys=mod_keys, image_only=True),
            transforms.EnsureChannelFirstd(keys=mod_keys),
            transforms.Spacingd(keys=mod_keys, pixdim=autopet_spacing, mode=resampling_mode),
            transforms.CenterSpatialCropd(keys=mod_keys, roi_size=[192, 192, 288]),
            transforms.ScaleIntensityd(keys=['PT']),
            transforms.Resized(keys=mod_keys, spatial_size=(64, 64, 96), mode=resampling_mode),
            transforms.SqueezeDimd(keys=mod_keys, dim=0),
            transforms.SaveImaged(keys=['PT'], output_dir = processed_images_dir, output_postfix='', separate_folder=False),
            transforms.SaveImaged(keys=['GT'], output_dir = processed_labels_dir, output_postfix='', separate_folder=False)
        ]
    )
#%%
dataset = Dataset(datalist, transform=data_transforms)
dataloader = DataLoader(dataset, batch_size=1, pin_memory=True, num_workers=6)

def process_one_batch(data):
    return 
# %%
Parallel(n_jobs=-1)(
    delayed(process_one_batch)(data) for data in dataloader
)
