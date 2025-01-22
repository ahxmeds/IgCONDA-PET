#%%
from monai.data import CacheDataset
import pandas as pd 
import numpy as np 
import os 
from glob import glob 
from monai import transforms 
import torch 
import sys 
WORKING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(WORKING_DIR)
from config import METADIR, DATA2DDIR
#%%
def get_dict_datalist(df, with_GT=False):
    datalist = []
    ptfnames, labels = df['PTPATH'].tolist(), df['Label'].tolist()
    if with_GT:
        gtfnames = df['GTPATH'].tolist()
    for i in range(len(ptfnames)):
        ptpath = os.path.join(DATA2DDIR, ptfnames[i])
        label = float(labels[i])
        if with_GT:
            gtpath = os.path.join(DATA2DDIR, gtfnames[i])
            datalist.append({'PT':ptpath, 'GT':gtpath, 'Label': label})
        else:
            datalist.append({'PT':ptpath, 'Label': label})
    return datalist 

#%%
def get_train_valid_datasets(cacherate=1.0):
    train2D_fpath = os.path.join(METADIR, 'TRAIN_data2D_info.csv')
    valid2D_fpath = os.path.join(METADIR, 'VALID_data2D_info.csv')
    
    train_df = pd.read_csv(train2D_fpath)
    valid_df = pd.read_csv(valid2D_fpath)
    
    datalist_train = get_dict_datalist(train_df)
    datalist_valid = get_dict_datalist(valid_df)
    
    data_transforms = transforms.Compose(
        [
            transforms.LoadImaged(keys=['PT'], image_only=True),
            transforms.EnsureChannelFirstd(keys=['PT']),    
        ]
    )
    
    dataset_train = CacheDataset(datalist_train, transform=data_transforms, cache_rate=cacherate)
    dataset_valid = CacheDataset(datalist_valid, transform=data_transforms, cache_rate=cacherate)
    
    return dataset_train, dataset_valid

#%%
def get_hpablation_set_3d():
    datainfo3d_path = '/home/jhubadmin/Projects/diffusion2d_wsad/data_analysis/data3D_split/data3D_info.csv'
    df = pd.read_csv(datainfo3d_path)
    df = df[df['Dataset'] != 'sts'].reset_index(drop=True)
    df = df[df['Split'] == 'TEST'].reset_index(drop=True)
    df_autopet = df[df['Dataset'] == 'autopet'].reset_index(drop=True)
    df_hecktor = df[df['Dataset'] == 'hecktor'].reset_index(drop=True)
    df_dlbclbccv = df[df['Dataset'] == 'dlbcl-bccv'].reset_index(drop=True)
    df_pmbclbccv = df[df['Dataset'] == 'pmbcl-bccv'].reset_index(drop=True)
    df_dlbclsmhs = df[df['Dataset'] == 'dlbcl-smhs'].reset_index(drop=True)
    
    df_autopet_sample = df_autopet.sample(n=2, random_state=42).reset_index(drop=True)
    df_hecktor_sample = df_hecktor.sample(n=2, random_state=42).reset_index(drop=True)
    df_dlbclbccv_sample = df_dlbclbccv.sample(n=2, random_state=42).reset_index(drop=True)
    df_pmbclbccv_sample = df_pmbclbccv.sample(n=2, random_state=42).reset_index(drop=True)
    df_dlbclsmhs_sample = df_dlbclsmhs.sample(n=2, random_state=42).reset_index(drop=True)
    
    df_sample = pd.concat([df_autopet_sample, df_hecktor_sample, df_dlbclbccv_sample, df_pmbclbccv_sample, df_dlbclsmhs_sample]).reset_index(drop=True)
    return df_sample
#%%
def get_hpablation_set_2d():
    datainfo2d_path = '/home/jhubadmin/Projects/diffusion2d_wsad/data_analysis/data2D_split/TEST_data2D_info.csv'
    df2d = pd.read_csv(datainfo2d_path)
    df3d = get_hpablation_set_3d()
    ablation_ptids = df3d['PatientID'].tolist()
    
    df2d_ablation_ptids = df2d[df2d['PatientID'].isin(ablation_ptids)] 
    df2d_ablation_ptids_pos = df2d_ablation_ptids[df2d_ablation_ptids['Label'] == 2].reset_index(drop=True)
    return df2d_ablation_ptids_pos

def get_ablation_dataset():
    df2d_ablation = get_hpablation_set_2d()
    datalist = get_dict_datalist(df2d_ablation, with_GT=True)
    data_transforms = transforms.Compose(
        [
            transforms.LoadImaged(keys=['PT', 'GT'], image_only=False),
            transforms.EnsureChannelFirstd(keys=['PT', 'GT']),           
        ])
    
    return datalist, data_transforms
    
#%%
def get_test_dataset(label='2'): # label = '1', '2', 'all'
    test2D_fpath = os.path.join(METADIR, 'TEST_data2D_info.csv')
    df = pd.read_csv(test2D_fpath)
    ablation_ptids = get_hpablation_set_3d()['PatientID'].tolist()
    df = df[~df['PatientID'].isin(ablation_ptids)] # remove ablation patients
    
    if label != 'all':
        df_label = df[df['Label'] == int(label)]
    else:
        df_label = df
    df_label.reset_index(inplace=True, drop=True)
    datalist_test = get_dict_datalist(df_label, with_GT=True)

    data_transforms = transforms.Compose(
        [
            transforms.LoadImaged(keys=['PT', 'GT'], image_only=False),
            transforms.EnsureChannelFirstd(keys=['PT', 'GT']),           
        ])
    
    return datalist_test, data_transforms


def get_dict_datalist_random(num_samples, image_shape, with_GT=False):
    """
    Generates a list of dictionaries, each with a random tensor for the PT (and optionally GT) key and a random label (1 or 2).
    
    Args:
        num_samples (int): Number of samples to generate.
        image_shape (tuple): The shape of the image tensor (e.g., (1, 64, 64) for single-channel 64x64 images).
        with_GT (bool): Whether to include ground truth (GT) data in the datalist.
        
    Returns:
        list: A list of dictionaries containing random 'PT', 'GT' (optional), and 'Label'.
    """
    datalist = []
    for _ in range(num_samples):
        pt_data = torch.rand(image_shape)  # Generate random image data for PT
        label = float(np.random.randint(1, 3))    # Generate a random label (either 1 or 2)
        
        if with_GT:
            gt_data = torch.rand(image_shape)  # Generate random image data for GT
            datalist.append({'PT': pt_data, 'GT': gt_data, 'Label': label})
        else:
            datalist.append({'PT': pt_data, 'Label': label})
    
    return datalist


def get_train_valid_datasets_random(cacherate=1.0, num_train_samples=212160, num_valid_samples=14304, image_shape=(1, 64, 64)):
    # Generate random datasets instead of loading from disk
    datalist_train = get_dict_datalist_random(num_train_samples, image_shape)
    datalist_valid = get_dict_datalist_random(num_valid_samples, image_shape)
    
    # Define transformations (could be empty if not needed for random data)
    data_transforms = transforms.Compose(
        [
            transforms.EnsureChannelFirstd(keys=['PT']),  # This ensures the correct channel format
        ]
    )
    
    # Use CacheDataset to handle caching of random data (cacherate can be adjusted)
    dataset_train = CacheDataset(data=datalist_train, cache_rate=cacherate)
    dataset_valid = CacheDataset(data=datalist_valid, cache_rate=cacherate)
    
    return dataset_train, dataset_valid
# %%
