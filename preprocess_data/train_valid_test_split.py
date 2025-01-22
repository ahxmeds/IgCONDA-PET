#%%
import pandas as pd 
import numpy as np 
import os 
from glob import glob 
import SimpleITK as sitk 
import random
# %%
def read_image_array(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(path)).T

def get_datainfo_from_imageids(imageids_list, dataset, diag, split='train'):
    splits = [split]*len(imageids_list)
    datasets = [dataset]*len(imageids_list)
    tracers = ['FDG']*len(imageids_list)
    diagnosis = [diag]*len(imageids_list)
    return imageids_list, datasets, tracers, diagnosis, splits

def get_datainfo_from_imageids_autopet(imageids_list, df, dataset, split='train'):
    splits = [split]*len(imageids_list)
    datasets = [dataset]*len(imageids_list)
    tracers, diagnosis = [],[]
    for id in imageids_list:
        df_id = df[df['PatientID'] == id]
        tracers.append(df_id['Tracer'].item())
        diagnosis.append(df_id['Diagnosis'].item())
    return imageids_list, datasets, tracers, diagnosis, splits
#%%
#################################################################################################
#####################################        AUTOPET        #####################################
#################################################################################################
dataset = 'autopet'
main_dir = f'/data/blobfuse/default/diffusion2d_wsad/data/{dataset}'
ptpaths = sorted(glob(os.path.join(main_dir, 'images', '*.nii.gz')))
gtpaths = sorted(glob(os.path.join(main_dir, 'labels', '*.nii.gz')))
num_train, num_valid, num_test = 1369, 92, 150 #autopet

ImageIDs_pos = [] 
ImageIDs_neg = []
for path in gtpaths:
    imageid = os.path.basename(path)[:-7]
    gt = read_image_array(path)
    if np.all(gt == 0):
        ImageIDs_neg.append(imageid)
    else:
        ImageIDs_pos.append(imageid)
        
# sampling for test set
fdg_pos_ids = [id for id in ImageIDs_pos if id.startswith('fdg')]
psma_pos_ids = [id for id in ImageIDs_pos if id.startswith('psma')]
random.seed(42)
fdg_pos_test = random.sample(fdg_pos_ids, num_test//2)
psma_pos_test = random.sample(psma_pos_ids, num_test//2)
test_ids = fdg_pos_test + psma_pos_test
trainvalid_ids = list(set(ImageIDs_neg) | set(ImageIDs_pos) - set(test_ids)) 
valid_ids = random.sample(trainvalid_ids, num_valid)
train_ids = list(set(trainvalid_ids) - set(valid_ids)) 
# %%
df = pd.read_csv('/home/jhubadmin/Projects/autopet2024/task1/monai/data_analysis/datainfo.csv')
#%%
train_ids, train_datasets, train_tracers, train_diagnosis, train_splits = get_datainfo_from_imageids_autopet(train_ids, df, dataset, split='TRAIN')
valid_ids, valid_datasets, valid_tracers, valid_diagnosis, valid_splits = get_datainfo_from_imageids_autopet(valid_ids, df, dataset, split='VALID')
test_ids, test_datasets, test_tracers, test_diagnosis, test_splits = get_datainfo_from_imageids_autopet(test_ids, df, dataset, split='TEST')

df_autopet = pd.DataFrame(
    {
        'PatientID': train_ids + valid_ids + test_ids,
        'Dataset': train_datasets + valid_datasets + test_datasets,
        'Tracer': train_tracers + valid_tracers + test_tracers,
        'Diagnosis': train_diagnosis + valid_diagnosis + test_diagnosis,
        'Split': train_splits + valid_splits + test_splits,
    }
)
df_autopet.to_csv(f'{dataset}_info.csv', index=False)
# %%
#################################################################################################
######################################       HECKTOR        #####################################
#################################################################################################

dataset = 'hecktor'
main_dir = f'/data/blobfuse/default/diffusion2d_wsad/data/{dataset}'
ptpaths = sorted(glob(os.path.join(main_dir, 'images', '*.nii.gz')))
gtpaths = sorted(glob(os.path.join(main_dir, 'labels', '*.nii.gz')))
num_train, num_valid, num_test = 445, 30, 49 # hecktor

ImageIDs_pos = [] 
ImageIDs_neg = []
for path in gtpaths:
    imageid = os.path.basename(path)[:-7]
    gt = read_image_array(path)
    if np.all(gt == 0):
        ImageIDs_neg.append(imageid)
    else:
        ImageIDs_pos.append(imageid)

# %%
# All HECKTOR images are positive
random.seed(42)
test_ids = random.sample(ImageIDs_pos, num_test)
trainvalid_ids = list(set(ImageIDs_neg) | set(ImageIDs_pos) - set(test_ids)) 
valid_ids = random.sample(trainvalid_ids, num_valid)
train_ids = list(set(trainvalid_ids) - set(valid_ids)) 
# %%

diag = 'HEADNECK'
train_ids, train_datasets, train_tracers, train_diagnosis, train_splits = get_datainfo_from_imageids(train_ids, dataset, diag, split='TRAIN')
valid_ids, valid_datasets, valid_tracers, valid_diagnosis, valid_splits = get_datainfo_from_imageids(valid_ids, dataset, diag, split='VALID')
test_ids, test_datasets, test_tracers, test_diagnosis, test_splits = get_datainfo_from_imageids(test_ids, dataset, diag, split='TEST')

df_hecktor = pd.DataFrame(
    {
        'PatientID': train_ids + valid_ids + test_ids,
        'Dataset': train_datasets + valid_datasets + test_datasets,
        'Tracer': train_tracers + valid_tracers + test_tracers,
        'Diagnosis': train_diagnosis + valid_diagnosis + test_diagnosis,
        'Split': train_splits + valid_splits + test_splits,
    }
)
df_hecktor.to_csv(f'{dataset}_info.csv', index=False)
# %%
#################################################################################################
######################################       DLBCL-BCCV        ##################################
#################################################################################################

dataset = 'dlbcl-bccv'
main_dir = f'/data/blobfuse/default/diffusion2d_wsad/data/{dataset}'
ptpaths = sorted(glob(os.path.join(main_dir, 'images', '*.nii.gz')))
gtpaths = sorted(glob(os.path.join(main_dir, 'labels', '*.nii.gz')))
num_train, num_valid, num_test = 91, 6, 10 # dlbcl-bccv

ImageIDs_pos = [] 
ImageIDs_neg = []
for path in gtpaths:
    imageid = os.path.basename(path)[:-7]
    gt = read_image_array(path)
    if np.all(gt == 0):
        ImageIDs_neg.append(imageid)
    else:
        ImageIDs_pos.append(imageid)

# %%
random.seed(42)
test_ids = random.sample(ImageIDs_pos, num_test)
trainvalid_ids = list(set(ImageIDs_neg) | set(ImageIDs_pos) - set(test_ids)) 
valid_ids = random.sample(trainvalid_ids, num_valid)
train_ids = list(set(trainvalid_ids) - set(valid_ids)) 
# %%

diag = 'DLBCL'
train_ids, train_datasets, train_tracers, train_diagnosis, train_splits = get_datainfo_from_imageids(train_ids, dataset, diag, split='TRAIN')
valid_ids, valid_datasets, valid_tracers, valid_diagnosis, valid_splits = get_datainfo_from_imageids(valid_ids, dataset, diag, split='VALID')
test_ids, test_datasets, test_tracers, test_diagnosis, test_splits = get_datainfo_from_imageids(test_ids, dataset, diag, split='TEST')

df_dlbclbccv = pd.DataFrame(
    {
        'PatientID': train_ids + valid_ids + test_ids,
        'Dataset': train_datasets + valid_datasets + test_datasets,
        'Tracer': train_tracers + valid_tracers + test_tracers,
        'Diagnosis': train_diagnosis + valid_diagnosis + test_diagnosis,
        'Split': train_splits + valid_splits + test_splits,
    }
)
df_dlbclbccv.to_csv(f'{dataset}_info.csv', index=False)
# %%
#################################################################################################
######################################       PMBCL-BCCV        ##################################
#################################################################################################
dataset = 'pmbcl-bccv'
main_dir = f'/data/blobfuse/default/diffusion2d_wsad/data/{dataset}'
ptpaths = sorted(glob(os.path.join(main_dir, 'images', '*.nii.gz')))
gtpaths = sorted(glob(os.path.join(main_dir, 'labels', '*.nii.gz')))
num_train, num_valid, num_test = 118, 8, 13 # pmbcl-bccv

ImageIDs_pos = [] 
ImageIDs_neg = []
for path in gtpaths:
    imageid = os.path.basename(path)[:-7]
    gt = read_image_array(path)
    if np.all(gt == 0):
        ImageIDs_neg.append(imageid)
    else:
        ImageIDs_pos.append(imageid)

# %%
random.seed(42)
test_ids = random.sample(ImageIDs_pos, num_test)
trainvalid_ids = list(set(ImageIDs_neg) | set(ImageIDs_pos) - set(test_ids)) 
valid_ids = random.sample(trainvalid_ids, num_valid)
train_ids = list(set(trainvalid_ids) - set(valid_ids)) 
# %%

diag = 'PMBCL'
train_ids, train_datasets, train_tracers, train_diagnosis, train_splits = get_datainfo_from_imageids(train_ids, dataset, diag, split='TRAIN')
valid_ids, valid_datasets, valid_tracers, valid_diagnosis, valid_splits = get_datainfo_from_imageids(valid_ids, dataset, diag, split='VALID')
test_ids, test_datasets, test_tracers, test_diagnosis, test_splits = get_datainfo_from_imageids(test_ids, dataset, diag, split='TEST')

df_pmbclbccv = pd.DataFrame(
    {
        'PatientID': train_ids + valid_ids + test_ids,
        'Dataset': train_datasets + valid_datasets + test_datasets,
        'Tracer': train_tracers + valid_tracers + test_tracers,
        'Diagnosis': train_diagnosis + valid_diagnosis + test_diagnosis,
        'Split': train_splits + valid_splits + test_splits,
    }
)
df_pmbclbccv.to_csv(f'{dataset}_info.csv', index=False)
# %%
# %%
#################################################################################################
######################################       DLBCL-SMHS        ##################################
#################################################################################################
dataset = 'dlbcl-smhs'
main_dir = f'/data/blobfuse/default/diffusion2d_wsad/data/{dataset}'
ptpaths = sorted(glob(os.path.join(main_dir, 'images', '*.nii.gz')))
gtpaths = sorted(glob(os.path.join(main_dir, 'labels', '*.nii.gz')))
num_train, num_valid, num_test = 187, 13, 20 # dlbcl-smhs

ImageIDs_pos = [] 
ImageIDs_neg = []
for path in gtpaths:
    imageid = os.path.basename(path)[:-7]
    gt = read_image_array(path)
    if np.all(gt == 0):
        ImageIDs_neg.append(imageid)
    else:
        ImageIDs_pos.append(imageid)

# %%
random.seed(42)
test_ids = random.sample(ImageIDs_pos, num_test)
trainvalid_ids = list(set(ImageIDs_neg) | set(ImageIDs_pos) - set(test_ids)) 
valid_ids = random.sample(trainvalid_ids, num_valid)
train_ids = list(set(trainvalid_ids) - set(valid_ids)) 
# %%
diag = 'DLBCL'
train_ids, train_datasets, train_tracers, train_diagnosis, train_splits = get_datainfo_from_imageids(train_ids, dataset, diag, split='TRAIN')
valid_ids, valid_datasets, valid_tracers, valid_diagnosis, valid_splits = get_datainfo_from_imageids(valid_ids, dataset, diag, split='VALID')
test_ids, test_datasets, test_tracers, test_diagnosis, test_splits = get_datainfo_from_imageids(test_ids, dataset, diag, split='TEST')

df_dlbclsmhs = pd.DataFrame(
    {
        'PatientID': train_ids + valid_ids + test_ids,
        'Dataset': train_datasets + valid_datasets + test_datasets,
        'Tracer': train_tracers + valid_tracers + test_tracers,
        'Diagnosis': train_diagnosis + valid_diagnosis + test_diagnosis,
        'Split': train_splits + valid_splits + test_splits,
    }
)
df_dlbclsmhs.to_csv(f'{dataset}_info.csv', index=False)
# %%
# %%
#################################################################################################
######################################       DLBCL-SMHS        ##################################
#################################################################################################
dataset = 'sts'
main_dir = f'/data/blobfuse/default/diffusion2d_wsad/data/{dataset}'
ptpaths = sorted(glob(os.path.join(main_dir, 'images', '*.nii.gz')))
gtpaths = sorted(glob(os.path.join(main_dir, 'labels', '*.nii.gz')))
num_train, num_valid, num_test = 0, 0, len(gtpaths) # sts

ImageIDs_pos = [] 
ImageIDs_neg = []
for path in gtpaths:
    imageid = os.path.basename(path)[:-7]
    gt = read_image_array(path)
    if np.all(gt == 0):
        ImageIDs_neg.append(imageid)
    else:
        ImageIDs_pos.append(imageid)

# %%
# All images in STS are positive
random.seed(42)
test_ids = random.sample(ImageIDs_pos, num_test)
trainvalid_ids = list(set(ImageIDs_neg) | set(ImageIDs_pos) - set(test_ids)) 
valid_ids = random.sample(trainvalid_ids, num_valid)
train_ids = list(set(trainvalid_ids) - set(valid_ids)) 
# %%
diag = 'STS'
train_ids, train_datasets, train_tracers, train_diagnosis, train_splits = get_datainfo_from_imageids(train_ids, dataset, diag, split='TRAIN')
valid_ids, valid_datasets, valid_tracers, valid_diagnosis, valid_splits = get_datainfo_from_imageids(valid_ids, dataset, diag, split='VALID')
test_ids, test_datasets, test_tracers, test_diagnosis, test_splits = get_datainfo_from_imageids(test_ids, dataset, diag, split='TEST')

df_sts = pd.DataFrame(
    {
        'PatientID': train_ids + valid_ids + test_ids,
        'Dataset': train_datasets + valid_datasets + test_datasets,
        'Tracer': train_tracers + valid_tracers + test_tracers,
        'Diagnosis': train_diagnosis + valid_diagnosis + test_diagnosis,
        'Split': train_splits + valid_splits + test_splits,
    }
)
df_sts.to_csv(f'{dataset}_info.csv', index=False)
# %%
# combine all datasets dataframe/datainfo

datasets = ['autopet', 'hecktor', 'dlbcl-bccv', 'pmbcl-bccv', 'dlbcl-smhs', 'sts']
infodir = '/home/jhubadmin/Projects/diffusion2d_wsad/data_analysis/data3D_split'
dfs = [pd.read_csv(os.path.join(infodir, f'{d}_info.csv')) for d in datasets]
# %%
df_all = pd.concat(dfs, axis=0)
df_all.to_csv('data3D_info.csv', index=False)
# %%
