#%%
import os 
import pandas as pd 
import numpy as np 
from glob import glob 
from monai import transforms
import SimpleITK as sitk 
import matplotlib.pyplot as plt 
import pickle
from joblib import Parallel, delayed
import time 
#%%
def read_image_array_and_spacing(path):
    image = sitk.ReadImage(path)
    spacing = image.GetSpacing()
    array = sitk.GetArrayFromImage(image).T
    return array, spacing

def pad_zeros_at_front(num, N):
    return  str(num).zfill(N)

def save_and_fetch_2d_filepaths(datadir, patientid, savedir):
    ptpath = os.path.join(datadir, 'images', f'{patientid}_0001.nii.gz')
    gtpath = os.path.join(datadir, 'labels', f'{patientid}.nii.gz')
    pt, spacing = read_image_array_and_spacing(ptpath)
    gt, spacing = read_image_array_and_spacing(gtpath)
    slice_spacing = (spacing[0], spacing[1])
    ptpaths_list, gtpaths_list, labels_list = [], [], []
    for i in range(96):
        pt2d = pt[:, :, i].T
        gt2d = gt[:, :, i].T
        if np.all(gt2d == 0):
            labels_list.append(1)
        else:
            labels_list.append(2)
        pt2d_img = sitk.GetImageFromArray(pt2d)
        gt2d_img = sitk.GetImageFromArray(gt2d)
        pt2d_img.SetSpacing(slice_spacing)
        gt2d_img.SetSpacing(slice_spacing)
        pt2d_fpath = os.path.join(savedir, f'{patientid}_{pad_zeros_at_front(i, 3)}_pt.nii.gz')
        gt2d_fpath = os.path.join(savedir, f'{patientid}_{pad_zeros_at_front(i, 3)}_gt.nii.gz')
        ptpaths_list.append(pt2d_fpath)
        gtpaths_list.append(gt2d_fpath)
        sitk.WriteImage(pt2d_img, pt2d_fpath)
        sitk.WriteImage(gt2d_img, gt2d_fpath)
    return ptpaths_list, gtpaths_list, labels_list

def process_one_patient(row, datadir_main, savedir, idx):
    patientid, dataset = row['PatientID'], row['Dataset']
    datadir = os.path.join(datadir_main, f'{dataset}')
    ptpath_list, gtpath_list, label_list = save_and_fetch_2d_filepaths(datadir, patientid, savedir)
    
    patientid_list = [patientid] * 96
    dataset_list = [dataset] * 96
    tracer_list = [row['Tracer']] * 96
    diagnosis_list = [row['Diagnosis']] * 96
    split_list = [row['Split']] * 96
    sliceid_list = np.arange(96)
    print(idx)
    return patientid_list, dataset_list, tracer_list, diagnosis_list, split_list, sliceid_list, label_list, ptpath_list, gtpath_list

#%%
datadir_main = '/data/blobfuse/default/diffusion2d_wsad/data'
savedir = '/data/blobfuse/default/diffusion2d_wsad/data/data2D_axial'
os.makedirs(savedir, exist_ok=True)
splits = ['TRAIN', 'VALID', 'TEST']
for split in splits:
    df3D_all = pd.read_csv('/home/jhubadmin/Projects/diffusion2d_wsad/data_analysis/data3D_split/data3D_info.csv')
    df3D_split = df3D_all[df3D_all['Split'] == split]
    df3D_split.reset_index(inplace=True, drop=True)
    
    print(f'Running for split = {split}')
    print(f'{len(df3D_split)} 3D images will be processed to 2D images!')
    
    start = time.time()
    results = Parallel(n_jobs=-1)(delayed(process_one_patient)(row, datadir_main, savedir, idx) for idx, row in df3D_split.iterrows())

    # Combine Results
    PATIENTID, DATASET, TRACER, DIAGNOSIS, SPLIT, SLICEID, LABEL, PTPATH, GTPATH = [], [], [], [], [], [], [], [], []
    for result in results:
        PATIENTID.extend(result[0])
        DATASET.extend(result[1])
        TRACER.extend(result[2])
        DIAGNOSIS.extend(result[3])
        SPLIT.extend(result[4])
        SLICEID.extend(result[5])
        LABEL.extend(result[6])
        PTPATH.extend(result[7])
        GTPATH.extend(result[8])

    # Save Combined Data to CSV
    df2d_all = pd.DataFrame(
        {
            'PatientID': PATIENTID,
            'Dataset': DATASET,
            'Tracer': TRACER,
            'Diagnosis': DIAGNOSIS,
            'Split': SPLIT,
            'SliceID': SLICEID,
            'Label': LABEL,
            'PTPATH': PTPATH,
            'GTPATH': GTPATH
        }
    )
    df2d_fpath = os.path.join('/home/jhubadmin/Projects/diffusion2d_wsad/data_analysis/data2D_split', f'{split}_data2D_info.csv')
    df2d_all.to_csv(df2d_fpath, index=False)
    elapsed = time.time() - start 
    print(f'Time taken: {elapsed/60} min')

