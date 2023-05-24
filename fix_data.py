import nibabel as nib
import numpy as np
import os
from tqdm import tqdm


ori_path = '/cpfs01/shared/gmai/medical_preprocessed/3d/semantic_seg/mr_unknown/COSMOS2022'

for i in tqdm(os.listdir(os.path.join(ori_path, 'imagesTr'))):
    data = nib.load(os.path.join(ori_path, 'imagesTr', i))
    data_d = data.get_fdata()
    new_data = nib.Nifti1Image(data_d, data.affine)
    nib.save(new_data, os.path.join(ori_path, 'imagesTr', i))







