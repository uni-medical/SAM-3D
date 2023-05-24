import os
import numpy as np
import nibabel as nib
from all_data_paths_py import img_datas
from tqdm import tqdm

img_datas = [
# '/cpfs01/shared/gmai/medical_preprocessed/3d/semantic_seg/mr_unknown/Heart_Seg_MRI',
# '/cpfs01/shared/gmai/medical_preprocessed/3d/semantic_seg/mr_unknown/MMWHS',
# '/cpfs01/shared/gmai/medical_preprocessed/3d/semantic_seg/mr_unknown/MSD_Heart',
# '/cpfs01/shared/gmai/medical_preprocessed/3d/semantic_seg/mr_unknown/MSD_Hippocampus',
# '/cpfs01/shared/gmai/medical_preprocessed/3d/semantic_seg/mr_unknown/PROMISE12',
# '/cpfs01/shared/gmai/medical_preprocessed/3d/semantic_seg/mr_unknown/STACOM_SLAWT',
# '/cpfs01/shared/gmai/medical_preprocessed/3d/semantic_seg/mr_unknown/abdomen_tissue_fromSJTU',
# '/cpfs01/shared/gmai/medical_preprocessed/3d/semantic_seg/mr_unknown/cSeg-2022',
    '/cpfs01/shared/gmai/medical_preprocessed/3d/semantic_seg/ct/ISLES2018',
]

for pa in tqdm(img_datas):
    labels = os.listdir(os.path.join(pa, 'labelsTr'))
    for label in labels:
        label_path = os.path.join(pa, 'labelsTr', label)
        label_data = nib.load(label_path)
        label_d = label_data.get_fdata()
        if label_d.sum() == 0:
            print(label_path)
        else:
            print(label_d.sum())





