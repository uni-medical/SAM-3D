from data_loader_all import NIIDataset_Union_ALL
from all_data_paths_py import img_datas
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torchio as tio
import numpy as np


ref = tio.ScalarImage('/cpfs01/shared/gmai/medical_preprocessed/3d/semantic_seg/ct/MSD_Liver/imagesTr/liver_0.nii.gz')

img_datas = [
    # '/cpfs01/shared/gmai/medical_preprocessed/3d/semantic_seg/mr_unknown/COSMOS2022',
    '/cpfs01/shared/gmai/medical_preprocessed/3d/semantic_seg/ct/ISLES2018',
]

train_dataset = NIIDataset_Union_ALL(paths=img_datas, transform=tio.Compose([
    tio.ToCanonical(),
    tio.Resample(ref),
    tio.Resize((160,160,160)),
    tio.CropOrPad(mask_name='crop_mask', target_shape=(128,128,128)), # crop only object region
    tio.KeepLargestComponent(label_keys='crop_mask'),
    tio.RandomAffine(degrees=[-np.pi/8, np.pi/8], scales=[0.8, 1.25]),
    tio.RandomFlip(axes=(0, 1, 2)),
    
    # tio.RemapLabels({2:1, 3:1}),
]))


train_dataloader = DataLoader(
    dataset=train_dataset,
    sampler=None,
    batch_size=1, 
    shuffle=True,
    num_workers=32,
    pin_memory=True,
)


names = []
for x,y in tqdm(train_dataloader):
    # print(x.shape)
    # print(y.shape)
    if x is None and y is None :
        print('===============================')
        print('x,y is None')
        # print(name)
        print('===============================')
        # names.extend(name)
    if x.shape != y.shape:
        print('===============================')
        print(f'x.shape: {x.shape}')
        print(f'y.shape: {y.shape}')
        # print(name)
        # names.extend(name)
        print('===============================')


with open('error.py', 'w') as f:
    f.writelines('data = [\n')
    for i in sorted(names):
        f.writelines('\''+i+'\',\n')
    f.writelines(']\n')

