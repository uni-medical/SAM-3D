from torch.utils.data import Dataset
import torchio as tio
import torch
import numpy as np
import os
import torch


class NIIDataset_Union_ALL(Dataset): 
    def __init__(self, paths, data_type='Tr', image_size=128, transform=None):
        self.paths = paths
        self.data_type = data_type

        self._set_file_paths(self.paths)
        self.image_size = image_size
        self.transform = transform
    
    def __len__(self):
        return len(self.label_paths)

    def __getitem__(self, index):

        subject = tio.Subject(
            image = tio.ScalarImage(self.image_paths[index]),
            label = tio.LabelMap(self.label_paths[index]),
        )
        if self.transform:
            subject = self.transform(subject)
        
        if subject.label.data.sum() < 1000:
            return self.__getitem__(np.random.randint(self.__len__()))

        return subject.image.data.clone().detach(), subject.label.data.clone().detach() # , self.image_paths[index]
    
    def _set_file_paths(self, paths):
        self.image_paths = []
        self.label_paths = []

        for path in paths:
            d = os.path.join(path, f'labels{self.data_type}')
            if os.path.exists(d):
                for name in os.listdir(d):
                    base = os.path.basename(name).split('.nii.gz')[0]
                    self.image_paths.append(f"{path}/images{self.data_type}/{base}.nii.gz")
                    self.label_paths.append(f"{path}/labels{self.data_type}/{base}.nii.gz")
