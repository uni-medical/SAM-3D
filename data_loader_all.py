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

        if index >= self.__len__():
            return self.__getitem__(np.random.randint(self.__len__()))
        
        subject = tio.Subject(
            image = tio.ScalarImage(self.image_paths[index]),
            label = tio.LabelMap(self.label_paths[index]),
        )
        subject.label.set_data(subject.label.data.type(torch.uint8))
        if subject.label.data.sum() == 0:
            self.image_paths.remove(self.image_paths[index])
            self.label_paths.remove(self.label_paths[index])
            return self.__getitem__(np.random.randint(self.__len__()))
        
        # subject.add_image(tio.LabelMap(tensor=(subject.label.data.clone() > 0),
        #                                affine=subject.label.affine),
        #                     image_name="crop_mask")
        
        ############################ get two class ############################
        return_label = subject.label.data.clone().detach()
        # 给定的 tensor
        labels_num = torch.unique(return_label)
        # 获取大于0的索引
        nonzero_indices = torch.nonzero(labels_num > 0)
        # 从索引中随机选择一个
        random_index = torch.randint(0, nonzero_indices.size(0), (1,))
        selected_value = labels_num[nonzero_indices[random_index]]
        # 输出结果
        return_label_num = selected_value.item()
        return_label[return_label != return_label_num] = 0
        return_label[return_label == return_label_num] = 1
        
        ############################ get two class ############################
        subject.add_image(tio.LabelMap(tensor=return_label,
                                       affine=subject.label.affine),
                            image_name="crop_mask")

        # if subject.crop_mask.data.sum() == 0:
        if return_label.sum() < 100:
            # print('===========================')
            # print(self.label_paths[index])
            # print('===========================')
            return self.__getitem__(np.random.randint(self.__len__()))
        
        if self.transform:
            try:
                subject = self.transform(subject)
            except Exception as e:
                # print('===============================')
                # print(e)
                # print(subject.image.data.shape)
                # print(subject.label.data.shape)
                # print(self.image_paths[index])
                # print('===============================')
                return self.__getitem__(np.random.randint(self.__len__()))
        # return torch.tensor(img_embed).float(), torch.tensor(gt2D[None, :,:]).long()  # , torch.tensor(bboxes).float()
        # return self.image_paths[index], subject.image.data.float(), subject.label.data[0, ...]
        final_return_image = subject.image.data.clone().detach()
        final_return_label = subject.crop_mask.data.clone().detach()
        
        if final_return_label.sum() < 100:
            return self.__getitem__(np.random.randint(self.__len__()))
        
        return final_return_image, final_return_label # , self.image_paths[index]
    
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
                    # break
            # else:
            #     print(f"{d} not exists!")
