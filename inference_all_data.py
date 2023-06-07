import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import os
join = os.path.join
import torch
from segment_anything.build_sam3D import sam_model_registry3D
from segment_anything.utils.transforms3D import ResizeLongestSide3D
from tqdm import tqdm
import argparse
import traceback
from PIL import Image
import SimpleITK as sitk
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import SimpleITK as sitk
import torchio as tio
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument('-cp', '--checkpoint_path', type=str, default='./work_dir/SAM-ViT-B_multi_choose/sam_model_latest.pth')
parser.add_argument('-dt', '--data_type', type=str, default='Ts')
parser.add_argument('-pm', '--point_method', type=str, default='random')
parser.add_argument('--multi', action='store_true', default=False)
parser.add_argument('--union', action='store_true', default=False)
# parser.add_argument('-tdp', '--test_data_path', type=str, default='../data/AMOS2022_ct_pancreas')
parser.add_argument('--num_clicks', type=int, default=10)
parser.add_argument('-mt', '--model_type', type=str, default='vit_b_ori')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--img_size', type=int, default=128)
parser.add_argument('--save_name', type=str, default='out_dice.py')
args = parser.parse_args()

if torch.cuda.is_available():
    torch.cuda.init()


from utils import get_next_click3D_torch


click_methods = {
    'random': get_next_click3D_torch,
    'choose': None,
}

def finetune_model_predict3D(img3D, gt3D, sam_trans, sam_model_tune, device='cuda:0', click_method='random',num_clicks=5):
    # H, W = img_np.shape[:2]
    H, W, D = img3D.shape[:3]
    # resize_img = sam_trans.apply_image(img_np)

    # resize_img_tensor = torch.as_tensor(resize_img.transpose(2, 0, 1)).to('cuda:0')
    # input_image = sam_model_tune.preprocess(resize_img_tensor[None,:,:,:]) # (1, 3, 1024, 1024)
    # gt2D = torch.as_tensor(gt2D).to('cuda:0').reshape(1,1,256,256) # (B, 1, 1024, 1024)

    click_points = []
    click_labels = []

    pred_list = []

    iou_list = []
    dice_list = []
    prev_masks = torch.zeros_like(gt3D).to(gt3D.device)
    low_res_masks = F.interpolate(prev_masks.float(), size=(args.img_size//4,args.img_size//4,args.img_size//4))
    
    for num_click in range(num_clicks):

        with torch.no_grad():

            image_embedding = sam_model_tune.image_encoder(img3D.to(device)) # (1, 384, 16, 16, 16)

            batch_points, batch_labels = click_methods[click_method](prev_masks.to(device), gt3D.to(device))

            points_co = torch.cat(batch_points, dim=0).to(device)  # 得到batch的每个点的坐标
            points_la = torch.cat(batch_labels, dim=0).to(device)  # 得到batch的每个点的label

            click_points.append(points_co)
            click_labels.append(points_la)

            points_coords = torch.cat(click_points, dim=1).to(device)
            points_labels = torch.cat(click_labels, dim=1).to(device)

            if args.multi:
                points_input = points_coords
                labels_input = points_labels
            else:
                points_input = points_co
                labels_input = points_la


            sparse_embeddings, dense_embeddings = sam_model_tune.prompt_encoder(
                points=[points_input, labels_input],
                boxes=None,
                masks=low_res_masks.to(device),
            )
            low_res_masks, _ = sam_model_tune.mask_decoder(
                image_embeddings=image_embedding.to(device), # (B, 384, 64, 64, 64)
                image_pe=sam_model_tune.prompt_encoder.get_dense_pe(), # (1, 384, 64, 64, 64)
                sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 384)
                dense_prompt_embeddings=dense_embeddings, # (B, 384, 64, 64, 64)
                multimask_output=False,
                )
            prev_masks = F.interpolate(low_res_masks, size=gt3D.shape[-3:], mode='trilinear', align_corners=False)

            medsam_seg_prob = torch.sigmoid(prev_masks)  # (B, 1, 64, 64, 64)
            # convert soft mask to hard mask
            medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
            medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)
            pred_list.append(medsam_seg)
            # low_res_masks = medsam_seg

            def compute_iou(pred_mask, gt_semantic_seg):
                
                in_mask = np.logical_and(gt_semantic_seg, pred_mask)
                out_mask = np.logical_or(gt_semantic_seg, pred_mask)
                iou = np.sum(in_mask) / np.sum(out_mask)
                return iou
            
            def compute_dice(mask_gt, mask_pred):
                """Compute soerensen-dice coefficient.
                Returns:
                the dice coeffcient as float. If both masks are empty, the result is NaN
                """
                volume_sum = mask_gt.sum() + mask_pred.sum()
                if volume_sum == 0:
                    return np.NaN
                volume_intersect = (mask_gt & mask_pred).sum()
                return 2*volume_intersect / volume_sum


            iou_list.append(round(compute_iou(medsam_seg, gt3D[0][0].detach().cpu().numpy()), 4))

            dice_list.append(round(compute_dice(gt3D[0][0].detach().cpu().numpy().astype(np.uint8), medsam_seg), 4))

    # return medsam_seg
    return pred_list, points_co.cpu().numpy(), points_la.cpu().numpy(), iou_list, dice_list


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

        return subject.image.data.clone().detach(), subject.label.data.clone().detach(), self.image_paths[index]
    
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

from data_paths_no_rescale import img_datas

# train_dataset = NIIDataset_Val(path='./dataset/MSD01_BrainTumor_flair', transform=tio.Compose(
train_dataset = NIIDataset_Union_ALL(
    # paths=[args.test_data_path,], 
    paths=img_datas,
    data_type=args.data_type, 
    transform=tio.Compose([
        # tio.ToCanonical(),
        # tio.Resample(1),
        # tio.Resize((160,160,160)),
        # tio.CropOrPad(mask_name='crop_mask', target_shape=(256,256,256)), # crop only object region
        # tio.CropOrPad(mask_name='crop_mask', target_shape=(args.img_size,args.img_size,args.img_size)),
        # tio.RandomAffine(degrees=[-np.pi/8, np.pi/8], scales=[0.8, 1.25]),
        tio.KeepLargestComponent(),
        # tio.RandomFlip(axes=(0, 1, 2)),
        # tio.RemapLabels({2:1, 3:1}),
    ]
))

train_dataloader = DataLoader(
    dataset=train_dataset,
    sampler=None,
    batch_size=1, 
    shuffle=True
)

checkpoint_path = args.checkpoint_path

device = args.device
# sam_model_tune = sam_model_registry3D[args.model_type](checkpoint=checkpoint_path).to(device)
if args.union:
    sam_model_tune = sam_model_registry3D[args.model_type](checkpoint=None).to(device)
    if checkpoint_path is not None:
        # with open(checkpoint_path, "rb") as f:
        model_dict = torch.load(checkpoint_path, map_location=device)
        state_dict = model_dict['model_state_dict']
        sam_model_tune.load_state_dict(state_dict)
else:
    sam_model_tune = sam_model_registry3D[args.model_type](checkpoint=checkpoint_path).to(device)
sam_trans = ResizeLongestSide3D(sam_model_tune.image_encoder.img_size)

all_iou_list = []
all_dice_list = []  

out_dice = dict()
for image3D, gt3D, img_name in tqdm(train_dataloader):

    norm_transform = tio.ZNormalization(masking_method=lambda x: x > 0)
    image3D = norm_transform(image3D.squeeze(dim=1)) # (N, C, W, H, D)
    image3D = image3D.unsqueeze(dim=1)
    
    # seg_mask = np.zeros_like(gt3D)

    seg_mask_list, points, labels, iou_list, dice_list = finetune_model_predict3D(image3D, gt3D, sam_trans, sam_model_tune, device=device, click_method=args.point_method, num_clicks=args.num_clicks)
    
    # print(points)
    # print(labels)
    # print(iou_list)
    per_iou = max(iou_list)
    # print(per_iou)
    all_iou_list.append(per_iou)
    all_dice_list.append(max(dice_list))
    print(dice_list)
    out_dice[img_name] = max(dice_list)

# print(all_iou_list)
print('mean IoU : ', sum(all_iou_list)/len(all_iou_list))
print('mean Dice: ', sum(all_dice_list)/len(all_dice_list))

with open(args.save_name, 'w') as f:
    f.writelines(f'mean dice: \t{np.mean(all_dice_list)}\n')

    for k, v in out_dice.items():
        f.writelines(f'\'{k}\': {v},\n')