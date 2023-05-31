import numpy as np
from multiprocessing import Process, Manager, Lock
import random 
import matplotlib.pyplot as plt
import cv2
import os
import warnings
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F


def load_sam_2d_weight(model_2d_path, model_3d):
    model_2d = torch.load(model_2d_path)
    for k in model_2d.keys():
        if 'image_encoder.pos_embed' in k:
            # 随机初始化权重
            model_3d.state_dict()[k].data.normal_(mean=0.0, std=0.02)
        elif 'image_encoder.patch_embed.proj.weight' in k:
            # 随机初始化权重
            # model_3d.state_dict()[k].data.normal_(mean=0.0, std=0.02)
            model_3d.state_dict()[k] = (model_2d['image_encoder.patch_embed.proj.weight'].sum(dim=1)/3).unsqueeze(-1).repeat(1,1,1,1,16)/16
        elif 'image_encoder.blocks.2.attn.rel_pos_h' in k:
            # 随机初始化权重
            model_3d.state_dict()[k].data.normal_(mean=0.0, std=0.02)
        elif 'image_encoder.blocks.2.attn.rel_pos_w' in k:
            model_3d.state_dict()[k].data.normal_(mean=0.0, std=0.02)
        elif 'image_encoder.blocks.5.attn.rel_pos_h' in k:
            model_3d.state_dict()[k].data.normal_(mean=0.0, std=0.02)
        elif 'image_encoder.blocks.5.attn.rel_pos_w' in k:
            model_3d.state_dict()[k].data.normal_(mean=0.0, std=0.02)
        elif 'image_encoder.blocks.8.attn.rel_pos_h' in k:
            model_3d.state_dict()[k].data.normal_(mean=0.0, std=0.02)
        elif 'image_encoder.blocks.8.attn.rel_pos_w' in k:
            model_3d.state_dict()[k].data.normal_(mean=0.0, std=0.02)
        elif 'image_encoder.blocks.11.attn.rel_pos_h' in k:
            model_3d.state_dict()[k].data.normal_(mean=0.0, std=0.02)
        elif 'image_encoder.blocks.11.attn.rel_pos_w' in k:
            model_3d.state_dict()[k].data.normal_(mean=0.0, std=0.02)
        elif 'image_encoder.neck.0.weight' in k:
            model_3d.state_dict()[k] = model_2d['image_encoder.neck.0.weight'].unsqueeze(-1)
        elif 'image_encoder.neck.2.weight' in k:
            model_3d.state_dict()[k] = model_2d['image_encoder.neck.2.weight'].unsqueeze(-1).repeat(1,1,1,1,3)/3
        elif 'prompt_encoder.pe_layer.positional_encoding_gaussian_matrix' in k:
            # 随机初始化权重
            model_3d.state_dict()[k].data.normal_(mean=0.0, std=0.02)
        elif 'prompt_encoder.mask_downscaling.0.weight' in k:
            model_3d.state_dict()[k] = model_2d['prompt_encoder.mask_downscaling.0.weight'].unsqueeze(-1).repeat(1,1,1,1,2)/2
        elif 'prompt_encoder.mask_downscaling.3.weight' in k:
            model_3d.state_dict()[k] = model_2d['prompt_encoder.mask_downscaling.3.weight'].unsqueeze(-1).repeat(1,1,1,1,2)/2
        elif 'prompt_encoder.mask_downscaling.6.weight' in k:
            model_3d.state_dict()[k] = model_2d['prompt_encoder.mask_downscaling.6.weight'].unsqueeze(-1).repeat(1,1,1,1,2)/2
        elif 'mask_decoder.output_upscaling.0.weight' in k:
            model_3d.state_dict()[k] = model_2d['mask_decoder.output_upscaling.0.weight'].unsqueeze(-1).repeat(1,1,1,1,2)/2
        elif 'mask_decoder.output_upscaling.3.weight' in k:
            model_3d.state_dict()[k] = model_2d['mask_decoder.output_upscaling.3.weight'].unsqueeze(-1).repeat(1,1,1,1,2)/2
        else:
            model_3d.state_dict()[k] = model_2d[k]
    return model_3d



def get_next_click3D_torch(prev_seg, gt_semantic_seg):

    def compute_dice(mask_pred, mask_gt):
        mask_threshold = 0.5

        mask_pred = (mask_pred > mask_threshold)
        # mask_gt = mask_gt.astype(bool)
        mask_gt = (mask_gt > 0)
        
        volume_sum = mask_gt.sum() + mask_pred.sum()
        if volume_sum == 0:
            return np.NaN
        volume_intersect = (mask_gt & mask_pred).sum()
        return 2*volume_intersect / volume_sum

    mask_threshold = 0.5

    batch_points = []
    batch_labels = []
    dice_list = []

    pred_masks = (prev_seg > mask_threshold)
    true_masks = (gt_semantic_seg > 0)
    fn_masks = torch.logical_and(true_masks, torch.logical_not(pred_masks))
    fp_masks = torch.logical_and(torch.logical_not(true_masks), pred_masks)


    for i in range(gt_semantic_seg.shape[0]):

        fn_points = torch.argwhere(fn_masks[i])
        fp_points = torch.argwhere(fp_masks[i])
        if len(fn_points) > 0 and len(fp_points) > 0:
            if np.random.random() > 0.5:
                point = fn_points[np.random.randint(len(fn_points))]
                is_positive = True
            else:
                point = fp_points[np.random.randint(len(fp_points))]
                is_positive = False
        elif len(fn_points) > 0:
            point = fn_points[np.random.randint(len(fn_points))]
            is_positive = True
        elif len(fp_points) > 0:
            point = fp_points[np.random.randint(len(fp_points))]
            is_positive = False
        # bp = torch.tensor(point[1:]).reshape(1,1,3) 
        bp = point[1:].clone().detach().reshape(1,1,3) 
        bl = torch.tensor([int(is_positive),]).reshape(1,1)
        batch_points.append(bp)
        batch_labels.append(bl)
        dice_list.append(compute_dice(pred_masks[i], true_masks[i]))

    return batch_points, batch_labels, (sum(dice_list)/len(dice_list)).item()    


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251/255, 252/255, 30/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_point(point, label, ax):
    if label == 0:
        ax.add_patch(plt.Circle((point[1], point[0]), 1, color='red'))
    else:
        ax.add_patch(plt.Circle((point[1], point[0]), 1, color='green'))
    # plt.scatter(point[0], point[1], label=label)


