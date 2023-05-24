# set up environment
import numpy as np
import random 
import matplotlib.pyplot as plt
import cv2
import os
join = os.path.join
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import monai
import SimpleITK as sitk
import torchio as tio
from torch.utils.data.distributed import DistributedSampler
# from .segment_anything import SamPredictor, sam_model_registry3D
from segment_anything.build_sam3D import sam_model_registry3D
# from segment_anything.utils.transforms import ResizeLongestSide
import argparse
from torch.cuda import amp
import torch.multiprocessing as mp
from multiprocessing import Manager
from multiprocessing.managers import BaseManager

from torch.nn.parallel import DistributedDataParallel as DDP

ref = tio.ScalarImage('/cpfs01/shared/gmai/medical_preprocessed/3d/semantic_seg/ct/MSD_Liver/imagesTr/liver_0.nii.gz')

# set seeds
torch.manual_seed(2023)
np.random.seed(2023)


# %% set up parser
parser = argparse.ArgumentParser()
parser.add_argument('--task_name', type=str, default='Union_SAM-ViT-B_multi_random_using2dweight_multigpu')
parser.add_argument('--click_type', type=str, default='random')
parser.add_argument('--multi_click', action='store_true', default=False)
parser.add_argument('--model_type', type=str, default='vit_b_ori')
parser.add_argument('--checkpoint', type=str, default='./work_dir/SAM/sam_vit_b.pth')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--work_dir', type=str, default='./work_dir')
# parser.add_argument('--data_dir', type=str, default='/cpfs01/user/guosizheng/SAM3D/dataset/MSD01_BrainTumor_flair')
# parser.add_argument('--log_out_dir', type=str, default='./work_dir/MSD01_BrainTumor_flair')

# train
parser.add_argument('--num_workers', type=int, default=32)
parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0,1,2,3,4,5,6,7])
parser.add_argument('--multi_gpu', action='store_true', default=False)


# lr_scheduler
parser.add_argument('--lr_scheduler', type=str, default='multisteplr')
# parser.add_argument('--warmup_epochs', type=int, default=0)
# parser.add_argument('--warmup_factor', type=float, default=1e-6)
# parser.add_argument('--warmup_method', type=str, default='linear')
# parser.add_argument('--final_lr', type=float, default=1e-6)
# parser.add_argument('--power', type=float, default=1.0)
# parser.add_argument('--max_iters', type=int, default=1000)
# parser.add_argument('--min_lr', type=float, default=1e-6)
# parser.add_argument('--step_size', type=int, default=5)
# parser.add_argument('--gamma', type=float, default=0.5)
parser.add_argument('--step_size', type=list, default=[600, 900])
parser.add_argument('--gamma', type=float, default=0.1)


parser.add_argument('--num_epochs', type=int, default=1000)
parser.add_argument('--img_size', type=int, default=128)
parser.add_argument('--batch_size', type=int, default=12)
parser.add_argument('--lr', type=float, default=8e-4)
parser.add_argument('--weight_decay', type=float, default=0.1)

# parser.add_argument('-lwl', '--layer_wize_lr', action='store_true', default=False)
# parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--amp', action='store_true', default=False)

parser.add_argument('--port', type=int, default=12361)

args = parser.parse_args()

###################################### Logging ######################################
import datetime
import logging
logger = logging.getLogger(__name__)

LOG_OUT_DIR = join(args.work_dir, args.task_name)

cur_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
logging.basicConfig(
    format='[%(asctime)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.INFO,
    filemode='w',
    filename=os.path.join(LOG_OUT_DIR, f'output_{cur_time}.log'))
###################################### Logging ######################################

# args.multi_gpu = True

###################################### Click type ######################################
from utils import load_sam_2d_weight, get_next_click3D_torch
click_methods = {
    'random': get_next_click3D_torch,
    'choose': None,
}
###################################### Click type ######################################


# set up model for fine-tuning 
device = args.device
MODEL_SAVE_PATH = join(args.work_dir, args.task_name)
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

def build_model(args):

    sam_model = sam_model_registry3D[args.model_type](checkpoint=None).to(device)
    # 复用2D的权重
    
    sam_model = load_sam_2d_weight(args.checkpoint, sam_model)
    if args.multi_gpu:
        sam_model = DDP(sam_model, device_ids=[args.rank], output_device=args.rank)
    # sam_model.train()
    return sam_model

################################################## Data ##################################################
# from dataloader_part import NIIDataset_Union, img_datas
from data_loader_all import NIIDataset_Union_ALL
from all_data_paths_py import img_datas

def get_dataloaders(args):
    train_dataset = NIIDataset_Union_ALL(paths=img_datas, transform=tio.Compose([
        tio.ToCanonical(),
        tio.Resample(ref),
        tio.Resize((160,160,160)),
        tio.CropOrPad(mask_name='crop_mask', target_shape=(args.img_size,args.img_size,args.img_size)), # crop only object region
        tio.KeepLargestComponent(label_keys='crop_mask'),
        tio.RandomAffine(degrees=[-np.pi/8, np.pi/8], scales=[0.8, 1.25]),
        tio.RandomFlip(axes=(0, 1, 2)),
        # tio.RemapLabels({2:1, 3:1}),
    ]))

    if args.multi_gpu:
        train_sampler = DistributedSampler(train_dataset)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    train_dataloader = DataLoader(
        dataset=train_dataset,
        sampler=train_sampler,
        batch_size=args.batch_size, 
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    return train_dataloader
################################################## Data ##################################################


########################################## Trainer ##########################################

class BaseTrainer:
    def __init__(self, model, dataloaders, args):

        self.model = model
        self.dataloaders = dataloaders
        self.args = args
        self.best_loss = np.inf
        self.best_dice = 0.0
        self.step_best_loss = np.inf
        self.step_best_dice = 0.0
        self.losses = []
        self.dices = []
        self.ious = []
        self.set_loss_fn()
        self.set_optimizer()
        self.set_lr_scheduler()
        self.init_checkpoint(join(self.args.work_dir, self.args.task_name, 'sam_model_latest.pth'))
        self.norm_transform = tio.ZNormalization(masking_method=lambda x: x > 0)
        
    def set_loss_fn(self):
        self.seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
        # self.nfl_loss = NormalizedFocalLoss()
    
    def set_optimizer(self):
        # self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        if self.args.multi_gpu:
            sam_model = self.model.module
        else:
            sam_model = self.model

        self.optimizer = torch.optim.AdamW([
            {'params': sam_model.image_encoder.parameters()},# , 'lr': self.args.lr * 1.0},
            {'params': sam_model.prompt_encoder.parameters()},
            {'params': sam_model.mask_decoder.parameters()},
        ], lr=self.args.lr, betas=(0.9,0.999), weight_decay=self.args.weight_decay)

    def set_lr_scheduler(self):
        if self.args.lr_scheduler == "multisteplr":
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                                self.args.step_size,
                                                                self.args.gamma)
        elif self.args.lr_scheduler == "steplr":
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                                self.args.step_size[0],
                                                                self.args.gamma)
        else:
            self.lr_scheduler = None

    def init_checkpoint(self, ckp_path):
        last_ckpt = None
        if os.path.exists(ckp_path):
            if self.args.multi_gpu:
                dist.barrier()
                last_ckpt = torch.load(ckp_path, map_location=self.args.device)
            else:
                last_ckpt = torch.load(ckp_path, map_location=self.args.device)
        
        if last_ckpt:
            if self.args.multi_gpu:
                self.model.module.load_state_dict(last_ckpt['model_state_dict'])
            else:
                self.model.load_state_dict(last_ckpt['model_state_dict'])
            # self.start_epoch = last_ckpt['epoch']
            self.start_epoch = 0
            # self.optimizer.load_state_dict(last_ckpt['optimizer_state_dict'])
            # self.lr_scheduler.load_state_dict(last_ckpt['lr_scheduler_state_dict'])
            # self.losses = last_ckpt['losses']
            # self.dices = last_ckpt['dices']
            # self.best_loss = last_ckpt['best_loss']
            # self.best_dice = last_ckpt['best_dice']
            print(f"Loaded checkpoint from {ckp_path} (epoch {self.start_epoch})")
        else:
            self.start_epoch = 0
            print(f"No checkpoint found at {ckp_path}, start training from scratch")

    def save_checkpoint(self, epoch, state_dict, describe="last"):
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": state_dict,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
            "losses": self.losses,
            "dices": self.dices,
            "best_loss": self.best_loss,
            "best_dice": self.best_dice,
            "args": self.args,
            "used_datas": img_datas,
        }, join(MODEL_SAVE_PATH, f"sam_model_{describe}.pth"))

    def decode(self, sam_model, image_embedding, low_res_masks, gt3D, points=None):
        sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
            points=points,
            boxes=None,
            masks=low_res_masks,
        )
        low_res_masks, iou_predictions = sam_model.mask_decoder(
            image_embeddings=image_embedding.to(device), # (B, 256, 64, 64)
            image_pe=sam_model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            multimask_output=False,
        )
        prev_masks = F.interpolate(low_res_masks, size=gt3D.shape[-3:], mode='trilinear', align_corners=False)
        cur_loss = self.seg_loss(prev_masks, gt3D)
        return low_res_masks, prev_masks, cur_loss
    
    def train_epoch(self, epoch, num_clicks):
        epoch_loss = 0
        epoch_iou = 0
        epoch_dice = 0
        self.model.train()
        if self.args.multi_gpu:
            sam_model = self.model.module
        else:
            sam_model = self.model
        # Just train on the first 20 examples
        for step, (image3D, gt3D) in enumerate(tqdm(self.dataloaders)):

            image3D = self.norm_transform(image3D.squeeze(dim=1)) # (N, C, W, H, D)
            image3D = image3D.unsqueeze(dim=1)
            
            image3D = image3D.to(device)
            gt3D = gt3D.to(device).type(torch.long)
            # gt3D
            
            if self.args.amp:
                with amp.autocast():
                    image_embedding = sam_model.image_encoder(image3D)
            else:
                image_embedding = sam_model.image_encoder(image3D)

            # click points存储在点击数量上的点坐标（例如每个batch有4个，则其中每个数据就是(4,1,3)的维度）
            click_points = []
            click_labels = []

            pred_list = []

            loss = 0

            # random_insert = np.random.randint(1, 10)
            random_insert = num_clicks
            # do not compute gradients for image encoder and prompt encoder
            for num_click in range(11):
                cur_loss = 0
                if num_click == 0:
                    prev_masks = torch.zeros_like(gt3D).to(gt3D.device)
                    low_res_masks = F.interpolate(prev_masks.float(), size=(args.img_size//4,args.img_size//4,args.img_size//4))
                elif num_click == random_insert or num_click == 10:
                    if self.args.amp:
                        with amp.autocast():
                            low_res_masks, prev_masks, cur_loss = self.decode(sam_model, image_embedding, low_res_masks, gt3D, points=None)
                    else:
                        low_res_masks, prev_masks, cur_loss = self.decode(sam_model, image_embedding, low_res_masks, gt3D, points=None)
                    loss += cur_loss
                    continue
                else:
                    low_res_masks = low_res_masks
                    prev_masks = prev_masks
                # batch_points存储在batch维度上的点坐标（其中每个是(1,1,3)的维度）
                batch_points, batch_labels, dice = click_methods[args.click_type](prev_masks, gt3D)

                points_co = torch.cat(batch_points, dim=0).to(device)
                points_la = torch.cat(batch_labels, dim=0).to(device)

                click_points.append(points_co)
                click_labels.append(points_la)

                points_multi = torch.cat(click_points, dim=1).to(device)
                labels_multi = torch.cat(click_labels, dim=1).to(device)

                if self.args.multi_click:
                    points_input = points_multi
                    labels_input = labels_multi
                else:
                    points_input = points_co
                    labels_input = points_la

                if self.args.amp:
                    with amp.autocast():
                        low_res_masks, prev_masks, cur_loss = self.decode(sam_model, image_embedding, low_res_masks, gt3D, points=[points_input, labels_input])
                else:
                    low_res_masks, prev_masks, cur_loss = self.decode(sam_model, image_embedding, low_res_masks, gt3D, points=[points_input, labels_input])
                
                loss += cur_loss

            self.optimizer.zero_grad()
            if self.args.amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            epoch_loss += loss.item()
            epoch_dice += dice

            if not self.args.multi_gpu or (self.args.multi_gpu and self.args.rank == 0):
                if step % 50 == 0:
                    print(f'Epoch: {epoch}, Step: {step}, Loss: {loss.item()}, Dice: {dice}')
                    if dice > self.step_best_dice:
                        self.step_best_dice = dice
                        self.save_checkpoint(
                            epoch,
                            sam_model.state_dict(),
                            describe=f'{epoch}_step_dice_best'
                        )
                    if loss.item() < self.step_best_loss:
                        self.step_best_loss = loss.item()
                        self.save_checkpoint(
                            epoch,
                            sam_model.state_dict(),
                            describe=f'{epoch}_step_loss_best'
                        )
            
        epoch_loss /= step
        epoch_dice /= step
        # epoch_iou /= step

        return epoch_loss, epoch_iou, epoch_dice, pred_list
    
    def plot_result(self, plot_data, description, save_name):

        plt.plot(plot_data)
        plt.title(description)
        plt.xlabel('Epoch')
        plt.ylabel(f'{save_name}')
        # plt.show() # comment this line if you are running on a server
        plt.savefig(join(MODEL_SAVE_PATH, f'{save_name}.png'))
        plt.close()


    def train(self):
        
        # logger.info(f'args : {self.args}')
        # logger.info(f'model : {self.model}')
        # logger.info(f'Used datasets : {img_datas}')
        self.scaler = amp.GradScaler()

        ############################## 一个epoch的训练过程开始 #############################################
        for epoch in range(self.start_epoch, self.args.num_epochs):
            print(f'Epoch: {epoch}/{self.args.num_epochs - 1}')

            if self.args.multi_gpu:
                dist.barrier()
                self.dataloaders.sampler.set_epoch(epoch)

                # sam_model = torch.nn.DataParallel(sam_model)
            num_clicks = np.random.randint(1, 10)
            epoch_loss, epoch_iou, epoch_dice, pred_list = self.train_epoch(epoch, num_clicks)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            if self.args.multi_gpu:
                dist.barrier()
        
            ##################################### 保存权重和loss #########################################
            if not self.args.multi_gpu or (self.args.multi_gpu and self.args.rank == 0):
                self.losses.append(epoch_loss)
                self.dices.append(epoch_dice)
                # self.ious.append(epoch_iou)
                print(f'EPOCH: {epoch}, Loss: {epoch_loss}')
                print(f'EPOCH: {epoch}, Dice: {epoch_dice}')
                # print(f'EPOCH: {epoch}, Loss: {epoch_loss}, IoU: {epoch_iou}')
                # logger.info(f'Epoch\t {epoch}\t : loss: {epoch_loss}, dice: {epoch_dice}')

                if self.args.multi_gpu:
                    state_dict = self.model.module.state_dict()
                else:
                    state_dict = self.model.state_dict()
                
                # save latest checkpoint
                self.save_checkpoint(
                    epoch, 
                    state_dict, 
                    describe='latest'
                )

                # save train loss best checkpoint
                if epoch_loss < self.best_loss: 
                    self.best_loss = epoch_loss
                    self.save_checkpoint(
                        epoch,
                        state_dict,
                        describe='loss_best'
                    )
                
                # save train dice best checkpoint
                if epoch_dice > self.best_dice: 
                    self.best_dice = epoch_dice
                    self.save_checkpoint(
                        epoch,
                        state_dict,
                        describe='dice_best'
                    )

                self.plot_result(self.losses, 'Dice + Cross Entropy Loss', 'Loss')
                self.plot_result(self.dices, 'Dice', 'Dice')
                # self.plot_result(self.ious, 'IoU', 'iou')
            ##################################### 保存权重和loss #########################################
        ########################### 一个epoch的训练过程结束 #############################################
        logger.info('=====================================================================')
        logger.info(f'Best loss: {self.best_loss}')
        logger.info(f'Best dice: {self.best_dice}')
        logger.info(f'Total loss: {self.losses}')
        logger.info(f'Total dice: {self.dices}')
        logger.info('=====================================================================')
        logger.info(f'args : {self.args}')
        # logger.info(f'model : {self.model}')
        logger.info(f'Used datasets : {img_datas}')
        logger.info('=====================================================================')

########################################## Trainer ##########################################


def device_config(args):
    try:
        if not args.multi_gpu:
            # Single GPU
            # args.multi_gpu = False
            if args.device == 'mps':
                args.device = torch.device('mps')
            else:
                args.device = torch.device(f"cuda:{args.gpu_ids[0]}")
        else:
            # args.multi_gpu = True
            args.nodes = 1
            args.ngpus_per_node = len(args.gpu_ids)
            args.world_size = args.nodes * args.ngpus_per_node

    except RuntimeError as e:
        print(e)


def main():
    mp.set_sharing_strategy('file_system')
    device_config(args)
    if args.multi_gpu:
        mp.spawn(
            main_worker,
            nprocs=args.world_size,
            args=(args, )
        )
    else:
        random.seed(2023)
        np.random.seed(2023)
        torch.manual_seed(2023)

        # Load datasets
        dataloaders = get_dataloaders(args)
        
        # Build model
        model = build_model(args)

        # Create trainer
        trainer = BaseTrainer(model, dataloaders, args)

        # Train
        trainer.train()

def main_worker(rank, args):
    setup(rank, args.world_size)

    torch.cuda.set_device(rank)
    args.num_workers = int(args.num_workers / args.ngpus_per_node)
    args.device = torch.device(f"cuda:{rank}")
    args.rank = rank

    random.seed(2023)
    np.random.seed(2023)
    torch.manual_seed(2023)

    dataloaders = get_dataloaders(args)

    model = build_model(args)

    trainer = BaseTrainer(model, dataloaders, args)

    trainer.train()

    cleanup()


def setup(rank, world_size):
    # initialize the process group
    dist.init_process_group(
        backend='nccl',
        init_method=f'tcp://127.0.0.1:{args.port}',
        world_size=world_size,
        rank=rank
    )

def cleanup():
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
