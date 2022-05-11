# -*- coding: utf-8 -*-

# Commented out IPython magic to ensure Python compatibility.
# %%writefile IntmdSequential.py

import torch.nn as nn


class IntermediateSequential(nn.Sequential):
    def __init__(self, *args, return_intermediate=True):
        super().__init__(*args)
        self.return_intermediate = return_intermediate

    def forward(self, input):
        if not self.return_intermediate:
            return super().forward(input)

        intermediate_outputs = {}
        output = input
        for name, module in self.named_children():
            output = intermediate_outputs[name] = module(output)

        return output, intermediate_outputs


# Commented out IPython magic to ensure Python compatibility.
# %%writefile PositionalEncoding.py

import torch
import torch.nn as nn

class FixedPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_length=512):
        super(FixedPositionalEncoding, self).__init__()

        pe = torch.zeros(max_length, embedding_dim)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / embedding_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return x


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_position_embeddings, embedding_dim, seq_length):
        super(LearnedPositionalEncoding, self).__init__()

        self.position_embeddings = nn.Parameter(torch.zeros(1, 1728, 512)) #8x

    def forward(self, x, position_ids=None):

        position_embeddings = self.position_embeddings
        return x + position_embeddings


# Commented out IPython magic to ensure Python compatibility.
# %%writefile TABS_Model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from Transformer import TransformerModel
from PositionalEncoding import LearnedPositionalEncoding

class up_conv_3D(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv_3D, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor = 2),
            nn.Conv3d(ch_in, ch_out, kernel_size = 3, stride = 1, padding = 1, bias = True),
            nn.GroupNorm(8, ch_out),
            # nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace = True)
        )

    def forward(self,x):
        x = self.up(x)
        return x


class conv_block_3D(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block_3D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size = 3, stride = 1, padding = 1, bias = True),
            nn.GroupNorm(8, ch_out),
            nn.ReLU(inplace = True),
            nn.Conv3d(ch_out, ch_out, kernel_size = 3, stride = 1, padding = 1, bias = True),
            nn.GroupNorm(8, ch_out),
            nn.ReLU(inplace = True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x

class resconv_block_3D(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(resconv_block_3D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size = 3, stride = 1, padding = 1, bias = True),
            nn.GroupNorm(8, ch_out),
            nn.ReLU(inplace = True),
            nn.Conv3d(ch_out, ch_out, kernel_size = 3, stride = 1, padding = 1, bias = True),
            nn.GroupNorm(8, ch_out),
            nn.ReLU(inplace = True)
        )
        self.Conv_1x1 = nn.Conv3d(ch_in, ch_out, kernel_size = 1, stride = 1, padding = 0)

    def forward(self,x):
        residual = self.Conv_1x1(x)
        x = self.conv(x)
        return residual + x


class SE_block_3D(nn.Module):
    """
    3D extension of Squeeze-and-Excitation (SE) block adapted from:
     Rickmann et. al., `Project & Excite' Modules for Segmentation of Volumetric Medical Scans
     https://arxiv.org/abs/1906.04649
    """
    def __init__(self, ch_in, reduction_ratio=8):
        super(SE_block_3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.reduction_ratio = reduction_ratio
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction_ratio, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction_ratio, ch_in, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, channels, _, _, _ = x.size()
        
        # Squeezing
        squeeze_tensor = self.avg_pool(x).view(batch_size, channels)
        # Excitation
        squeeze_tensor = self.fc(squeeze_tensor).view(batch_size, channels, 1, 1, 1)
        return x * squeeze_tensor.expand_as(x)
    
class PE_block(nn.Module):
    """
    Project-and-Excite (PE) block adapted from:
     Rickmann et. al., `Project & Excite' Modules for Segmentation of Volumetric Medical Scans
     https://arxiv.org/abs/1906.04649
    """
    def __init__(self, ch_in, reduction_ratio=8):
        super(PE_block, self).__init__()
        self.reduction_ratio = reduction_ratio
        self.fc = nn.Sequential(
            nn.Conv3d(in_channels=ch_in, out_channels= ch_in // reduction_ratio, kernel_size=1, stride=1),
            nn.ReLU(inplace = True),
            nn.Conv3d(in_channels=ch_in // reduction_ratio, out_channels= ch_in, kernel_size=1, stride=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, channels, depth, height, width = x.size()
        
        # Projection
        squeeze_tensor_w = F.adaptive_avg_pool3d(x, (1, 1, width))
        squeeze_tensor_h = F.adaptive_avg_pool3d(x, (1, height, 1))
        squeeze_tensor_d = F.adaptive_avg_pool3d(x, (depth, 1, 1))
        
        final_squeeze_tensor = sum([squeeze_tensor_w.view(batch_size, channels, 1, 1, width),
                                    squeeze_tensor_h.view(batch_size, channels, 1, height, 1),
                                    squeeze_tensor_d.view(batch_size, channels, depth, 1, 1)])
        # Excitation
        final_squeeze_tensor = self.fc(final_squeeze_tensor)
        return x * final_squeeze_tensor.expand_as(x)

class TABS(nn.Module):
    def __init__(
        self,
        img_dim = 192,
        patch_dim = 8,
        img_ch = 1,
        output_ch = 3,
        embedding_dim = 512,
        num_heads = 8,
        num_layers = 4,
        hidden_dim = 1728,
        dropout_rate = 0.1,
        attn_dropout_rate = 0.1,
        ):
        super(TABS,self).__init__()

        self.Maxpool = nn.MaxPool3d(kernel_size=2,stride=2)

        self.Conv1 = resconv_block_3D(ch_in=img_ch,ch_out=8)

        self.Conv2 = resconv_block_3D(ch_in=8,ch_out=16)

        self.Conv3 = resconv_block_3D(ch_in=16,ch_out=32)

        self.Conv4 = resconv_block_3D(ch_in=32,ch_out=64)

        self.Conv5 = resconv_block_3D(ch_in=64,ch_out=128)

        self.Up5 = up_conv_3D(ch_in=128,ch_out=64)
        self.Up_conv5 = resconv_block_3D(ch_in=128, ch_out=64)

        self.Up4 = up_conv_3D(ch_in=64,ch_out=32)
        self.Up_conv4 = resconv_block_3D(ch_in=64, ch_out=32)

        self.Up3 = up_conv_3D(ch_in=32,ch_out=16)
        self.Up_conv3 = resconv_block_3D(ch_in=32, ch_out=16)

        self.Up2 = up_conv_3D(ch_in=16,ch_out=8)
        self.Up_conv2 = resconv_block_3D(ch_in=16, ch_out=8)

        self.Conv_1x1 = nn.Conv3d(8,output_ch,kernel_size=1,stride=1,padding=0)
        self.gn = nn.GroupNorm(8, 128)
        self.relu = nn.ReLU(inplace=True)

        self.num_patches = int((img_dim // patch_dim) ** 3)
        self.seq_length = self.num_patches
        self.flatten_dim = 128 * img_ch

        self.position_encoding = LearnedPositionalEncoding(
            self.seq_length, embedding_dim, self.seq_length
        )

        self.act = nn.Softmax(dim=1)

        self.reshaped_conv = conv_block_3D(512, 128)

        self.transformer = TransformerModel(
            embedding_dim,
            num_layers,
            num_heads,
            hidden_dim,

            dropout_rate,
            attn_dropout_rate,
        )

        self.conv_x = nn.Conv3d(
            128,
            embedding_dim,
            kernel_size=3,
            stride=1,
            padding=1
            )

        self.pre_head_ln = nn.LayerNorm(embedding_dim)

        self.img_dim = 192
        self.patch_dim = 8
        self.img_ch = 1
        self.output_ch = 3
        self.embedding_dim = 512
        
        self.SE = SE_block_3D(ch_in=128) # Squeeze-Excitation
#         self.PE = PE_block(ch_in=128) # Projection-Excitation

    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x = self.Conv5(x5)
        
        x = self.SE(x) # Squeeze-Excitation
#         x = self.PE(x) # Projection-Excitation
        
        x = self.gn(x)
        x = self.relu(x)
        x = self.conv_x(x)

        x = x.permute(0, 2, 3, 4, 1).contiguous()
        x = x.view(x.size(0), -1, self.embedding_dim)

        x = self.position_encoding(x)

        x, intmd_x = self.transformer(x)
        x = self.pre_head_ln(x)

        encoder_outputs = {}
        all_keys = []
        for i in [1, 2, 3, 4]:
            val = str(2 * i - 1)
            _key = 'Z' + str(i)
            all_keys.append(_key)
            encoder_outputs[_key] = intmd_x[val]
        all_keys.reverse()

        x = encoder_outputs[all_keys[0]]
        x = self._reshape_output(x)
        x = self.reshaped_conv(x)

        d5 = self.Up5(x)
        d5 = torch.cat((x4,d5),dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        d1 = self.act(d1)

        return d1

    def _reshape_output(self, x):
        x = x.view(
            x.size(0),
            int(self.img_dim//2 / self.patch_dim),
            int(self.img_dim//2 / self.patch_dim),
            int(self.img_dim//2 / self.patch_dim),
            self.embedding_dim,
        )
        x = x.permute(0, 4, 1, 2, 3).contiguous()

        return x

# Commented out IPython magic to ensure Python compatibility.
# %%writefile Transformer.py

import torch.nn as nn
from IntmdSequential import IntermediateSequential


class SelfAttention(nn.Module):
    def __init__(
        self, dim, heads=8, qkv_bias=False, qk_scale=None, dropout_rate=0.0
    ):
        super().__init__()
        self.num_heads = heads
        head_dim = dim // heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))


class PreNormDrop(nn.Module):
    def __init__(self, dim, dropout_rate, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fn = fn

    def forward(self, x):
        return self.dropout(self.fn(self.norm(x)))


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=dropout_rate),
        )

    def forward(self, x):
        return self.net(x)


class TransformerModel(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        mlp_dim,
        dropout_rate=0.1,
        attn_dropout_rate=0.1,
    ):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers.extend(
                [
                    Residual(
                        PreNormDrop(
                            dim,
                            dropout_rate,
                            SelfAttention(dim, heads=heads, dropout_rate=attn_dropout_rate),
                        )
                    ),
                    Residual(
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout_rate))
                    ),
                ]
            )
        self.net = IntermediateSequential(*layers)


    def forward(self, x):
        return self.net(x)


# Import stuff
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from glob import glob
import nibabel as nib
from torchvision import transforms
import torch
from torch.utils.data import Dataset, DataLoader
import random

class MRIDataset():

    def __init__(self, mri_dir, masks_dir, protocol, mode, val_fold, test_fold):
        self.mode = mode
        self.mri_dir = mri_dir
        self.masks_dir = masks_dir
        self.protocol = protocol
        self.val_fold = val_fold
        self.test_fold = test_fold
        self.train_folds = [1,2,3,4,5]

        self.maskFiles = []

        # This dataloader is only for cross validation, but can be adapted for test as well
        assert self.mode in ['train','val', 'test', 'gen_outputs']

        if self.mode == 'train':
            # For this dataset, I did cross validation. If you would like to also have a test fold, uncomment line 35

            self.train_folds.remove(self.test_fold)
            self.train_folds.remove(self.val_fold)

            # Get lists of the image file paths and each code
            self.imageFiles, self.imageCodes = self.getpaths(self.train_folds,self.mri_dir, self.protocol, self.mode)

            # Create dictionary of mask file paths corresponding to each image code
            self.masks_dict = self.getmasks(self.masks_dir, self.protocol, self.imageCodes)

        elif mode == 'val' or mode == 'test' or mode == 'gen_outputs':
            if mode == 'val':
                self.imageFiles, self.imageCodes = self.getpaths([self.val_fold], self.mri_dir, self.protocol, self.mode)

            else:
                self.imageFiles, self.imageCodes = self.getpaths([self.test_fold],self.mri_dir,self.protocol, self.mode)

            self.masks_dict = self.getmasks(self.masks_dir, self.protocol, self.imageCodes)

    def __getitem__(self,idx):
        image_filepath = self.imageFiles[idx]
        code = self.imageCodes[idx]

        #now we can load the nifti file and process it
        image, stacked_masks = self.loadimage(image_filepath, code, self.protocol)

        sample = {'T1': image,
              'label': stacked_masks,
              'code': code
              }

        return sample

    def __len__(self):
        return len(self.imageFiles)

    def getpaths(self, folds, mri_dir, protocol, mode):
        imageFiles = []
        imageCodes = []
        for fold in folds:
            # Add image file paths for every image in a given fold
            imageFiles.extend(sorted(glob(mri_dir + '/Fold_' + str(fold) + '/*.nii')))
        if protocol == 'dlbs':
            for path in imageFiles:
                # The different domains have the codes in slightly different portions of the path
                imageCodes.append(path[-72:-65])
        elif protocol == 'SALD':
            for path in imageFiles:
                imageCodes.append(path[-51:-45])
        elif protocol == 'IXI':
            for path in imageFiles:
                imageCodes.append(path[-59:-56])
        elif protocol == 'total':
            for path in imageFiles:
                if 'dlbs' in path:
                    imageCodes.append(path[-72:-65])
                elif 'SALD' in path:
                    imageCodes.append(path[-51:-45])
                elif 'IXI' in path:
                    imageCodes.append(path[-59:-56])
        # for testing purposes with dlbs
        else:
            for path in imageFiles:
                imageCodes.append(path[-72:-65])
        print("The {} dataset for domain {} now has {} images".format(mode, protocol, len(imageFiles)))
        return imageFiles, imageCodes

    def getmasks(self, masks_dir, protocol, imageCodes):
        intermediate = []
        maskFiles = []
        # Start by making an intermediate list of all the masks paths.
        intermediate.extend(sorted(glob(masks_dir + '/*.nii')))
        

        for file in intermediate:
            if protocol == 'dlbs':
                cur_code = file[-20:-13]
            if protocol == 'SALD':
                cur_code = file[-16:-10]
            if protocol == 'IXI':
                cur_code = file[-16:-13]
            if protocol == 'total':
                if 'dlbs' in file:
                    cur_code = file[-20:-13]
                if 'SALD' in file:
                    cur_code = file[-16:-10]
                if 'IXI' in file:
                    cur_code = file[-16:-13]
            if cur_code in imageCodes:
                # Only add the mask files that correspond to images we care about
                maskFiles.append(file)

        # Create masks dict
        masks_dict = { i : [] for i in imageCodes }
        for mask in maskFiles:
            for code in imageCodes:
                if code in mask:
                    masks_dict[code].append(mask)
        return masks_dict

    def loadimage(self, image_filepath, code, protocol):
        # Load, splice, and pad image
        image = nib.load(image_filepath)
        image = image.slicer[:,15:207,:]
        image = np.array(image.dataobj)
        image = np.pad(image, [(5, 5), (0, 0), (5,5)], mode='constant')
        image = torch.from_numpy(image)
        image = image.unsqueeze(0)
        image = image.float()
        max = torch.max(image)
        if protocol == 'dlbs':
            image = 2*(image / max) - 1
        if protocol == 'IXI':
            image = 2*(image / max) - 1
        if protocol == 'SALD' or protocol == 'total':
            image = 2*(image / max) - 1
            
        # -----------------------  Data Augmentation -----------------------
        transform = tio.RandomAffine(
            scales=(0.9, 1.2),
            degrees=15,
        )
        image = transform(image)
        
        transform = tio.RandomGamma(log_gamma=(-0.3, 0.3))
        image = transform(image)
        # ------------------------------------------------------------------

        masks = ['mask_1', 'mask_2', 'mask_3']
        mask_tensors = { i : {} for i in masks }
        # Load, slice, and pad each mask. Add them to masks dictionary
        for num in range(1,4):
            mask_tensors['mask_'+ str(num)] = nib.load(self.masks_dict[code][num-1])
            mask_tensors['mask_'+ str(num)] = mask_tensors['mask_'+ str(num)].slicer[:,15:207,:]
            mask_tensors['mask_'+ str(num)] = np.array(mask_tensors['mask_'+ str(num)].dataobj)
            mask_tensors['mask_'+ str(num)] = np.pad(mask_tensors['mask_'+ str(num)], [(5, 5), (0, 0), (5, 5)], mode='constant')
            mask_tensors['mask_'+ str(num)] = torch.from_numpy(mask_tensors['mask_'+ str(num)])

        # Stack all 3 individual brain masks to a single 3 channel GT
        stacked_masks = torch.stack([mask_tensors['mask_1'], mask_tensors['mask_2'], mask_tensors['mask_3']], dim=0)
        stacked_masks = stacked_masks.float()

        return image, stacked_masks

class Arg:
    MRI_dir_SALD = '../input/sald-data/'
    GT_dir_SALD = '../input/sald-mask/Masks_SALD/'
    def __init__(self):
        self.protocol = 'SALD'
        self.possible_protocols = ['SALD']
        self.end_epoch = 120
        self.seed = 1000
        self.lr = 0.00001
        self.weight_decay = 1e-5
        self.amsgrad = True
        self.val_fold = 5
        self.test_fold = 1
        self.batch_size = 1
        self.num_workers = 2
        self.start_epoch = 0
        self.save_root = ''

# !pip install torchio

import torch
import torchio as tio

## TRAIN CODE

# Import stuff
import os
import random
import logging
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import torch.optim

import torch.distributed as dist
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio
import nibabel as nib

from TABS_Model import TABS


args = Arg()

model = TABS()
# Declare lists to keep track of training and val losses over the epochs
train_global_losses = []
val_global_losses = []
best_epoch = 0

def main_worker():

    assert args.protocol in args.possible_protocols, 'Protocol must be one of 4 possible protocols: dlbs, SALD, IXI, total'

    if args.protocol == 'total':
        args.end_epoch = 200

    dirs = {i : [] for i in args.possible_protocols}
    for protocol in args.possible_protocols:
        mri_dir = getattr(args, 'MRI_dir_' + protocol)
        gt_dir = getattr(args, 'GT_dir_' + protocol)
        dirs[protocol].append(mri_dir)
        dirs[protocol].append(gt_dir)

    MRI_dir = dirs[args.protocol][0]
    GT_dir = dirs[args.protocol][1]

    # Set seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    print(count_parameters(model))
    
    model.cuda()

    print('Model Built!')

    # Using adam optimizer (amsgrad variant) with weight decay
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=args.amsgrad)

    # MSE loss for this task (regression). Using reduction value of sum because we want to specify the number of voxels to divide by (only in the brain map)
    criterion = nn.MSELoss(reduction='sum')
#     criterion = criterion.cuda(args.local_rank)  # not necessary
    criterion = criterion.cuda()

    # Obtain train dataset
    Train_MRIDataset = MRIDataset(MRI_dir, GT_dir, protocol=args.protocol, mode='train', val_fold=args.val_fold, test_fold=args.test_fold)
    
    # Obtain train dataloader
    Train_dataloader = DataLoader(Train_MRIDataset, shuffle=True, num_workers=args.num_workers, batch_size=args.batch_size, pin_memory=True)

    # Obtain val dataset
    Val_MRIDataset = MRIDataset(MRI_dir, GT_dir, protocol=args.protocol, mode='val', val_fold=args.val_fold, test_fold=args.test_fold)

    # Obtain val_dataloader
    Val_dataloader = DataLoader(Val_MRIDataset, shuffle=False, num_workers=args.num_workers, batch_size=args.batch_size, pin_memory=True)

    start_time = time.time()

    # Enable gradient calculation for training
    torch.set_grad_enabled(True)

    print('Start to train!')

    # Main training/validation loop
    for epoch in range(args.start_epoch, args.end_epoch):

        # Declare lists to keep track of losses and metrics within the epoch
        train_epoch_losses = []
        val_epoch_losses = []
        val_epoch_pcorr = []
        val_epoch_psnr = []
        start_epoch = time.time()

        model.train()

        count = 0
        for i, data in enumerate(Train_dataloader):
            # break

            adjust_learning_rate(optimizer, epoch, args.end_epoch, args.lr)

            mri_images = data['T1']
            targets = data['label']

            mri_images = mri_images.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

            loss, isolated_images, stacked_brain_map  = get_loss(model, criterion, mri_images, targets, 'train')

            train_epoch_losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Transition to val mode
        model.eval()

        # Avoid computing gradients during validation to save memory
        with torch.no_grad():

            for i, data in enumerate(Val_dataloader):

                mri_images = data['T1']
                targets = data['label']

                mri_images = mri_images.cuda(non_blocking=True)
                targets = targets.cuda(non_blocking=True)

                loss, isolated_images, stacked_brain_map  = get_loss(model, criterion, mri_images, targets, 'val')

                val_epoch_losses.append(loss.item())

                # Calculate the pearson correlation between the output and ground truth only for the voxels of the brain map.

                for g in range(0,len(isolated_images)):
                    cur_pcorr, cur_psnr = overall_metrics(isolated_images[g], targets[g], stacked_brain_map[g])
                    val_epoch_pcorr.append(cur_pcorr)
                    val_epoch_psnr.append(cur_psnr)

        end_epoch = time.time()

        # Average train and val loss over every MRI scan in the epoch. Save to global losses which tracks across epochs
        train_net_loss = sum(train_epoch_losses) / len(train_epoch_losses)
        val_net_loss = sum(val_epoch_losses) / len(val_epoch_losses)
        train_global_losses.append(train_net_loss)
        val_global_losses.append(val_net_loss)
        # Average pearson correlation and psnr over the epochs
        psnr = sum(val_epoch_psnr) / len(val_epoch_psnr)
        pcorr = sum(val_epoch_pcorr) / len(val_epoch_pcorr)

        print('Epoch: {} | Train Loss: {} | Val Loss: {} | PSNR: {} | Pearson: {}'.format(epoch, train_net_loss, val_net_loss, psnr, pcorr))

        checkpoint_dir = args.save_root
        # Save the model if it reaches a new min validation loss
        if val_global_losses[-1] == min(val_global_losses):
            print('saving model at the end of epoch ' + str(epoch))
            best_epoch = epoch
            file_name = os.path.join(checkpoint_dir, 'ResTransUNet3D_model_epoch_{}.pth'.format(epoch, val_global_losses[-1]))
            if epoch > 150:
                torch.save({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optim_dict': optimizer.state_dict(),
                    },
                    file_name)

    end_time = time.time()
    total_time = (end_time - start_time) / 3600
    print('The total training time is {:.2f} hours'.format(total_time))

    print('----------------------------------The training process finished!-----------------------------------')

    log_name = os.path.join(args.save_root, 'loss_log_restransunet.txt')

    with open(log_name, "a") as log_file:
        now = time.strftime("%c")
        log_file.write('================ Loss (%s) ================\n' % now)
        log_file.write('best_epoch: ' + str(best_epoch) + '\n')
        log_file.write('train_losses: ')
        log_file.write('%s\n' % train_global_losses)
        log_file.write('val_losses: ')
        log_file.write('%s\n' % val_global_losses)
        log_file.write('train_time: ' + str(total_time))

    learning_curve(best_epoch, train_global_losses, val_global_losses)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Input the best epoch, lists of global (across epochs) train and val losses. Plot learning curve
def learning_curve(best_epoch, train_global_losses, val_global_losses):
    fig, ax1 = plt.subplots(figsize=(12, 8))

    ax1.set_xlabel('Epochs')
    ax1.set_xticks(np.arange(0, int(len(train_global_losses) + 1), 10))

    ax1.set_ylabel('Loss')
    ax1.plot(train_global_losses, '-r', label='Training loss', markersize=3)
    ax1.plot(val_global_losses, '-b', label='Validation loss', markersize=3)
    ax1.axvline(best_epoch, color='m', lw=4, alpha=0.5, label='Best epoch')
    ax1.legend(loc='upper left')
    save_name = 'Learning_Curve_restransunet3d_' + args.protocol + '_' + str(args.val_fold) + '.png'
    plt.savefig(os.path.join(args.save_root, save_name))

def adjust_learning_rate(optimizer, epoch, max_epoch, init_lr, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(init_lr * np.power(1 - (epoch / max_epoch), power), 8)

# Calculate pearson correlation and psnr only between the voxels of the brain map (do by total brain not tissue type during training)
def overall_metrics(isolated_image, target, stacked_brain_map):
    # Flatten the GT, isolated output, and brain mask
    GT_flattened = torch.flatten(target)
    iso_flattened = torch.flatten(isolated_image)
    mask_flattened = torch.flatten(stacked_brain_map)

    # Only save the part of the flattened GT/output that corresponds to nonzero values of the brain mask
    GT_flattened = GT_flattened[mask_flattened.nonzero(as_tuple=True)]
    iso_flattened = iso_flattened[mask_flattened.nonzero(as_tuple=True)]

    iso_flattened = iso_flattened.cpu().detach().numpy()
    GT_flattened = GT_flattened.cpu().detach().numpy()

    pearson = np.corrcoef(iso_flattened, GT_flattened)[0][1]
    psnr = peak_signal_noise_ratio(iso_flattened, GT_flattened)

    return pearson, psnr

# Given the model, criterion, input, and GT, this function calculates the loss and returns the isolated output (stripped of background) and brain map
def get_loss(model, criterion, mri_images, targets, mode):

    if mode == 'val':
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

    # Get brain map from FSL GT by taking all values >0 --> 1
    output = model(mri_images)

    test = torch.squeeze(mri_images,dim=1)
    brain_map = (test > -1).float()

    stacked_brain_map = torch.stack([brain_map, brain_map, brain_map], dim=1)

    isolated_images = torch.mul(stacked_brain_map, output)

    loss = criterion(isolated_images, targets)
    num_brain_voxels = stacked_brain_map.sum()
    loss = loss / num_brain_voxels

    return loss, isolated_images, stacked_brain_map

if __name__ == '__main__':
    assert torch.cuda.is_available(), "Currently, we only support CUDA version"
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    main_worker()

# !pip install MedPy

## TEST CODE

import numpy as np

import torch
import torch.distributed as dist
import torch.nn as nn
import random
from medpy.metric.binary import hd
import nibabel as nib
import os

from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio
from scipy.stats import spearmanr
from sklearn.metrics import jaccard_score
from TABS_Model import TABS

def get_loss(model, criterion, mri_images, targets, mode):

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Get brain map from FSL GT by taking all values >0 --> 1
    output = model(mri_images)

    test = torch.squeeze(mri_images,dim=1)
    brain_map = (test > -1).float()

    stacked_brain_map = torch.stack([brain_map, brain_map, brain_map], dim=1)

    isolated_images = torch.mul(stacked_brain_map, output)

    loss = criterion(isolated_images, targets)
    num_brain_voxels = stacked_brain_map.sum()
    loss = loss / num_brain_voxels

    return loss, isolated_images, stacked_brain_map

def tissue_wise_probability_metrics(isolated_image, target, stacked_brain_map):
    criterion = nn.MSELoss()
    criterion = criterion.cuda()

    # metrics dict to store metric for each tissue type
    metrics_list = ['pearson_corr', 'spearman_corr', 'psnr', 'mse']
    metrics = { i : [] for i in metrics_list }

    # list of flattened tensors I'm gonna collect and their corresponding dict
    necessary_flattened_tensors = ['GT_flattened_0', 'GT_flattened_1', 'GT_flattened_2', 'iso_flattened_0', 'iso_flattened_1', 'iso_flattened_2']
    flattened_tensors = { i : {} for i in necessary_flattened_tensors }

    # flattened single channel brain mask (192x192x192 --> flat)
    mask_flattened = torch.flatten(stacked_brain_map[0])

    # Only save the part of the flattened GT/output that correspond to nonzero values of the brain mask
    for i in range(0,3):
        # flatten gt of channel i (each channel corresponds to a tissue type)
        flattened_tensors['GT_flattened_' + str(i)] = torch.flatten(target[i])
        # choose only the portion of the flattened gt that correspons to the brain
        flattened_tensors['GT_flattened_' + str(i)] = flattened_tensors['GT_flattened_' + str(i)][mask_flattened.nonzero(as_tuple=True)]
        # make this now a numpy array
        flattened_tensors['GT_flattened_' + str(i)] = flattened_tensors['GT_flattened_' + str(i)].cpu().detach().numpy()

        # repeat for the model output image
        flattened_tensors['iso_flattened_' + str(i)] = torch.flatten(isolated_image[i])
        flattened_tensors['iso_flattened_' + str(i)] = flattened_tensors['iso_flattened_' + str(i)][mask_flattened.nonzero(as_tuple=True)]
        flattened_tensors['iso_flattened_' + str(i)] = flattened_tensors['iso_flattened_' + str(i)].cpu().detach().numpy()

    for i in range(0,3):
        # get output and gt from dict i just constructed
        model_output = flattened_tensors['iso_flattened_' + str(i)]
        GT = flattened_tensors['GT_flattened_' + str(i)]

        # get metrics using the numpy arrays of both (cropped to brain)
        cur_pcorr = np.corrcoef(model_output, GT)[0][1]
        cur_scorr = spearmanr(model_output, GT)[0]
        cur_psnr = peak_signal_noise_ratio(model_output, GT)

        cur_mse = criterion(torch.tensor(model_output).cuda(), torch.tensor(GT).cuda())

        metrics['pearson_corr'].append(cur_pcorr)
        metrics['spearman_corr'].append(cur_scorr)
        metrics['psnr'].append(cur_psnr)
        metrics['mse'].append(cur_mse.item())

    return metrics

def tissue_wise_map_metrics(isolated_image, target, stacked_brain_map):
    # metrics dict to store metric for each tissue type
    metrics_list = ['DICE', 'HD', 'Jaccard']
    metrics = { i : [] for i in metrics_list }

    # list of flattened tensors (segmentation masks) I'm gonna collect and their corresponding dict
    necessary_masks_list = ['GT_0', 'GT_1', 'GT_2', 'iso_0', 'iso_1', 'iso_2']
    necessary_tensors = { i : {} for i in necessary_masks_list }

    # current output and gt is 3x192x192x192. Basically, each voxel of the brain has 3 probabilities assigned to it for each tissue type. Taking the argmax gives us the most likely tissue type of each voxel (now 1x192x192x192)
    full_map_model = torch.argmax(isolated_image,0)
    full_map_GT = torch.argmax(target,0)
    mask = stacked_brain_map[0]
    mask_flattened = torch.flatten(stacked_brain_map[0])

    for i in range(0,3):
        # now that we have the argmax, we can imagine the brain with each voxel having a value of 0,1,2. To get the masks for each tissue type, we save a new tensor corresponding to 1 where the argmax tensor has a value of the given tissue type and 0 otherwise.
        necessary_tensors['GT_' + str(i)] = (full_map_GT==i).float()
        necessary_tensors['iso_' + str(i)] = (full_map_model==i).float()
        if i == 0:
            # make sure background is 0
            necessary_tensors['GT_' + str(i)] = torch.mul(necessary_tensors['GT_' + str(i)], mask)
            necessary_tensors['iso_' + str(i)] = torch.mul(necessary_tensors['iso_' + str(i)], mask)

        # calc HD with the segmentation masks
        h_dist = hd(necessary_tensors['iso_' + str(i)].cpu().detach().numpy(), necessary_tensors['GT_' + str(i)].cpu().detach().numpy())
        metrics['HD'].append(h_dist)

        # now make cropped 1d numpy arrays only containing mask values for within the brain for dice calculation
        necessary_tensors['GT_' + str(i)] = torch.flatten(necessary_tensors['GT_' + str(i)])
        necessary_tensors['GT_' + str(i)] = necessary_tensors['GT_' + str(i)][mask_flattened.nonzero(as_tuple=True)]
        necessary_tensors['GT_' + str(i)] = necessary_tensors['GT_' + str(i)].cpu().detach().numpy()
        necessary_tensors['iso_' + str(i)] = torch.flatten(necessary_tensors['iso_' + str(i)])
        necessary_tensors['iso_' + str(i)] = necessary_tensors['iso_' + str(i)][mask_flattened.nonzero(as_tuple=True)]
        necessary_tensors['iso_' + str(i)] = necessary_tensors['iso_' + str(i)].cpu().detach().numpy()

    for i in range(0,3):
        model_output = necessary_tensors['iso_' + str(i)]
        GT = necessary_tensors['GT_' + str(i)]
        # dice formula
        dice = np.sum(model_output[GT==1])*2.0 / (np.sum(model_output) + np.sum(GT))
        jaccard = jaccard_score(GT, model_output)

        metrics['DICE'].append(dice)
        metrics['Jaccard'].append(jaccard)

    return metrics

if __name__ == '__main__':

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    dirs = { i : [] for i in args.possible_protocols}
    for protocol in args.possible_protocols:
        mri_dir = getattr(args, 'MRI_dir_' + protocol)
        gt_dir = getattr(args, 'GT_dir_' + protocol)
        dirs[protocol].append(mri_dir)
        dirs[protocol].append(gt_dir)

    MRI_dir = dirs[args.protocol][0]
    GT_dir = dirs[args.protocol][1]

    Test_MRIDataset = MRIDataset(MRI_dir, GT_dir, protocol=args.protocol, mode='test', val_fold = 5, test_fold = args.test_fold)

    Test_dataloader = DataLoader(Test_MRIDataset, shuffle=False, num_workers=args.num_workers, batch_size=args.batch_size, pin_memory=False)

    criterion = nn.MSELoss(reduction='sum')
    criterion = criterion.cuda()

    probability_metrics_list = ['pearson_corr', 'spearman_corr', 'psnr', 'mse']
    probability_metrics = { i : [] for i in probability_metrics_list }
    map_metrics_list = ['DICE', 'HD', 'Jaccard']
    map_metrics = { i : [] for i in map_metrics_list }

    model.eval()

    with torch.no_grad():
        val_losses = []
        test = []
        val_psnr = []
        val_corr = []

        count = 0
        for i, data in enumerate(Test_dataloader):

            mri_images = data['T1']
            targets = data['label']
            samples = data['code']

            mri_images = mri_images.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

            loss, isolated_images, stacked_brain_maps  = get_loss(model, criterion, mri_images, targets, 'val')

            val_losses.append(loss)
            #
            for g in range(0,len(isolated_images)):
                isolated_image = isolated_images[g]
                target = targets[g]
                stacked_brain_map = stacked_brain_maps[g]
                metrics_maps = tissue_wise_map_metrics(isolated_image, target, stacked_brain_map)
                metrics =  tissue_wise_probability_metrics(isolated_image, target, stacked_brain_map)

                for metric in probability_metrics_list:
                    probability_metrics[metric].append(metrics[metric])
                for metric in map_metrics_list:
                    map_metrics[metric].append(metrics_maps[metric])

    val_net_loss = sum(val_losses)/len(val_losses)

    overall_pcorr = probability_metrics['pearson_corr']
    overall_pcorr = np.array(overall_pcorr)
    avg_pcorr = sum(overall_pcorr)/len(overall_pcorr)
    sd_pcorr = np.std(overall_pcorr, axis=0, ddof=1)

    overall_scorr = probability_metrics['spearman_corr']
    overall_scorr = np.array(overall_scorr)
    avg_scorr = sum(overall_scorr)/len(overall_scorr)
    sd_scorr = np.std(overall_scorr, axis=0, ddof=1)

    overall_psnr = probability_metrics['psnr']
    overall_psnr = np.array(overall_psnr)
    avg_psnr = sum(overall_psnr)/len(overall_psnr)
    sd_psnr = np.std(overall_psnr, axis=0, ddof=1)

    overall_mse = probability_metrics['mse']
    overall_mse = np.array(overall_mse)
    avg_mse = sum(overall_mse)/len(overall_mse)
    sd_mse = np.std(overall_mse, axis=0, ddof=1)

    overall_DICE = map_metrics['DICE']
    overall_DICE = np.array(overall_DICE)
    avg_DICE = sum(overall_DICE)/len(overall_DICE)
    sd_DICE = np.std(overall_DICE, axis=0, ddof=1)

    overall_HD = map_metrics['HD']
    overall_HD = np.array(overall_HD)
    avg_HD = sum(overall_HD)/len(overall_HD)
    sd_HD = np.std(overall_HD, axis=0, ddof=1)

    overall_jaccard = map_metrics['Jaccard']
    overall_jaccard = np.array(overall_jaccard)
    avg_jaccard = sum(overall_jaccard)/len(overall_jaccard)
    sd_jaccard = np.std(overall_jaccard, axis=0, ddof=1)

    print('Probability-Based Metrics:')
    print('Val Loss: {} | Pearson: {} SD: {} | Spearman: {} SD: {} | psnr: {} SD: {} | MSE: {} SD: {} |'.format(val_net_loss, avg_pcorr, sd_pcorr, avg_scorr, sd_scorr, avg_psnr, sd_psnr, avg_mse, sd_mse))

    print('Map-Based Metrics:')
    print('DICE: {} SD: {} | HD: {} SD: {} | Jaccard: {} SD: {}'.format(avg_DICE, sd_DICE, avg_HD, sd_HD, avg_jaccard, sd_jaccard))

    log_name = os.path.join(args.save_root, 'test_restransunet_ixi.txt')
    with open(log_name, "a") as log_file:
        log_file.write('Pearson: {} SD: {} | Spearman: {} SD: {} | psnr: {} SD: {} | MSE: {} SD: {} |'.format(avg_pcorr, sd_pcorr, avg_scorr, sd_scorr, avg_psnr, sd_psnr, avg_mse, sd_mse))
        log_file.write('\n')
        log_file.write('DICE: {} SD: {} | HD: {} SD: {} | Jaccard: {} SD: {}'.format(avg_DICE, sd_DICE, avg_HD, sd_HD, avg_jaccard, sd_jaccard))
        log_file.write('\n')
        log_file.write('pcorr')
        log_file.write('%s\n' % overall_pcorr)
        log_file.write('scorr')
        log_file.write('%s\n' % overall_scorr)
        log_file.write('MSE')
        log_file.write('%s\n' % overall_mse)
        log_file.write('dice')
        log_file.write('%s\n' % overall_DICE)
        log_file.write('jaccard')
        log_file.write('%s\n' % overall_jaccard)
        log_file.write('hd')
        log_file.write('%s\n' % overall_HD)
