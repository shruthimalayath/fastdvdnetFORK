from torch.utils.data import Dataset
import glob
import os
import cv2
import torch
import numpy as np
import random
from dataset_paired import PairedThermalDataset
from torch.utils.data import DataLoader
from utils import normalize_augment_pair
from models import FastDVDnet
import torch

#PairedThermalDataSet test
dataset = PairedThermalDataset(
    noisy_root="/mnt/data/users/smalayath/fastdvdnetFORK/DAVIS_clean",
    clean_root="/mnt/data/users/smalayath/fastdvdnetFORK/DAVIS_clean",
)

sample = dataset[0]        
print(len(sample))
noisy, clean, sigma = sample
print(noisy.shape)
print(clean.shape)
print(sigma)
print(noisy.dtype)
print(clean.dtype)
print(noisy.min(), noisy.max())
print(clean.min(), clean.max())
print((noisy-clean).std())


#PyTorch DatLoader test
loader = DataLoader(dataset,batch_size = 4,shuffle = True)
batch = next(iter(loader))
noisy, clean, sigma = batch
print(noisy.shape)
print(clean.shape)
print(sigma.shape)


#normalize_augment_pair test
imgn_train, gt_train = normalize_augment_pair(noisy,clean,ctrl_fr_idx=2)
print(imgn_train.shape)
print(gt_train.shape)
print(imgn_train.min())
print(imgn_train.max())


model = FastDVDnet().cuda()
imgn_train = imgn_train.cuda()
N, _, H, W = imgn_train.shape

noise_map = torch.full(
    (N, 1, H, W),
    25/255.,
    device=imgn_train.device
)

with torch.no_grad():
    out = model(imgn_train, noise_map)

print(out.shape)


'''
output:
python pre_train_test.py
3
torch.Size([5, 3, 96, 96])
torch.Size([5, 3, 96, 96])
tensor(13.6049)
torch.float32
torch.float32
tensor(0.) tensor(254.0568)
tensor(3.) tensor(226.)
tensor(13.5447)
torch.Size([4, 5, 3, 96, 96])
torch.Size([4, 5, 3, 96, 96])
torch.Size([4])
torch.Size([4, 15, 96, 96])
torch.Size([4, 3, 96, 96])
tensor(0.)
tensor(1.)
torch.Size([4, 3, 96, 96]) --> model accepts input

'''