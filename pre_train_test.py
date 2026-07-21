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


dataset = PairedThermalDataset(
    noisy_root="/mnt/data/users/smalayath/fastdvdnetFORK/DAVIS_clean",
    clean_root="/mnt/data/users/smalayath/fastdvdnetFORK/DAVIS_clean",
)

#sample = dataset[0]
#print(len(sample))
#noisy, clean, sigma = sample
#print(noisy.shape)
#print(clean.shape)
#print(sigma)
#print(noisy.dtype)
#print(clean.dtype)
#print(noisy.min(), noisy.max())
#print(clean.min(), clean.max())
#print((noisy-clean).std())

loader = DataLoader(
    dataset,
    batch_size = 4,
    shuffle = True
)

batch = next(iter(loader))
noisy, clean, sigma = batch
#print(noisy.shape)
#print(clean.shape)
#print(sigma.shape)



imgn_train, gt_train = normalize_augment_pair(
    noisy,
    clean,
    ctrl_fr_idx=2
)

print(imgn_train.shape)
print(gt_train.shape)

print(imgn_train.min())
print(imgn_train.max())
