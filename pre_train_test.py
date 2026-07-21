from torch.utils.data import Dataset
import glob
import os
import cv2
import torch
import numpy as np
import random
from dataset_paired import PairedThermalDataset


dataset = PairedThermalDataset(
    noisy_root=r"C:\Users\SMalayat\Downloads\DAVIS-2017-Unsupervised-trainval-480p\DAVIS\JPEGImages\480p\DAVIS_clean",
    clean_root=r"C:\Users\SMalayat\Downloads\DAVIS-2017-Unsupervised-trainval-480p\DAVIS\JPEGImages\480p\DAVIS_clean",
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