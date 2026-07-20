from torch.utils.data import Dataset
import glob
import os
import cv2
import torch
import numpy as np
import random

# Loads a dataset of paired thermal images for training and evaluation. The dataset consists of noisy and clean thermal image sequences stored in separate directories. 
# Each sample consists of a sequence of 5 frames, with the center frame being randomly selected from the available frames in the sequence. 
# The frames are cropped to a specified patch size, and the pixel values are normalized to the range [0, 1]. The resulting crops are returned as PyTorch tensors.
class PairedThermalDataset(Dataset):

    def __init__(self, noisy_root, clean_root, patch_size=96, temp_patch_size=5, epoch_size=256000):

        self.noisy_root = noisy_root
        self.clean_root = clean_root
        self.patch_size = patch_size
        self.temp_patch_size = temp_patch_size
        self.epoch_size = epoch_size
        self.sequences = sorted(os.listdir(noisy_root))

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, idx):

        seq = random.choice(self.sequences)
        noisy_dir = os.path.join(self.noisy_root, seq)
        clean_dir = os.path.join(self.clean_root, seq)
        noisy_files = sorted(glob.glob(os.path.join(noisy_dir, "*.tif")))
        clean_files = sorted(glob.glob(os.path.join(clean_dir, "*.tif")))
        center = random.randint(2, len(noisy_files)-3)

        noisy_frames = []
        clean_frames = []

        for i in range(center-2, center+3):
            noisy = cv2.imread(noisy_files[i], cv2.IMREAD_UNCHANGED)
            clean = cv2.imread(clean_files[i], cv2.IMREAD_UNCHANGED)
            H, W = noisy.shape
            noisy_frames.append(noisy)
            clean_frames.append(clean)

        x = random.randint(0, W-self.patch_size)
        y = random.randint(0, H-self.patch_size)

        noisy_crop = []
        clean_crop = []

        for n, c in zip(noisy_frames, clean_frames):
            noisy_crop.append(n[y:y+self.patch_size, x:x+self.patch_size])
            clean_crop.append(c[y:y+self.patch_size,x:x+self.patch_size])

        noisy_crop = np.stack(noisy_crop)
        clean_crop = np.stack(clean_crop)

        #14-bit: normalize to [0, 1] and replicate to 3 channels
        #noisy_crop = noisy_crop.astype(np.float32)/16383.0
        #clean_crop = clean_crop.astype(np.float32)/16383.0

        #8-bit: normalize to [0, 1] and replicate to 3 channels
        noisy_crop = noisy_crop.astype(np.float32)/255.0
        clean_crop = clean_crop.astype(np.float32)/255.0

        noisy_crop = np.repeat(noisy_crop[:, None, :, :], 3, axis=1)
        clean_crop = np.repeat(clean_crop[:, None, :, :], 3, axis=1)
        return (torch.from_numpy(noisy_crop), torch.from_numpy(clean_crop))