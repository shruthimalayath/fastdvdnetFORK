"""
Paired validation dataset for noisy/clean thermal sequences.
"""

import os
import glob
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import random

# Loads a dataset of paired thermal images for validation. 
# The dataset consists of noisy and clean thermal image sequences stored in separate
# Different from dataset_paired.py, because this doesn't random cropping or patch extraction, but instead loads the entire sequence of frames for each sample.
class PairedValDataset(Dataset):
    def __init__(self, noisy_root, clean_root, max_num_fr=15):

        self.noisy_root = noisy_root
        self.clean_root = clean_root
        self.max_num_fr = max_num_fr
        self.sequences = sorted(os.listdir(noisy_root))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):

        seq_name = self.sequences[idx]
        noisy_dir = os.path.join(self.noisy_root, seq_name)
        clean_dir = os.path.join(self.clean_root, seq_name)
        #noisy_files = sorted(glob.glob(os.path.join(noisy_dir, "*.tif")))[:self.max_num_fr]
        #clean_files = sorted(glob.glob(os.path.join(clean_dir, "*.tif")))[:self.max_num_fr]
        noisy_files = sorted(glob.glob(os.path.join(noisy_dir, "*.jpg")))[:self.max_num_fr]
        clean_files = sorted(glob.glob(os.path.join(clean_dir, "*.jpg")))[:self.max_num_fr]

        if len(noisy_files) != len(clean_files):
            raise ValueError(
                f"Sequence {seq_name}: "
                f"{len(noisy_files)} noisy frames but "
                f"{len(clean_files)} clean frames"
            )

        noisy_frames = []
        clean_frames = []

        for nf, cf in zip(noisy_files, clean_files):

            noisy = cv2.imread(nf, cv2.IMREAD_UNCHANGED)
            clean = cv2.imread(cf, cv2.IMREAD_UNCHANGED)

            if noisy is None:
                raise ValueError(f"Could not read {nf}")
            if clean is None:
                raise ValueError(f"Could not read {cf}")

            noisy_frames.append(noisy)
            clean_frames.append(clean)

        noisy_seq = np.stack(noisy_frames, axis=0)
        clean_seq = np.stack(clean_frames, axis=0)

        # convert: [F,H,W] -> [F,3,H,W] for thermal
        #noisy_seq = np.repeat( noisy_seq[:, None, :, :], 3, axis=1)
        #clean_seq = np.repeat( clean_seq[:, None, :, :], 3, axis=1)

        #for RGB training #1 -----------------------------------------------
        #during training clean & noise sequences both point to the same clean directory
        #noise is added here; each temporal window of 5 gets a random Gaussian noise value

        noisy_seq = noisy_seq.transpose(0, 3, 1, 2)
        clean_seq = clean_seq.transpose(0, 3, 1, 2)


        sigma = random.uniform(5, 55)
        noise = np.random.randn(*clean_seq.shape) * sigma
        noisy_seq = clean_seq.astype(np.float32) + noise
        noisy_seq = np.clip(noisy_seq, 0, 255)
        #------------------------------------------------------------------

        return (torch.from_numpy(noisy_seq), torch.from_numpy(clean_seq))