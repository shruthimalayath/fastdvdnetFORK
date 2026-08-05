import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset_paired import PairedThermalDataset  # or your dataset class
from utils import normalize_augment_pair
from random import choices



def normalize_augment_pair(noisy_seq, clean_seq, ctrl_fr_idx):
    """
    Args:
        noisy_seq: [N, num_frames, C, H, W]
        clean_seq: [N, num_frames, C, H, W]

    Returns:
        imgn_train: [N, num_frames*C, H, W]
        gt_train:   [N, 3, H, W]
    """

    def get_transform():
        aug_list = [
            lambda x: x,
            lambda x: torch.flip(x, dims=[2]),
            lambda x: torch.rot90(x, k=1, dims=[2, 3]),
            lambda x: torch.flip(torch.rot90(x, k=1, dims=[2, 3]), dims=[2]),
            lambda x: torch.rot90(x, k=2, dims=[2, 3]),
            lambda x: torch.flip(torch.rot90(x, k=2, dims=[2, 3]), dims=[2]),
            lambda x: torch.rot90(x, k=3, dims=[2, 3]),
            lambda x: torch.flip(torch.rot90(x, k=3, dims=[2, 3]), dims=[2]),
        ]

        w_aug = [32, 12, 12, 12, 12, 12, 12, 12]
        return choices(aug_list, w_aug)[0]

    # normalize

    #noisy_seq = noisy_seq.float() / 255.
    #clean_seq = clean_seq.float() / 255.

    noisy_seq = noisy_seq.reshape(noisy_seq.size(0), -1, noisy_seq.size(-2), noisy_seq.size(-1))
    clean_seq = clean_seq.reshape(clean_seq.size(0), -1, clean_seq.size(-2), clean_seq.size(-1))

    # choose one augmentation
    transf = get_transform()
    print("Augmentation chosen:", getattr(transf, "__name__", str(transf)))

    # apply same augmentation to both
    noisy_seq = transf(noisy_seq)
    clean_seq = transf(clean_seq)

    gt_train = clean_seq[ :,3*ctrl_fr_idx:3*ctrl_fr_idx+3,:,:]

    return noisy_seq, gt_train


OUT_DIR = "pair_aug_check"
os.makedirs(OUT_DIR, exist_ok=True)

# instantiate dataset (use your real paths)
dataset = PairedThermalDataset(
    noisy_root="/mnt/data/users/smalayath/fastdvdnetFORK/DAVIS_test_root/DAVIS_clean_test",
    clean_root="/mnt/data/users/smalayath/fastdvdnetFORK/DAVIS_test_root/DAVIS_clean_test",
)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

# take first sample
noisy_seq, clean_seq, sigma = next(iter(loader))  # shapes: [N=1, F, C, H, W] and values 0..255
print("loaded shapes:", noisy_seq.shape, clean_seq.shape, "sigma:", sigma)

# optionally insert a visible square into noisy central frame to make comparing easier
N, F, C, H, W = noisy_seq.shape
ctrl = F // 2
noisy_seq_vis = noisy_seq.clone()
sq = max(8, min(20, H//10, W//10))
top = H//2 - sq//2
left = W//2 - sq//2
# add bright patch to red channel of central frame (pixel units)
noisy_seq_vis[0, ctrl, 0, top:top+sq, left:left+sq] = (noisy_seq_vis[0, ctrl, 0, top:top+sq, left:left+sq].float() + 80).clamp(0,255).byte()

# call normalize_augment_pair on the pair you want to test.
# NOTE: the function expects [N, F, C, H, W] uint8-ish (it may normalize internally if uncommented)
imgn_train, gt_train = normalize_augment_pair(noisy_seq_vis, clean_seq, ctrl_fr_idx=ctrl)

# imgn_train shape: [N, F*C, H, W]
# extract augmented center frames (channels flattened)
start_ch = 3 * ctrl
aug_noisy_center = imgn_train[0, start_ch:start_ch+3, :, :].cpu().numpy()  # shape (3,H,W)
aug_clean_center = gt_train[0].cpu().numpy()                               # shape (3,H,W)

# convert to HxWxC uint8 for saving (if normalize_augment_pair did not normalize, values already 0..255)
def to_bgr_uint8(chw):
    hwc = np.transpose(chw, (1,2,0))
    hwc = np.clip(hwc, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hwc, cv2.COLOR_RGB2BGR)

# save originals (center frames before augmentation) for visual comparison
orig_noisy = noisy_seq_vis[0, ctrl].cpu().numpy()
orig_clean = clean_seq[0, ctrl].cpu().numpy()
cv2.imwrite(os.path.join(OUT_DIR, "orig_noisy_center.png"), to_bgr_uint8(orig_noisy))
cv2.imwrite(os.path.join(OUT_DIR, "orig_clean_center.png"), to_bgr_uint8(orig_clean))

cv2.imwrite(os.path.join(OUT_DIR, "aug_noisy_center.png"), to_bgr_uint8(aug_noisy_center))
cv2.imwrite(os.path.join(OUT_DIR, "aug_clean_center.png"), to_bgr_uint8(aug_clean_center))

# print small numeric patches (red channel) for quick numeric comparison
def print_patch(arr, name, top=top, left=left, h=5, w=5):
    if arr.shape[0] == 3:
        patch = arr[0, top:top+h, left:left+w]
    else:
        patch = arr[top:top+h, left:left+w, 0]
    print(f"{name} patch:\n", patch.astype(int))

print_patch(orig_noisy, "orig_noisy_center")
print_patch(orig_clean, "orig_clean_center")
print_patch(aug_noisy_center, "aug_noisy_center")
print_patch(aug_clean_center, "aug_clean_center")

print("Saved images to", OUT_DIR)