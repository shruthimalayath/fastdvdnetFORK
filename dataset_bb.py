from torch.utils.data import Dataset
import glob
import os
import cv2
import torch
import numpy as np
import random
import re
import tifffile

"""
- Preloads all frames (noisy + clean) into CPU RAM at initialization using tifffile.
    *Samples a random sequence
    *chooses a temporal center and collects temp_patch_size frames around it
    *Performs a random spatial crop of size patch_size
    *Repeats the single grayscale channel to 3 channels
    *Returns (noisy, clean) as torch.FloatTensors with shapes (F, 3, H, W)
    *If blackbody_noise_map is provided, ALSO crops it at the SAME (y,x)
     location as the noisy/clean patch and returns it as a third item,
     shape (1, H, W), raw units -- keeps the noise map spatially aligned
     with the actual patch it accompanies.
"""


def _frame_numeric_key(basename: str) -> int:
    """Extract the first integer in filename for numeric sorting (fallback 0)."""
    m = re.search(r"(\d+)", basename)
    return int(m.group(1)) if m else 0


class PairedThermalDataset(Dataset):
    def __init__(
        self,
        noisy_root: str,
        clean_root: str,
        patch_size: int = 96,
        temp_patch_size: int = 5,
        epoch_size: int = 256000,
        preload: bool = True,
        progress: bool = False,
        blackbody_noise_map: np.ndarray = None,
    ):
        self.noisy_root = noisy_root
        self.clean_root = clean_root
        self.patch_size = patch_size
        self.temp_patch_size = temp_patch_size
        self.epoch_size = epoch_size

        # optional: raw-units [H,W] calibration map, cropped at the SAME
        # (y,x) as each patch so the noise map stays spatially aligned
        self.blackbody_noise_map = (
            blackbody_noise_map.astype(np.float32) if blackbody_noise_map is not None else None
        )

        # only keep directory entries (same assumption as  original code)
        entries = sorted(os.listdir(self.noisy_root))
        self.sequences = [e for e in entries if os.path.isdir(os.path.join(self.noisy_root, e))]

        # caches: seq_name -> np.array shape (F_all, H, W), dtype as read by tifffile
        self.noisy_cache = {}
        self.clean_cache = {}

        if preload:
            self._preload_all(progress=progress)

        # filter sequences that were not successfully cached
        self.sequences = [s for s in self.sequences if s in self.noisy_cache and s in self.clean_cache]
        if len(self.sequences) == 0:
            raise RuntimeError("No valid sequences found after preload. Check noisy/clean directories and filenames.")

    def __len__(self):
        return int(self.epoch_size)

    def _preload_all(self, progress: bool = False):
        """Preload every sequence into memory using tifffile.imread and print per-sequence status."""
        seq_iter = self.sequences
        if progress:
            try:
                from tqdm import tqdm
                seq_iter = tqdm(self.sequences, desc="Preloading sequences")
            except Exception:
                seq_iter = self.sequences

        skipped_no_files = []
        skipped_no_common = []
        skipped_read_error = []
        loaded = []

        for seq in seq_iter:
            noisy_dir = os.path.join(self.noisy_root, seq)
            clean_dir = os.path.join(self.clean_root, seq)

            noisy_files_all = sorted(glob.glob(os.path.join(noisy_dir, "*.tif")))
            clean_files_all = sorted(glob.glob(os.path.join(clean_dir, "*.tif")))

            if len(noisy_files_all) == 0 or len(clean_files_all) == 0:
                skipped_no_files.append(seq)
                print(f"[Preload] SKIP (no files): {seq}")
                continue

            # pair by basename intersection and numeric sort
            noisy_map = {os.path.basename(p): p for p in noisy_files_all}
            clean_map = {os.path.basename(p): p for p in clean_files_all}
            common = list(set(noisy_map.keys()) & set(clean_map.keys()))
            if len(common) == 0:
                skipped_no_common.append(seq)
                print(f"[Preload] SKIP (no common basenames): {seq} (noisy={len(noisy_files_all)}, clean={len(clean_files_all)})")
                continue

            common.sort(key=_frame_numeric_key)
            noisy_paths = [noisy_map[b] for b in common]
            clean_paths = [clean_map[b] for b in common]

            try:
                noisy_frames = [tifffile.imread(p) for p in noisy_paths]
                clean_frames = [tifffile.imread(p) for p in clean_paths]
            except Exception as e:
                skipped_read_error.append((seq, str(e)))
                print(f"[Preload] SKIP (read error): {seq} -> {e}")
                continue

            try:
                noisy_arr = np.stack(noisy_frames, axis=0)  # (F_all, H, W)
                clean_arr = np.stack(clean_frames, axis=0)
            except Exception as e:
                skipped_read_error.append((seq, str(e)))
                print(f"[Preload] SKIP (stack error): {seq} -> {e}")
                continue

            # store arrays (keep dtype as-is)
            self.noisy_cache[seq] = noisy_arr
            self.clean_cache[seq] = clean_arr
            loaded.append(seq)
            print(f"[Preload] LOADED: {seq}  frames={noisy_arr.shape[0]} dtype={noisy_arr.dtype}")

        # summary
        print(f"[Preload] Summary: loaded={len(loaded)}, skipped_no_files={len(skipped_no_files)}, skipped_no_common={len(skipped_no_common)}, skipped_errors={len(skipped_read_error)}")
        if skipped_no_files:
            print("[Preload] sequences with missing side (no files):", skipped_no_files)
        if skipped_no_common:
            print("[Preload] sequences with no common basenames:", skipped_no_common)
        if skipped_read_error:
            print("[Preload] sequences with read/stack errors:", skipped_read_error)

    def __getitem__(self, idx):

        seq = random.choice(self.sequences)

        noisy_full = self.noisy_cache.get(seq, None)
        clean_full = self.clean_cache.get(seq, None)
        if noisy_full is None or clean_full is None:
            raise ValueError(f"Sequence {seq} not loaded in cache")

        F_all = noisy_full.shape[0]
        half = self.temp_patch_size // 2

        # robust center selection: if sequence shorter than temp_patch_size, choose middle
        if F_all <= self.temp_patch_size:
            center = F_all // 2
        else:
            center = random.randint(half, F_all - 1 - half)

        # temporal indices (clamped)
        indices = list(range(center - half, center + half + 1))
        indices = [min(max(i, 0), F_all - 1) for i in indices]

        # extract window -> (F, H, W)
        noisy_seq = noisy_full[indices]
        clean_seq = clean_full[indices]

        # spatial crop (assumes H,W >= patch_size as in your original code)
        H, W = noisy_seq.shape[1], noisy_seq.shape[2]
        x = random.randint(0, W - self.patch_size)
        y = random.randint(0, H - self.patch_size)

        noisy_crop = noisy_seq[:, y : y + self.patch_size, x : x + self.patch_size]  # (F, ph, pw)
        clean_crop = clean_seq[:, y : y + self.patch_size, x : x + self.patch_size]

        # add channel dim and repeat to 3 channels (same behavior as your original)
        noisy_crop = noisy_crop[:, None, :, :]         # (F,1,H,W)
        clean_crop = clean_crop[:, None, :, :]
        noisy_crop = np.repeat(noisy_crop, 3, axis=1)  # (F,3,H,W)
        clean_crop = np.repeat(clean_crop, 3, axis=1)

        # convert to torch.FloatTensor (normalization should be done downstream)
        noisy_t = torch.from_numpy(noisy_crop).float()
        clean_t = torch.from_numpy(clean_crop).float()

        if self.blackbody_noise_map is not None:
            bb_H, bb_W = self.blackbody_noise_map.shape
            if (bb_H, bb_W) != (H, W):
                raise ValueError(
                    f"Blackbody noise map shape ({bb_H},{bb_W}) doesn't match "
                    f"sequence '{seq}' frame shape ({H},{W}). Recalibrate against "
                    f"frames of the same resolution as your training sequences."
                )
            # SAME (y,x) crop as the noisy/clean patch above -- true alignment
            bb_crop = self.blackbody_noise_map[y : y + self.patch_size, x : x + self.patch_size]  # (ph,pw)
            bb_crop = bb_crop[None, :, :].copy()  # (1,ph,pw)
            bb_t = torch.from_numpy(bb_crop).float()
            return noisy_t, clean_t, bb_t

        return noisy_t, clean_t