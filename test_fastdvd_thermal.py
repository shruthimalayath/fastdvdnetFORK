#!/usr/bin/env python3
"""
Denoise a TIFF sequence using FastDVDnet with a provided per-pixel noise map.

Usage examples:
  # per-pixel numpy map (preferred)
  python test_fastdvdnet_with_map.py --model_file ./model.pth --test_path /path/to/seq \
    --noise_map ./noise_map.npy --save_path ./out --cuda

  # image noise map (will be converted to single-channel & resized)
  python test_fastdvdnet_with_map.py --model_file ./model.pth --test_path /path/to/seq \
    --noise_map ./noise_map.png --save_path ./out --max_val 65535 --cuda

  # scalar noise map (uniform)
  python test_fastdvdnet_with_map.py --model_file ./model.pth --test_path /path/to/seq \
    --noise_sigma 25 --max_val 255 --save_path ./out --cuda
"""

#testing uses constant noise map. 

import os
import argparse
import time
import glob
import numpy as np
import cv2
import torch
import torch.nn as nn

from models import FastDVDnet
from fastdvdnet import denoise_seq_fastdvdnet
from utils_thermal import batch_psnr, init_logger_test, variable_to_cv2_image, remove_dataparallel_wrapper, open_sequence, close_logger

NUM_IN_FR_EXT = 5  # temporal size of patch
OUTIMGEXT = '.tif'  # output images format

def save_out_seq(seqnoisy, seqclean, save_dir, sigmaval, suffix, save_noisy):
    """Saves the denoised and noisy sequences under save_dir"""
    seq_len = seqnoisy.size()[0]
    os.makedirs(save_dir, exist_ok=True)
    for idx in range(seq_len):
        fext = OUTIMGEXT
        noisy_name = os.path.join(save_dir, ('n{}_{}').format(sigmaval, idx) + fext)
        if len(suffix) == 0:
            out_name = os.path.join(save_dir, ('n{}_FastDVDnet_{}').format(sigmaval, idx) + fext)
        else:
            out_name = os.path.join(save_dir, ('n{}_FastDVDnet_{}_{}').format(sigmaval, suffix, idx) + fext)

        if save_noisy:
            noisyimg = variable_to_cv2_image(seqnoisy[idx].clamp(0., 1.))
            cv2.imwrite(noisy_name, noisyimg)

        outimg = variable_to_cv2_image(seqclean[idx].unsqueeze(dim=0))
        cv2.imwrite(out_name, outimg)

def load_noise_map(path, H, W, max_val):
    """Load noise map from .npy or image. Return numpy float32 array normalized to [0,1], shape (H,W)."""
    if path is None:
        return None
    ext = os.path.splitext(path)[1].lower()
    if ext == '.npy':
        arr = np.load(path)
    else:
        # read image (cv2 BGR or grayscale)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"Could not read noise_map file {path}")
        if img.ndim == 3:
            # convert to float and average channels into single channel
            arr = img.astype(np.float32).mean(axis=2)
        else:
            arr = img.astype(np.float32)
    # normalize if necessary
    # if values appear <= 1.0 we assume already normalized; otherwise divide by max_val
    if arr.max() > 1.0:
        arr = arr.astype(np.float32) / float(max_val)
    else:
        arr = arr.astype(np.float32)
    # resize if needed
    if arr.shape != (H, W):
        arr = cv2.resize(arr, (W, H), interpolation=cv2.INTER_LINEAR)
    # ensure non-negative
    arr = np.clip(arr, 0.0, None)
    return arr

def test_fastdvdnet_with_map(**args):
    start_time = time.time()
    os.makedirs(args['save_path'], exist_ok=True)
    logger = init_logger_test(args['save_path'])

    device = torch.device('cuda') if args['cuda'] and torch.cuda.is_available() else torch.device('cpu')

    print('Loading model ...')
    model_temp = FastDVDnet(num_input_frames=NUM_IN_FR_EXT)

    #state_temp_dict = torch.load(args['model_file'], map_location=device)
    #if args['cuda'] and torch.cuda.is_available():
        #device_ids = [0]
        #model_temp = nn.DataParallel(model_temp, device_ids=device_ids).cuda()
    #else:
        #state_temp_dict = remove_dataparallel_wrapper(state_temp_dict)
    #model_temp.load_state_dict(state_temp_dict)

    #fixed dict stuff
        # Load saved weights (robust to different checkpoint formats & DataParallel)
    ckpt = torch.load(args['model_file'], map_location=device)

    # Extract the model state_dict if the file contains a checkpoint dict
    if isinstance(ckpt, dict):
        if 'state_dict' in ckpt:
            state_temp_dict = ckpt['state_dict']
        elif 'model_state_dict' in ckpt:
            state_temp_dict = ckpt['model_state_dict']
        else:
            # assume the dict is already a state_dict-like mapping
            state_temp_dict = ckpt
    else:
        state_temp_dict = ckpt

    # Wrap model for GPU if requested
    if args['cuda'] and torch.cuda.is_available():
        device_ids = [0]
        model_temp = nn.DataParallel(model_temp, device_ids=device_ids).cuda()

    # Align 'module.' prefix between saved keys and current model
    is_model_dp = isinstance(model_temp, nn.DataParallel)
    has_module_prefix = any(k.startswith('module.') for k in state_temp_dict.keys())

    if has_module_prefix and not is_model_dp:
        # strip 'module.' from checkpoint keys
        new_state = {}
        for k, v in state_temp_dict.items():
            new_state[k[len('module.'):]] = v if k.startswith('module.') else v
        state_temp_dict = new_state
    elif (not has_module_prefix) and is_model_dp:
        # add 'module.' to keys so they match DataParallel
        state_temp_dict = {'module.' + k: v for k, v in state_temp_dict.items()}

    # Finally load into the model
    model_temp.load_state_dict(state_temp_dict)



    model_temp.eval()

    # Load sequence
    seq_np, _, _ = open_sequence(args['test_path'], args['gray'], expand_if_needed=False, max_num_fr=args['max_num_fr_per_seq'])
    if seq_np is None:
        raise RuntimeError(f"Could not open sequence at {args['test_path']}")
    # seq_np expected shape: [F, C, H, W]
    F, C, H, W = seq_np.shape
    print(f"Loaded sequence: frames={F}, channels={C}, H={H}, W={W}")

    # normalize to [0,1]
    max_val = float(args['max_val'])
    seq_np = seq_np.astype(np.float32) / max_val

    # Build tensors
    seq = torch.from_numpy(seq_np).to(device)  # ground-truth or input depending on use
    # In this script we assume the input sequence is already the noisy sequence (test_path is noisy)
    seqn = seq.clone()

    # --- Hard-coded constant noise map (replace existing preparation block) ---
    # Option A - specify sigma in original image units (e.g. 25 for 8-bit, or 2000 for 14-bit)
    CONST_SIGMA_ORIG = 25.0        # <-- change this to the numeric sigma you want
    sigma_norm = float(CONST_SIGMA_ORIG) / float(max_val)

    # Option B - or specify normalized sigma directly (uncomment if you prefer)
    # sigma_norm = 0.02            # <-- normalized in [0,1], e.g. 0.02

    # Make a per-pixel constant noise map (H x W) and convert to tensor [1,1,H,W]
    nm_np = np.full((H, W), float(sigma_norm), dtype=np.float32)
    noise_map_t = torch.from_numpy(nm_np).unsqueeze(0).unsqueeze(0).to(device)
    noise_map_t = noise_map_t.clamp(min=1e-6)
    print(f"Using hard-coded constant noise map: sigma_norm={sigma_norm:.6e}")
    # ------------------------------------------------------------------------

    # Run denoiser
    with torch.no_grad():
        t0 = time.time()
        denframes = denoise_seq_fastdvdnet(seq=seqn, noise_std=noise_map_t, temp_psz=NUM_IN_FR_EXT, model_temporal=model_temp)
        t1 = time.time()
        denoise_time = (t1 - t0)

    # Compute PSNR if a clean reference is available (script treats seq as noisy by default)
    # If user provided --clean_seq, compute PSNR against that; else skip
    if args.get('clean_seq', None):
        seq_clean_np, _, _ = open_sequence(args['clean_seq'], args['gray'], expand_if_needed=False, max_num_fr=args['max_num_fr_per_seq'])
        if seq_clean_np is not None:
            seq_clean_np = seq_clean_np.astype(np.float32) / max_val
            seq_clean = torch.from_numpy(seq_clean_np).to(device)
            psnr_deno = batch_psnr(denframes.cpu(), seq_clean.cpu(), 1.0)
            psnr_noisy = batch_psnr(seqn.cpu(), seq_clean.cpu(), 1.0)
            print(f"PSNR noisy={psnr_noisy:.4f}, PSNR denoised={psnr_deno:.4f}, denoise_time={denoise_time:.3f}s")
        else:
            print("No clean sequence found or could not open; skipping PSNR")
    else:
        print(f"Denoise finished, time={denoise_time:.3f}s (no clean reference provided)")

    # Save outputs
    if not args['dont_save_results']:
        #label = args['label'] if args['label'] else ('map' if args['noise_map'] else f"s{int(args['noise_sigma'])}")
        label = args.get('label') or f"s{int(CONST_SIGMA_ORIG)}"
        save_out_seq(seqn, denframes, args['save_path'], label, args['suffix'], args['save_noisy'])

    close_logger(logger)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Denoise a sequence with FastDVDnet using a provided noise map")
    parser.add_argument("--model_file", type=str, default="./model.pth", help='path to pretrained denoiser')
    parser.add_argument("--test_path", type=str, required=True, help='path to sequence to denoise (folder of tiffs)')
    parser.add_argument("--clean_seq", type=str, default=None, help='optional clean sequence folder (for PSNR measurement)')
    #parser.add_argument("--noise_map", type=str, default=None, help='path to per-pixel noise map (.npy or image). normalized by --max_val')
    #parser.add_argument("--noise_sigma", type=float, default=None, help='scalar noise sigma (in same units as --max_val). If provided, overrides noise_map')
    parser.add_argument("--label", type=str, default="", help='label to use in saved filenames instead of "map"')
    parser.add_argument("--suffix", type=str, default="", help='suffix to add to output name')
    parser.add_argument("--max_num_fr_per_seq", type=int, default=20, help='max number of frames to load per sequence')
    parser.add_argument("--dont_save_results", action='store_true', help="don't save output images")
    parser.add_argument("--save_noisy", action='store_true', help="save noisy frames")
    parser.add_argument("--no_gpu", action='store_true', help="run model on CPU")
    parser.add_argument("--save_path", type=str, default='./results', help='where to save outputs')
    parser.add_argument("--gray", action='store_true', help='treat input images as grayscale')
    parser.add_argument("--max_val", type=float, default=65535.0, help='max pixel value for normalization (use 255 for 8-bit)')
    argspar = parser.parse_args()

    argspar.cuda = not argspar.no_gpu and torch.cuda.is_available()

    print("\n### Testing FastDVDnet model (with provided noise map) ###")
    for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
        print('\t{}: {}'.format(p, v))
    print('\n')

    test_fastdvdnet_with_map(**vars(argspar))