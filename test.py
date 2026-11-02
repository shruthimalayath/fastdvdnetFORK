#!/usr/bin/env python3
"""
Denoise multiple thermal sequences with FastDVDnet, matching training-time
normalization exactly (--norm_mode / --noise_map_mode mirror the training
script's toggles).

Expects:
  --test_noisy_dir/
      seq_001/  frame_0000.tif, frame_0001.tif, ...
      seq_002/  ...
  --test_clean_dir/
      seq_001/  frame_0000.tif, frame_0001.tif, ...   (same names, same length)
      seq_002/  ...

Reports, per sequence and averaged overall:
  - PSNR(noisy, clean)      <- baseline, no model involved
  - PSNR(denoised, clean)   <- your model's result
  - Delta (denoised - noisy) <- how much the model actually helped

Both PSNRs are computed on raw 16-bit values with a fixed data_range=65535,
so numbers are physically meaningful and comparable across sequences
regardless of which --norm_mode you're testing.
"""
import os
import glob
import argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

from models import FastDVDnet
from utils_thermal import (
    init_logger_test, close_logger, load_raw_sequence,
    minmax_normalize, minmax_denormalize, get_const_noise_sigma_norm,
)

NUM_IN_FR_EXT = 5
OUTIMGEXT = '.tif'


def pad_indices(idx, F, radius):
    inds = []
    for d in range(-radius, radius + 1):
        j = idx + d
        if j < 0:
            j = 0
        elif j >= F:
            j = F - 1
        inds.append(j)
    return inds


def robust_load_state_dict(model, ckpt_path, device, use_cuda):
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict):
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        elif 'model_state_dict' in ckpt:
            state_dict = ckpt['model_state_dict']
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    if use_cuda and torch.cuda.is_available():
        model = nn.DataParallel(model, device_ids=[0]).cuda()

    is_dp = isinstance(model, nn.DataParallel)
    has_module = any(k.startswith('module.') for k in state_dict.keys())

    if has_module and not is_dp:
        state_dict = {k[len('module.'):] if k.startswith('module.') else k: v
                      for k, v in state_dict.items()}
    elif (not has_module) and is_dp:
        state_dict = {'module.' + k: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    return model


def find_matching_sequences(noisy_dir, clean_dir=None):
    """Return sorted sequence folders. If clean_dir is omitted, use noisy_dir only."""
    noisy_seqs = {d for d in os.listdir(noisy_dir)
                 if os.path.isdir(os.path.join(noisy_dir, d))}

    if clean_dir is None:
        return sorted(noisy_seqs)

    clean_seqs = {d for d in os.listdir(clean_dir)
                  if os.path.isdir(os.path.join(clean_dir, d))}
    common = sorted(noisy_seqs & clean_seqs)
    missing_in_clean = sorted(noisy_seqs - clean_seqs)
    missing_in_noisy = sorted(clean_seqs - noisy_seqs)
    if missing_in_clean:
        print(f"WARNING: sequences in noisy dir but not clean dir (skipped): {missing_in_clean}")
    if missing_in_noisy:
        print(f"WARNING: sequences in clean dir but not noisy dir (skipped): {missing_in_noisy}")
    return common


def denoise_sequence(model, seqn_raw, seqc_raw, args, device):
    """
    seqn_raw, seqc_raw: numpy [F, 3, H, W] raw (unnormalized) float32
    Returns: recon_raw [F, 3, H, W] numpy float32 (raw units), and per-frame
             normalized-domain outputs used only for optional preview.
    """
    if args.noise_map_mode == 'residual' and seqc_raw is None:
        raise ValueError(
            "--noise_map_mode residual requires a clean sequence. "
            "Either provide --test_clean_dir or switch to --noise_map_mode constant."
        )

    F, C, H, W = seqn_raw.shape
    radius = (NUM_IN_FR_EXT - 1) // 2

    seqn_t = torch.from_numpy(seqn_raw).to(device)
    seqc_t = None if seqc_raw is None else torch.from_numpy(seqc_raw).to(device)

    recon_raw = np.zeros_like(seqn_raw)

    with torch.no_grad():
        for t in range(F):
            inds = pad_indices(t, F, radius)
            window_noisy = torch.stack([seqn_t[i] for i in inds], dim=0)  # [T,C,H,W]
            window_noisy = window_noisy.unsqueeze(0)  # [1,T,C,H,W]

            # --- normalize this window (noisy-only stats, as real inference must) ---
            if args.norm_mode == 'minmax':
                window_norm, minv, maxv = minmax_normalize(window_noisy, return_stats=True)
            elif args.norm_mode == 'fixed':
                window_norm = window_noisy / args.max_val
                minv, maxv = None, None
            else:
                raise ValueError(f"Unknown norm_mode: {args.norm_mode}")

            imgn = window_norm.reshape(1, -1, H, W).to(device=device, dtype=torch.float32)

            # --- noise map ---
            ctrl_ch = radius * C
            noisy_center_norm = imgn[:, ctrl_ch:ctrl_ch + 1, :, :]  # [1,1,H,W]

            if args.noise_map_mode == 'residual':
                # Oracle-style eval: uses clean reference to build the same kind
                # of residual noise map training used. Not available at true
                # blind deployment, but valid for evaluating a checkpoint
                # trained with --noise_map_mode residual.
                clean_center_raw = seqc_t[t][:1, :, :].unsqueeze(0)  # [1,1,H,W] raw
                if args.norm_mode == 'minmax':
                    clean_center_norm = (clean_center_raw - minv[:, 0, 0, :, :] if False else clean_center_raw)
                    # normalize clean center using the SAME window min/max as noisy
                    minv0 = minv.squeeze()  # scalar-ish per-sample
                    maxv0 = maxv.squeeze()
                    clean_center_norm = (clean_center_raw - minv0) / (maxv0 - minv0 + 1e-6)
                else:
                    clean_center_norm = clean_center_raw / args.max_val
                residual = (noisy_center_norm - clean_center_norm).abs()
                noise_map = residual.clamp(min=1e-6)
            elif args.noise_map_mode == 'constant':
                sigma_norm = get_const_noise_sigma_norm(args.norm_mode, args.const_sigma, args.max_val)
                noise_map = imgn.new_full((1, 1, H, W), float(sigma_norm)).clamp(min=1e-6)
            else:
                raise ValueError(f"Unknown noise_map_mode: {args.noise_map_mode}")

            out = model(imgn, noise_map)  # [1, T*C or C, H, W]
            num_out_ch = out.shape[1]
            if num_out_ch == NUM_IN_FR_EXT * C:
                start_ch = radius * C
                out_center = out[0, start_ch:start_ch + C, :, :]
            elif num_out_ch == C:
                out_center = out[0]
            else:
                start_ch = max(0, (num_out_ch - C) // 2)
                out_center = out[0, start_ch:start_ch + C, :, :]

            # --- denormalize back to raw units using THIS window's stats ---
            out_center = out_center.unsqueeze(0)  # [1,C,H,W]
            if args.norm_mode == 'minmax':
                minv0 = minv.squeeze()
                maxv0 = maxv.squeeze()
                out_raw = out_center * (maxv0 - minv0 + 1e-6) + minv0
            else:
                out_raw = out_center * args.max_val

            recon_raw[t] = out_raw.squeeze(0).cpu().numpy()

    return recon_raw


def compute_psnr_raw(a_raw, b_raw, data_range=65535.0):
    F = a_raw.shape[0]
    total = 0.0

    for i in range(F):
        a_hwc = np.transpose(a_raw[i], (1, 2, 0))
        b_hwc = np.transpose(b_raw[i], (1, 2, 0))

        psnr = compare_psnr(b_hwc, a_hwc, data_range=data_range)
        total += psnr

    return total / F


def save_sequence_outputs(recon_raw, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for i in range(recon_raw.shape[0]):
        frame = recon_raw[i, 0]  # single-channel (grayscale repeated to 3, take ch0)
        frame16 = np.clip(np.rint(frame), 0, 65535).astype(np.uint16)
        cv2.imwrite(os.path.join(out_dir, f"frame_{i:04d}{OUTIMGEXT}"), frame16)


def main(args):
    os.makedirs(args.save_path, exist_ok=True)
    logger = init_logger_test(args.save_path)
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')

    clean_dir = args.test_clean_dir.strip() if args.test_clean_dir else None

    print('Loading model ...')
    model = FastDVDnet(num_input_frames=NUM_IN_FR_EXT)
    model = robust_load_state_dict(model, args.model_file, device, args.cuda)
    model = model.to(device)
    model.eval()

    seq_names = find_matching_sequences(args.test_noisy_dir, clean_dir)
    if len(seq_names) == 0:
        if clean_dir is None:
            raise RuntimeError("No sequence subfolders found in noisy dir.")
        raise RuntimeError("No matching sequence subfolders found in both noisy and clean dirs.")
    print(f"Found {len(seq_names)} sequences: {seq_names}")

    baseline_psnrs = []
    denoised_psnrs = []

    for seq_name in seq_names:
        noisy_dir = os.path.join(args.test_noisy_dir, seq_name)
        seqn_raw = load_raw_sequence(noisy_dir, max_num_fr=args.max_num_fr_per_seq)
        if seqn_raw is None:
            print(f"  [{seq_name}] SKIPPED: couldn't load .tif frames")
            continue

        if clean_dir is None:
            recon_raw = denoise_sequence(model, seqn_raw, None, args, device)
            print(f"  [{seq_name}] denoised without clean reference; PSNR skipped.")
            logger.info(f"[{seq_name}] denoised_only")
            if args.save_frames:
                save_sequence_outputs(recon_raw, os.path.join(args.save_path, seq_name))
            continue

        clean_dir_seq = os.path.join(clean_dir, seq_name)
        seqc_raw = load_raw_sequence(clean_dir_seq, max_num_fr=args.max_num_fr_per_seq)

        if seqc_raw is None:
            print(f"  [{seq_name}] SKIPPED: couldn't load clean .tif frames")
            continue
        if seqn_raw.shape != seqc_raw.shape:
            print(f"  [{seq_name}] SKIPPED: shape mismatch noisy={seqn_raw.shape} clean={seqc_raw.shape}")
            continue

        baseline = compute_psnr_raw(seqn_raw, seqc_raw, data_range=65535.0)

        recon_raw = denoise_sequence(model, seqn_raw, seqc_raw, args, device)
        denoised = compute_psnr_raw(recon_raw, seqc_raw, data_range=65535.0)

        delta = denoised - baseline
        print(f"  [{seq_name}] PSNR noisy-vs-clean: {baseline:.3f} dB | "
              f"denoised-vs-clean: {denoised:.3f} dB | delta: {delta:+.3f} dB")
        logger.info(f"[{seq_name}] baseline={baseline:.4f} denoised={denoised:.4f} delta={delta:+.4f}")

        baseline_psnrs.append(baseline)
        denoised_psnrs.append(denoised)

        if args.save_frames:
            save_sequence_outputs(recon_raw, os.path.join(args.save_path, seq_name))

    if baseline_psnrs:
        avg_baseline = float(np.mean(baseline_psnrs))
        avg_denoised = float(np.mean(denoised_psnrs))
        avg_delta = avg_denoised - avg_baseline
        print("\n=== Overall (averaged across sequences) ===")
        print(f"  PSNR noisy-vs-clean (baseline): {avg_baseline:.3f} dB")
        print(f"  PSNR denoised-vs-clean (model): {avg_denoised:.3f} dB")
        print(f"  Improvement (delta):            {avg_delta:+.3f} dB")
        logger.info(f"OVERALL baseline={avg_baseline:.4f} denoised={avg_denoised:.4f} delta={avg_delta:+.4f}")
    elif clean_dir is None:
        print("\nNo clean directory supplied; PSNR metrics were skipped.")
    else:
        print("No sequences were successfully processed.")

    close_logger(logger)
    print("\nDone. Outputs saved to", args.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test FastDVDnet on multiple paired thermal sequences, "
                    "with a noisy-vs-clean PSNR baseline for comparison.")
    parser.add_argument("--model_file", type=str, required=True,
                        help="path to checkpoint (.pth)")
    parser.add_argument("--test_noisy_dir", type=str, required=True,
                        help="parent dir containing noisy sequence subfolders")
    parser.add_argument("--test_clean_dir", type=str, default=None,
                        help="optional parent dir containing clean sequence subfolders (same names); "
                             "omit to denoise without PSNR evaluation")
    parser.add_argument("--save_path", type=str, default="./results")
    parser.add_argument("--max_num_fr_per_seq", type=int, default=200)
    parser.add_argument("--no_gpu", action='store_true')
    parser.add_argument("--save_frames", action='store_true',
                        help="save denoised .tif frames per sequence (off by default, just reports PSNR)")

    # Must match the checkpoint's training settings exactly
    parser.add_argument("--norm_mode", type=str, default="minmax",
                        choices=["minmax", "fixed"],
                        help="must match --norm_mode used to train this checkpoint")
    parser.add_argument("--noise_map_mode", type=str, default="residual",
                        choices=["residual", "constant"],
                        help="must match --noise_map_mode used to train this checkpoint")
    parser.add_argument("--const_sigma", type=float, default=25.0)
    parser.add_argument("--max_val", type=float, default=65535.0)

    args = parser.parse_args()
    args.cuda = not args.no_gpu and torch.cuda.is_available()

    print("> Parameters:")
    for k, v in vars(args).items():
        print(f"\t{k}: {v}")

    main(args)