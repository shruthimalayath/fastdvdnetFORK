#!/usr/bin/env python3
"""
Denoise a TIFF sequence using FastDVDnet, with optional per-frame min-max normalization
performed on the raw 16-bit scale (undoes open_sequence's /65535 first).

Run with --minmax to enable per-frame min-max on raw scale (experimental).
"""
import os
import argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
import glob
import tifffile

from models import FastDVDnet
from utils_thermal import variable_to_cv2_image, init_logger_test, open_sequence, close_logger, load_raw_sequence, minmax_normalize_pair

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
        new_state = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state[k[len('module.'):]] = v
            else:
                new_state[k] = v
        state_dict = new_state
    elif (not has_module) and is_dp:
        state_dict = {'module.' + k: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    return model


def save_single_channel_uint16(arr01, out_path):
    """arr01: numpy array in [0,1], shape (H,W) or (C,H,W) with C>=1 -> use first channel."""
    if arr01.ndim == 3:
        img = arr01[0]
    else:
        img = arr01
    img = np.clip(img, 0.0, 1.0)
    img16 = np.rint(img * 65535.0).astype(np.uint16)
    cv2.imwrite(out_path, img16)


def save_preview_png(frame_tensor, out_path):
    """Save a contrast-stretched 8-bit preview PNG for quick visual inspection."""
    a = frame_tensor.cpu().numpy()
    if a.ndim == 3:
        a = np.transpose(a, (1, 2, 0))
    mn, mx = float(a.min()), float(a.max())
    if mx - mn < 1e-12:
        vis = (np.clip(a, 0, 1) * 255.0).astype(np.uint8)
    else:
        vis = ((a - mn) / (mx - mn) * 255.0).astype(np.uint8)
    if vis.ndim == 3:
        cv2.imwrite(out_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    else:
        cv2.imwrite(out_path, vis)


def main(**args):
    os.makedirs(args['save_path'], exist_ok=True)
    logger = init_logger_test(args['save_path'])
    device = torch.device('cuda') if args['cuda'] and torch.cuda.is_available() else torch.device('cpu')

    print('Loading model ...')
    model = FastDVDnet(num_input_frames=NUM_IN_FR_EXT)
    model = robust_load_state_dict(model, args['model_file'], device, args['cuda'])
    try:
        model = model.to(device)
    except Exception:
        pass
    model.eval()

    # open_sequence/open_image normalize by dividing by 65535.
    #seqn_np, _, _ = open_sequence(args['test_noisy_path'], args['gray'], expand_if_needed=False,
                                  #max_num_fr=args['max_num_fr_per_seq'])
    #seqc_np, _, _ = open_sequence(args['test_clean_path'], args['gray'], expand_if_needed=False,
                                  #max_num_fr=args['max_num_fr_per_seq'])
    #if seqn_np is None or seqc_np is None:
        #raise RuntimeError("Could not open noisy or clean sequence paths")
    #if seqn_np.shape != seqc_np.shape:
        #raise RuntimeError(f"Noisy and clean sequences must have same shape. noisy:{seqn_np.shape} clean:{seqc_np.shape}")

    #F, C, H, W = seqn_np.shape
    #print(f"Loaded sequences: frames={F}, channels={C}, H={H}, W={W}")


    seqn_np = load_raw_sequence(args['test_noisy_path'], args['max_num_fr_per_seq'])
    seqc_np = load_raw_sequence(args['test_clean_path'], args['max_num_fr_per_seq'])

    if seqn_np is None or seqc_np is None:
        raise RuntimeError("Could not open noisy or clean sequence paths")
    if seqn_np.shape != seqc_np.shape:
        raise RuntimeError(f"Noisy and clean sequences must have same shape. noisy:{seqn_np.shape} clean:{seqc_np.shape}")

    F, C, H, W = seqn_np.shape
    print(f"Loaded sequences: frames={F}, channels={C}, H={H}, W={W}")

    seqn = torch.from_numpy(seqn_np).float()
    seqc = torch.from_numpy(seqc_np).float()

    # Use the same shared per-sample normalization as validation/training
    seqn, seqc, _, _ = minmax_normalize_pair(seqn, seqc)

    seqn = seqn.to(device)
    seqc = seqc.to(device)

    #use_minmax = args['minmax']
    #max_val = float(args['max_val'])

    # convert to float arrays (open_sequence returned normalized floats in [0,1])
    #seqn_np = seqn_np.astype(np.float32)
    #seqc_np = seqc_np.astype(np.float32)

    #print(f"DEBUG before scaling: noisy min={float(seqn_np.min()):.6e} max={float(seqn_np.max()):.6e}, "
          #f"clean min={float(seqc_np.min()):.6e} max={float(seqc_np.max()):.6e}")

    #if use_minmax:
        # Undo open_sequence normalization to recover approx raw 16-bit scale
        #seqn_raw = seqn_np * max_val
        #seqc_raw = seqc_np * max_val

        #print("Using PER-FRAME min-max normalization on RAW scale (undoing /65535 first).")
        #for i in range(F):
            #mn = float(seqn_raw[i].min())
            #mx = float(seqn_raw[i].max())
            #if mx - mn < 1e-12:
                #seqn_np[i] = np.clip(seqn_raw[i] - mn, 0.0, None)
                #seqc_np[i] = np.clip(seqc_raw[i] - mn, 0.0, None)
            #else:
                #seqn_np[i] = (seqn_raw[i] - mn) / (mx - mn)
                #seqc_np[i] = (seqc_raw[i] - mn) / (mx - mn)

            # Print raw min/max and normalized min/max for this frame
            #print(f"Frame {i:04d}: raw_min={mn:.6f}, raw_max={mx:.6f} -> norm_min={float(seqn_np[i].min()):.6e}, norm_max={float(seqn_np[i].max()):.6e}")

        #seqn_np = np.clip(seqn_np, 0.0, 1.0)
        #seqc_np = np.clip(seqc_np, 0.0, 1.0)
    #else:
        # keep open_sequence normalization (already /max_val)
        #seqn_np = np.clip(seqn_np, 0.0, 1.0)
        #seqc_np = np.clip(seqc_np, 0.0, 1.0)

    #print(f"DEBUG after scaling (sequence): noisy min={float(seqn_np.min()):.6e} max={float(seqn_np.max()):.6e}, "
          #f"clean min={float(seqc_np.min()):.6e} max={float(seqc_np.max()):.6e}")

    # move to device
    #seqn = torch.from_numpy(seqn_np).to(device)
    #seqc = torch.from_numpy(seqc_np).to(device)

    radius = (NUM_IN_FR_EXT - 1) // 2
    denoised_frames = []

    # run model
    with torch.no_grad():
        for t in range(F):
            try:
                inds = pad_indices(t, F, radius)
                patch_frames = torch.stack([seqn[i] for i in inds], dim=0)  # [T, C, H, W]
                T = patch_frames.shape[0]
                patch_flat = patch_frames.permute(1, 0, 2, 3).contiguous().view(1, T * C, H, W)

                noisy_center = seqn[t]
                clean_center = seqc[t]
                noisy_center_single = noisy_center[:1, :, :].unsqueeze(0)
                clean_center_single = clean_center[:1, :, :].unsqueeze(0)
                residual = (noisy_center_single - clean_center_single).abs()
                noise_map = residual.clamp(min=1e-6).to(dtype=patch_flat.dtype, device=patch_flat.device)

                out_patch = model(patch_flat, noise_map)
                num_out_ch = out_patch.shape[1]
                if num_out_ch == T * C:
                    start_ch = radius * C
                    out_center = out_patch[0, start_ch:start_ch + C, :, :]
                elif num_out_ch == C:
                    out_center = out_patch[0]
                elif num_out_ch >= C:
                    start_ch = max(0, (num_out_ch - C) // 2)
                    out_center = out_patch[0, start_ch:start_ch + C, :, :]
                else:
                    if num_out_ch == 1 and C == 3:
                        single = out_patch[0, 0:1, :, :]
                        out_center = single.repeat(3, 1, 1)
                    else:
                        raise RuntimeError(f"Unexpected number of output channels from model: {num_out_ch} (expected {C} or {T*C})")

                denoised_frames.append(out_center.cpu())
            except Exception as e:
                print(f"ERROR during model forward at frame {t}: {e}")
                raise

    # stack and reconstruct
    if len(denoised_frames) == 0:
        raise RuntimeError("No denoised frames were produced (denoised_frames is empty).")

    denoised = torch.stack(denoised_frames, dim=0)  # [F, C, H, W]
    seqn_cpu = seqn.cpu()
    seqc_cpu = seqc.cpu()

    den_min = float(denoised.min().item())
    den_max = float(denoised.max().item())
    seqn_min = float(seqn_cpu.min().item())
    seqn_max = float(seqn_cpu.max().item())
    print(f"DEBUG denoised range: min={den_min:.6e}, max={den_max:.6e}")
    print(f"DEBUG noisy range:    min={seqn_min:.6e}, max={seqn_max:.6e}")

    print("Using model output directly as the denoised image.")
    recon = denoised.clamp(0., 1.)

    # compute PSNR and save previews
    from skimage.metrics import peak_signal_noise_ratio as compare_psnr
    psnr_sum = 0.0
    first_report = True
    for i in range(F):
        den_arr = recon[i].numpy().astype(np.float32)
        den_hwc = np.transpose(den_arr, (1, 2, 0))
        clean_arr = seqc_cpu[i].numpy().astype(np.float32)
        clean_hwc = np.transpose(clean_arr, (1, 2, 0))

        if den_hwc.shape[2] != clean_hwc.shape[2]:
            if den_hwc.shape[2] == 1 and clean_hwc.shape[2] == 3:
                den_hwc = np.repeat(den_hwc, 3, axis=2)
            elif den_hwc.shape[2] == 3 and clean_hwc.shape[2] == 1:
                clean_hwc = np.repeat(clean_hwc, 3, axis=2)
            else:
                min_c = min(den_hwc.shape[2], clean_hwc.shape[2])
                den_hwc = den_hwc[:, :, :min_c]
                clean_hwc = clean_hwc[:, :, :min_c]

        h_den, w_den = den_hwc.shape[:2]
        h_cln, w_cln = clean_hwc.shape[:2]
        if (h_den != h_cln) or (w_den != w_cln):
            if first_report:
                print(f"Warning: frame {i} shape mismatch: denoised={den_hwc.shape}, clean={clean_hwc.shape}; cropping to common area for PSNR.")
                first_report = False
            h_min = min(h_den, h_cln)
            w_min = min(w_den, w_cln)
            den_crop = den_hwc[:h_min, :w_min, :]
            clean_crop = clean_hwc[:h_min, :w_min, :]
            psnr_sum += compare_psnr(clean_crop, den_crop, data_range=1.0)
        else:
            psnr_sum += compare_psnr(clean_hwc, den_hwc, data_range=1.0)

        # save a few preview PNGs
        if i < 4:
            preview_path = os.path.join(args['save_path'], f"preview_frame_{i:04d}.png")
            save_preview_png(torch.from_numpy(np.transpose(den_hwc, (2, 0, 1))), preview_path)

    psnr_avg = psnr_sum / F
    print(f"Average PSNR over sequence (reconstructed/used): {psnr_avg:.4f} dB")

    # Save outputs and print uint16 min/max per file
    for i in range(F):
        out_name = os.path.join(args['save_path'], f"frame_{i:04d}{OUTIMGEXT}")
        t = recon[i]               # tensor in [0,1], shape [C,H,W]
        t_np = t.numpy()

        # Build uint16 buffer same as saver
        if t_np.ndim == 3 and t_np.shape[0] >= 3:
            arr16 = np.rint(np.clip(np.transpose(t_np, (1, 2, 0)), 0.0, 1.0) * 65535.0).astype(np.uint16)
        elif t_np.ndim == 3 and t_np.shape[0] == 1:
            arr16 = np.rint(np.clip(t_np[0], 0.0, 1.0) * 65535.0).astype(np.uint16)
        else:
            arr16 = np.rint(np.clip(t_np, 0.0, 1.0) * 65535.0).astype(np.uint16)

        # Print per-file stats
        if arr16.ndim == 3:
            ch_stats = ", ".join([f"ch{c}:min={int(arr16[:,:,c].min())} max={int(arr16[:,:,c].max())}" for c in range(arr16.shape[2])])
            print(f"Saved {out_name}: shape={arr16.shape}, dtype={arr16.dtype}, {ch_stats}")
        else:
            print(f"Saved {out_name}: shape={arr16.shape}, dtype={arr16.dtype}, min={int(arr16.min())} max={int(arr16.max())}")

        # write file consistently with earlier logic
        save_as_single = False
        if t_np.ndim == 3 and t_np.shape[0] >= 3:
            if np.allclose(t_np[0], t_np[1], atol=1e-6) and np.allclose(t_np[0], t_np[2], atol=1e-6):
                save_as_single = True
        elif t_np.ndim == 2 or (t_np.ndim == 3 and t_np.shape[0] == 1):
            save_as_single = True

        if args['gray'] or save_as_single:
            # write first channel as single-channel uint16
            arr_write = arr16[:, :, 0] if arr16.ndim == 3 else arr16
            cv2.imwrite(out_name, arr_write)
        else:
            np_img = variable_to_cv2_image(t.clamp(0., 1.))
            cv2.imwrite(out_name, np_img)

    close_logger(logger)
    print("Done. Outputs saved to", args['save_path'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Denoise a sequence with FastDVDnet using residual noise map from clean/noisy pairs")
    parser.add_argument("--model_file", type=str, required=True, help='path to pretrained denoiser')
    parser.add_argument("--test_noisy_path", type=str, required=True, help='path to noisy sequence (folder)')
    parser.add_argument("--test_clean_path", type=str, required=True, help='path to clean sequence (folder)')
    parser.add_argument("--save_path", type=str, default='./results', help='where to save outputs')
    parser.add_argument("--max_num_fr_per_seq", type=int, default=200, help='max number of frames to load per sequence')
    parser.add_argument("--no_gpu", action='store_true', help="run on CPU")
    parser.add_argument("--gray", action='store_true', help='treat input images as grayscale (open_image will stack into 3 channels)')
    parser.add_argument("--max_val", type=float, default=65535.0, help='max pixel value used to normalize input (e.g. 65535 for 16-bit)')
    parser.add_argument("--minmax", action='store_true', help='use per-frame min-max normalization on RAW scale (undoes open_sequence /65535 first)')
    argspar = parser.parse_args()
    argspar.cuda = not argspar.no_gpu and torch.cuda.is_available()

    print("> Parameters:")
    for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
        print(f"\t{p}: {v}")
    main(**vars(argspar))