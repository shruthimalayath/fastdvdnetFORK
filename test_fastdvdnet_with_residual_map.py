#!/usr/bin/env python3
"""
Denoise a TIFF sequence using FastDVDnet, using the residual (noisy - clean) as the per-pixel noise map.

Usage example:
CUDA_VISIBLE_DEVICES=0 python test_fastdvdnet_with_residual_map_fixed.py \
    --model_file ./model.pth \
    --test_noisy_path /path/to/noisy_seq \
    --test_clean_path /path/to/clean_seq \
    --save_path ./out --gray --max_val 65535
"""
import os
import argparse
import time
import numpy as np
import cv2
import torch
import torch.nn as nn

from models import FastDVDnet
from utils_thermal import variable_to_cv2_image, init_logger_test, open_sequence, close_logger

NUM_IN_FR_EXT = 5  # temporal size of patch (must match model)
OUTIMGEXT = '.tif'  # save as 16-bit tiff


def pad_indices(idx, F, radius):
    """Return indices for temporal window centered at idx with radius, clamped at ends (replicate)."""
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

    # Wrap in DataParallel if using CUDA (keeps same behaviour as earlier scripts)
    if use_cuda and torch.cuda.is_available():
        model = nn.DataParallel(model, device_ids=[0]).cuda()

    # Align module prefix
    is_dp = isinstance(model, nn.DataParallel)
    has_module = any(k.startswith('module.') for k in state_dict.keys())

    if has_module and not is_dp:
        # strip module.
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


def save_frame_tensor_to_tiff(tensor_frame, out_path):
    """tensor_frame: torch tensor in [0,1], shape (C,H,W) or (H,W). Writes 16-bit TIFF."""
    np_img = variable_to_cv2_image(tensor_frame.clamp(0., 1.))
    cv2.imwrite(out_path, np_img)


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

    # Load noisy & clean sequences (they should have same number of frames & same resolution)
    seqn_np, _, _ = open_sequence(args['test_noisy_path'], args['gray'], expand_if_needed=False,
                                  max_num_fr=args['max_num_fr_per_seq'])
    seqc_np, _, _ = open_sequence(args['test_clean_path'], args['gray'], expand_if_needed=False,
                                  max_num_fr=args['max_num_fr_per_seq'])
    if seqn_np is None or seqc_np is None:
        raise RuntimeError("Could not open noisy or clean sequence paths")
    if seqn_np.shape != seqc_np.shape:
        raise RuntimeError(f"Noisy and clean sequences must have same shape. noisy:{seqn_np.shape} clean:{seqc_np.shape}")

    F, C, H, W = seqn_np.shape
    print(f"Loaded sequences: frames={F}, channels={C}, H={H}, W={W}")

    # Normalize to [0,1] only if not already normalized by open_sequence/open_image.
    max_val = float(args['max_val'])
    seqn_np = seqn_np.astype(np.float32)
    seqc_np = seqc_np.astype(np.float32)

    # Debug prints to ensure ranges
    print(f"DEBUG before scaling: noisy min={float(seqn_np.min()):.6e} max={float(seqn_np.max()):.6e}, "
          f"clean min={float(seqc_np.min()):.6e} max={float(seqc_np.max()):.6e}")

    if seqn_np.max() > 1.0 + 1e-6:
        seqn_np = seqn_np / max_val
    if seqc_np.max() > 1.0 + 1e-6:
        seqc_np = seqc_np / max_val

    # Move to tensors on device
    seqn = torch.from_numpy(seqn_np).to(device)  # [F,C,H,W]
    seqc = torch.from_numpy(seqc_np).to(device)

    # post-normalization debug
    print(f"DEBUG after scaling: noisy min={float(seqn.min()):.6e} max={float(seqn.max()):.6e}, "
          f"clean min={float(seqc.min()):.6e} max={float(seqc.max()):.6e}")

    radius = (NUM_IN_FR_EXT - 1) // 2

    # Output container
    denoised_frames = []

    with torch.no_grad():
        for t in range(F):
            # build temporal patch indices (replicate ends)
            inds = pad_indices(t, F, radius)
            # collect frames -> list of tensors shape [C,H,W], stack to [num_frames, C, H, W]
            patch_frames = torch.stack([seqn[i] for i in inds], dim=0)  # [T, C, H, W]
            # flatten frames to shape [1, T*C, H, W] matching training input
            T = patch_frames.shape[0]
            patch_flat = patch_frames.permute(1, 0, 2, 3).contiguous().view(1, T * C, H, W)  # [1, T*C, H, W]

            # compute noise_map from center frame residual abs(noisy - clean) on center index t
            noisy_center = seqn[t]   # [C,H,W]
            clean_center = seqc[t]   # [C,H,W]
            # If data was stacked RGB for grayscale sensors (3 identical channels), select first channel for residual
            noisy_center_single = noisy_center[:1, :, :].unsqueeze(0)  # [1,1,H,W]
            clean_center_single = clean_center[:1, :, :].unsqueeze(0)  # [1,1,H,W]
            residual = (noisy_center_single - clean_center_single).abs()  # [1,1,H,W]
            # ensure non-zero minimum to avoid numerical issues
            noise_map = residual.clamp(min=1e-6).to(dtype=patch_flat.dtype, device=patch_flat.device)

            # run model: model may return either:
            # - [N, T*C, H, W] (one output per input frame) OR
            # - [N, C, H, W]     (only center frame output)
            out_patch = model(patch_flat, noise_map)  # expected shape [1, num_out_ch, H, W]

            # DEBUG: optionally inspect model output shape
            # print(f"DEBUG: out_patch.shape={tuple(out_patch.shape)}, T={T}, C={C}, radius={radius}")

            num_out_ch = out_patch.shape[1]
            # Case 1: model gives outputs per-frame (T*C channels)
            if num_out_ch == T * C:
                start_ch = radius * C
                out_center = out_patch[0, start_ch:start_ch + C, :, :]  # [C,H,W]
            # Case 2: model directly returns center frame channels (C channels)
            elif num_out_ch == C:
                out_center = out_patch[0]  # [C,H,W]
            # Case 3: some other layout — pick the central block of channels
            elif num_out_ch >= C:
                start_ch = max(0, (num_out_ch - C) // 2)
                out_center = out_patch[0, start_ch:start_ch + C, :, :]
            # Case 4: too few channels (e.g., model returns single channel but we need 3)
            else:
                if num_out_ch == 1 and C == 3:
                    single = out_patch[0, 0:1, :, :]  # [1,H,W]
                    out_center = single.repeat(3, 1, 1)  # [3,H,W]
                else:
                    raise RuntimeError(f"Unexpected number of output channels from model: {num_out_ch} (expected {C} or {T*C})")

            denoised_frames.append(out_center.cpu())

    # Stack outputs produced by the model
    denoised = torch.stack(denoised_frames, dim=0)  # [F, C, H, W]

    # Bring noisy & clean sequences to CPU for comparison / saving
    seqn_cpu = seqn.cpu()   # noisy inputs normalized [0,1]
    seqc_cpu = seqc.cpu()   # clean references normalized [0,1]

    # Decide whether model outputs are residuals or absolute images.
    den_min = float(denoised.min().item())
    den_max = float(denoised.max().item())
    # Heuristic: if denoised contains negatives or > 1, treat as residual
    if den_min < -1e-3 or den_max > 1.0 + 1e-6:
        print(f"Model outputs look like residuals (min={den_min:.6e}, max={den_max:.6e}), reconstructing by adding to noisy input.")
        recon = (seqn_cpu + denoised).clamp(0., 1.)
    else:
        print(f"Model outputs look like absolute images (min={den_min:.6e}, max={den_max:.6e}), using model outputs directly.")
        recon = denoised.clamp(0., 1.)


        # ===== diagnostics =====
    # How different is recon from the noisy input?
    diff = (recon - seqn_cpu).abs()
    print("DIAG: recon vs noisy: max_diff={:.6e}, mean_diff={:.6e}".format(float(diff.max().item()), float(diff.mean().item())))
    print("DIAG: recon range min/max = {:.6e} / {:.6e}".format(float(recon.min().item()), float(recon.max().item())))
    print("DIAG: noisy range min/max = {:.6e} / {:.6e}".format(float(seqn_cpu.min().item()), float(seqn_cpu.max().item())))

    # Are recon and noisy numerically almost equal?
    print("DIAG: allclose(recon,noisy)?", torch.allclose(recon, seqn_cpu, atol=1e-6))

    # Inspect a small patch numeric sample (center 5x5) for frame 0
    r0 = recon[0].numpy()  # CHW
    n0 = seqn_cpu[0].numpy()
    print("DIAG sample recon[0,:,256:261,320:325]:\n", r0[:,256:261,320:325])
    print("DIAG sample noisy[0,:,256:261,320:325]:\n", n0[:,256:261,320:325])

    # Print first conv layer stats to ensure weights were loaded (model must be in scope)
    first_conv = None
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            first_conv = m
            break
    if first_conv is not None:
        w = first_conv.weight.detach().cpu()
        print("DIAG first conv weight stats: mean={:.6e} std={:.6e} min={:.6e} max={:.6e}".format(
            float(w.mean()), float(w.std()), float(w.min()), float(w.max())))
    else:
        print("DIAG: no Conv2d found in model (unexpected).")
    # =======================

    # Compute PSNR over reconstructed frames (robust cropping/channel handling)
    from skimage.metrics import peak_signal_noise_ratio as compare_psnr
    psnr_sum = 0.0
    first_report = True
    for i in range(F):
        den_arr = recon[i].numpy().astype(np.float32)
        den_hwc = np.transpose(den_arr, (1, 2, 0))
        clean_arr = seqc_cpu[i].numpy().astype(np.float32)
        clean_hwc = np.transpose(clean_arr, (1, 2, 0))

        # reconcile channels
        if den_hwc.shape[2] != clean_hwc.shape[2]:
            if den_hwc.shape[2] == 1 and clean_hwc.shape[2] == 3:
                den_hwc = np.repeat(den_hwc, 3, axis=2)
            elif den_hwc.shape[2] == 3 and clean_hwc.shape[2] == 1:
                clean_hwc = np.repeat(clean_hwc, 3, axis=2)
            else:
                min_c = min(den_hwc.shape[2], clean_hwc.shape[2])
                den_hwc = den_hwc[:, :, :min_c]
                clean_hwc = clean_hwc[:, :, :min_c]

        # crop to common spatial size if needed
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

    psnr_avg = psnr_sum / F
    print(f"Average PSNR over sequence (reconstructed/used): {psnr_avg:.4f} dB")

    # Save reconstructed outputs as 16-bit TIFFs
    for i in range(F):
        out_name = os.path.join(args['save_path'], f"frame_{i:04d}{OUTIMGEXT}")
        np_img = variable_to_cv2_image(recon[i].clamp(0., 1.))
        cv2.imwrite(out_name, np_img)

    close_logger(logger)
    print("Done. Outputs saved to", args['save_path'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Denoise a sequence with FastDVDnet using residual noise map from clean/noisy pairs")
    parser.add_argument("--model_file", type=str, required=True, help='path to pretrained denoiser')
    parser.add_argument("--test_noisy_path", type=str, required=True, help='path to noisy sequence (folder)')
    parser.add_argument("--test_clean_path", type=str, required=True, help='path to clean sequence (folder) - used to compute residual noise map')
    parser.add_argument("--save_path", type=str, default='./results', help='where to save outputs')
    parser.add_argument("--max_num_fr_per_seq", type=int, default=200, help='max number of frames to load per sequence')
    parser.add_argument("--no_gpu", action='store_true', help="run on CPU")
    parser.add_argument("--gray", action='store_true', help='treat input images as grayscale (open_image will stack into 3 channels)')
    parser.add_argument("--max_val", type=float, default=65535.0, help='max pixel value used to normalize input (e.g. 65535 for 16-bit)')
    argspar = parser.parse_args()
    argspar.cuda = not argspar.no_gpu and torch.cuda.is_available()

    print("> Parameters:")
    for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
        print(f"\t{p}: {v}")
    main(**vars(argspar))