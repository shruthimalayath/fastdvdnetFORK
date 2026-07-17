#!/bin/sh
"""
Denoise paired noisy/clean sequences using FastDVDnet.
"""
import os
import argparse
import time

import cv2
import torch
import torch.nn as nn

from models import FastDVDnet
from fastdvdnet import denoise_seq_fastdvdnet
from utils import (batch_psnr,init_logger_test,variable_to_cv2_image,remove_dataparallel_wrapper,open_sequence, close_logger)


#change: test is done only on noisy images and no noise sigma value is taken as input??
NUM_IN_FR_EXT = 5
OUTIMGEXT = ".tif"


def save_out_seq(seqnoisy, seqclean, save_dir, sigmaval, suffix, save_noisy):
    """Saves the denoised and noisy sequences under save_dir."""
    seq_len = seqnoisy.size()[0]
    for idx in range(seq_len):
        fext = OUTIMGEXT

        noisy_name = os.path.join(save_dir, ("n{}_{}").format(sigmaval, idx) + fext)

        if len(suffix) == 0:
            out_name = os.path.join(save_dir, ("n{}_FastDVDnet_{}").format(sigmaval, idx) + fext)
        else:
            out_name = os.path.join(save_dir, ("n{}_FastDVDnet_{}_{}").format(sigmaval, suffix, idx) + fext)

        if save_noisy:
            noisyimg = variable_to_cv2_image(seqnoisy[idx].clamp(0.0, 1.0))
            cv2.imwrite(noisy_name, noisyimg)

        outimg = variable_to_cv2_image(seqclean[idx].unsqueeze(dim=0))
        cv2.imwrite(out_name, outimg)


def test_fastdvdnet_paired(**args):
    """Denoise a paired noisy/clean sequence with FastDVDnet and evaluate PSNR."""
    start_time = time.time()

    if not os.path.exists(args["save_path"]):
        os.makedirs(args["save_path"])
    logger = init_logger_test(args["save_path"])

    if args["cuda"]:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("Loading model ...")
    model_temp = FastDVDnet(num_input_frames=NUM_IN_FR_EXT)

    state_temp_dict = torch.load(args["model_file"], map_location=device)

    if args["cuda"]:
        model_temp = nn.DataParallel(model_temp, device_ids=[0]).cuda()
    else:
        state_temp_dict = remove_dataparallel_wrapper(state_temp_dict)

    model_temp.load_state_dict(state_temp_dict)
    model_temp.eval()

    with torch.no_grad():
        print("Loading noisy and clean sequences ...")
        noisy_seq, _, _ = open_sequence(
            args["noisy_dir"],
            args["gray"],
            expand_if_needed=False,
            max_num_fr=args["max_num_fr_per_seq"],
        )
        clean_seq, _, _ = open_sequence(
            args["clean_dir"],
            args["gray"],
            expand_if_needed=False,
            max_num_fr=args["max_num_fr_per_seq"],
        )

        noisy_seq = torch.from_numpy(noisy_seq).float().to(device)
        clean_seq = torch.from_numpy(clean_seq).float().to(device)

        if noisy_seq.max() > 1.0:
            noisy_seq = noisy_seq / 16383.0
            clean_seq = clean_seq / 16383.0

        seq_time = time.time()

        # Compute noise level from residual between noisy and clean sequences
        residual = noisy_seq - clean_seq
        noisestd = residual.view(-1).std(unbiased=False).clamp(min=1e-6)
        noisestd = noisestd.view(1, 1, 1, 1)
        denframes = denoise_seq_fastdvdnet(
            seq=noisy_seq,
            noise_std=noisestd,
            temp_psz=NUM_IN_FR_EXT,
            model_temporal=model_temp,
        )

    stop_time = time.time()
    psnr = batch_psnr(denframes, clean_seq, 1.0)
    psnr_noisy = batch_psnr(noisy_seq, clean_seq, 1.0)
    loadtime = seq_time - start_time
    runtime = stop_time - seq_time
    seq_length = noisy_seq.size()[0]
    logger.info("Finished denoising {}".format(args["noisy_dir"]))
    logger.info(
        "\tDenoised {} frames in {:.3f}s, loaded seq in {:.3f}s".format(seq_length, runtime, loadtime)
    )
    logger.info("\tPSNR noisy {:.4f}dB, PSNR result {:.4f}dB".format(psnr_noisy, psnr))

    if not args["dont_save_results"]:
        save_out_seq(
            noisy_seq,
            denframes,
            args["save_path"],
            0,
            args["suffix"],
            args["save_noisy"],
        )

    close_logger(logger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Denoise paired noisy/clean sequences with FastDVDnet")
    parser.add_argument("--model_file", type=str, default="./model.pth", help="path to trained FastDVDnet model")
    parser.add_argument("--noisy_dir", type=str, default="./data/val/noisy", help="path to noisy sequence folder")
    parser.add_argument("--clean_dir", type=str, default="./data/val/clean", help="path to clean sequence folder")
    parser.add_argument("--suffix", type=str, default="", help="suffix to add to output name")
    parser.add_argument("--max_num_fr_per_seq", type=int, default=25, help="max number of frames to load per sequence")
    parser.add_argument("--dont_save_results", action="store_true", help="don't save output images")
    parser.add_argument("--save_noisy", action="store_true", help="save noisy frames")
    parser.add_argument("--no_gpu", action="store_true", help="run model on CPU")
    parser.add_argument("--save_path", type=str, default="./results", help="where to save outputs")
    parser.add_argument("--gray", action="store_true", help="perform denoising of grayscale images instead of RGB")

    argspar = parser.parse_args()

    # use CUDA?
    argspar.cuda = not argspar.no_gpu and torch.cuda.is_available()

    print("\n### Testing FastDVDnet model on paired data ###")
    print("> Parameters:")
    for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
        print("\t{}: {}".format(p, v))
    print("\n")

    test_fastdvdnet_paired(**vars(argspar))