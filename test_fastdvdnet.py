#!/bin/sh
"""
Denoise all the sequences existent in a given folder using FastDVDnet.
"""
import os
import argparse
import time
import cv2
import torch
import torch.nn as nn
import csv
import glob
import pandas as pd
from models import FastDVDnet
from fastdvdnet import denoise_seq_fastdvdnet
from utils import batch_psnr, init_logger_test, \
				variable_to_cv2_image, remove_dataparallel_wrapper, open_sequence, close_logger

NUM_IN_FR_EXT = 5 # temporal size of patch
MC_ALGO = 'DeepFlow' # motion estimation algorithm
OUTIMGEXT = '.tif' # output images format

def print_averages_from_csv(csv_path):
    """Read csv_path (psnr_results.csv) and print per-sigma and overall means for PSNR and denoise time."""
    if not os.path.exists(csv_path):
        print("CSV not found:", csv_path)
        return
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print("Failed to read CSV:", e)
        return

    # Ensure numeric columns
    df['sigma'] = pd.to_numeric(df['sigma'], errors='coerce')
    df['psnr_noisy'] = pd.to_numeric(df['psnr_noisy'], errors='coerce')
    df['psnr_denoised'] = pd.to_numeric(df['psnr_denoised'], errors='coerce')
    df['denoise_seconds'] = pd.to_numeric(df.get('denoise_seconds', 0.0), errors='coerce')

    df = df.dropna(subset=['sigma'])

    agg = df.groupby('sigma').agg(
        psnr_noisy_mean=('psnr_noisy', 'mean'),
        psnr_noisy_std =('psnr_noisy','std'),
        psnr_deno_mean=('psnr_denoised', 'mean'),
        psnr_deno_std =('psnr_denoised','std'),
        denoise_time_mean=('denoise_seconds', 'mean'),
        denoise_time_std =('denoise_seconds','std'),
        count=('sigma','count')
    ).reset_index().sort_values('sigma')

    pd.set_option('display.float_format', lambda x: f'{x:0.4f}')
    print("\nPer-sigma averages:")
    print(agg.to_string(index=False))

    print("\nOverall averages (all rows):")
    print(f"  PSNR noisy mean: {df['psnr_noisy'].mean():.4f}")
    print(f"  PSNR denoised mean: {df['psnr_denoised'].mean():.4f}")
    print(f"  Denoise time mean (s): {df['denoise_seconds'].mean():.4f}")
    print(f"  Rows counted: {int(df.shape[0])}")


def evaluate_testsets_root(args, model_temp, device):
	"""Minimal evaluator: runs sigmas [10,20,30,40,50] over
	   testsets under args['test_path'] and writes CSV to args['save_path']/psnr_results.csv.
	   Saves outputs under args['save_path']/FastDVDnet/<sigma>/<testset>/<sequence>/.
	   Times only the denoising call (compute time), printed and saved to CSV.
	"""
	sigmas = [10, 20, 30, 40, 50]
	out_csv = os.path.join(args['save_path'], 'psnr_results.csv')
	os.makedirs(args['save_path'], exist_ok=True)

	with open(out_csv, 'w', newline='') as csvf:
		writer = csv.writer(csvf)
		writer.writerow(['testset', 'sequence', 'sigma', 'psnr_noisy', 'psnr_denoised', 'denoise_seconds'])

		# iterate testset folders
		for tset in sorted(os.listdir(args['test_path'])):


			tset_path = os.path.join(args['test_path'], tset)
			if not os.path.isdir(tset_path):
				continue
			print(f"\nProcessing testset: {tset} ({tset_path})")
			for seq in sorted(os.listdir(tset_path)):
				seq_path = os.path.join(tset_path, seq)
				if not os.path.isdir(seq_path):
					continue
				print(" Loading sequence:", seq_path)
				seq_np, _, _ = open_sequence(seq_path, args['gray'], expand_if_needed=False, max_num_fr=args['max_num_fr_per_seq'])
				if seq_np is None:
					print("  WARNING: could not open sequence", seq_path)
					continue

				seq_t = torch.from_numpy(seq_np).to(device)  # shape (F,C,H,W)
				for sigma in sigmas:
					sigma_norm = float(sigma) / 255.0  # fixed assumption

					noise = torch.empty_like(seq_t).normal_(mean=0.0, std=sigma_norm).to(device)
					seqn = seq_t + noise
					noisestd = torch.FloatTensor([sigma_norm]).to(device)

					# time only the denoising call
					t0 = time.time()
					with torch.no_grad():
						denframes = denoise_seq_fastdvdnet(seq=seqn, noise_std=noisestd, temp_psz=NUM_IN_FR_EXT, model_temporal=model_temp)
					t1 = time.time()
					denoise_time = t1 - t0

					# compute PSNRs (on CPU)
					psnr_deno = batch_psnr(denframes.cpu(), seq_t.cpu(), 1.0)
					psnr_noisy = batch_psnr(seqn.cpu(), seq_t.cpu(), 1.0)

					print(f"  {tset}/{seq} sigma={sigma}: PSNR_noisy={psnr_noisy:.4f}, PSNR_denoised={psnr_deno:.4f}, denoise_time={denoise_time:.3f}s")
					writer.writerow([tset, seq, sigma, float(psnr_noisy), float(psnr_deno), denoise_time])
					csvf.flush()

					# save outputs in the same style as authors (under save_path)
					out_dir = os.path.join(args['save_path'], 'FastDVDnet', str(sigma), tset, seq)
					os.makedirs(out_dir, exist_ok=True)
					# reuse existing save_out_seq(seqnoisy, seqclean, save_dir, sigmaval, suffix, save_noisy)
					save_out_seq(seqn, denframes, out_dir, int(sigma), args['suffix'], args['save_noisy'])

	print("Evaluation finished. CSV at:", out_csv)

def save_out_seq(seqnoisy, seqclean, save_dir, sigmaval, suffix, save_noisy):
	"""Saves the denoised and noisy sequences under save_dir
	"""
	seq_len = seqnoisy.size()[0]
	for idx in range(seq_len):
		# Build Outname
		fext = OUTIMGEXT
		noisy_name = os.path.join(save_dir,\
						('n{}_{}').format(sigmaval, idx) + fext)
		if len(suffix) == 0:
			out_name = os.path.join(save_dir,\
					('n{}_FastDVDnet_{}').format(sigmaval, idx) + fext)
		else:
			out_name = os.path.join(save_dir,\
					('n{}_FastDVDnet_{}_{}').format(sigmaval, suffix, idx) + fext)

		# Save result
		if save_noisy:
			noisyimg = variable_to_cv2_image(seqnoisy[idx].clamp(0., 1.))
			cv2.imwrite(noisy_name, noisyimg)

		outimg = variable_to_cv2_image(seqclean[idx].unsqueeze(dim=0))
		cv2.imwrite(out_name, outimg)

def test_fastdvdnet(**args):
	"""Denoises all sequences present in a given folder. Sequences must be stored as numbered
	image sequences. The different sequences must be stored in subfolders under the "test_path" folder.

	Inputs:
		args (dict) fields:
			"model_file": path to model
			"test_path": path to sequence to denoise
			"suffix": suffix to add to output name
			"max_num_fr_per_seq": max number of frames to load per sequence
			"noise_sigma": noise level used on test set
			"dont_save_results: if True, don't save output images
			"no_gpu": if True, run model on CPU
			"save_path": where to save outputs as png
			"gray": if True, perform denoising of grayscale images instead of RGB
	"""
	# Start time
	start_time = time.time()

	# If save_path does not exist, create it
	if not os.path.exists(args['save_path']):
		os.makedirs(args['save_path'])
	logger = init_logger_test(args['save_path'])

	# Sets data type according to CPU or GPU modes
	if args['cuda']:
		device = torch.device('cuda')
	else:
		device = torch.device('cpu')

	# Create models
	print('Loading models ...')
	model_temp = FastDVDnet(num_input_frames=NUM_IN_FR_EXT)

	# Load saved weights
	state_temp_dict = torch.load(args['model_file'], map_location=device)
	if args['cuda']:
		device_ids = [0]
		model_temp = nn.DataParallel(model_temp, device_ids=device_ids).cuda()
	else:
		# CPU mode: remove the DataParallel wrapper
		state_temp_dict = remove_dataparallel_wrapper(state_temp_dict)
	model_temp.load_state_dict(state_temp_dict)

	# Sets the model in evaluation mode (e.g. it removes BN)
	model_temp.eval()

	    # If args['test_path'] is a directory containing subfolders (testsets), run multi-sigma evaluation:
    	# If args['test_path'] is a directory containing subfolders (testsets), run multi-sigma evaluation:
	if os.path.isdir(args['test_path']):
		# detect if the path contains subdirectories (assume root-of-testsets)
		has_subdirs = any(os.path.isdir(os.path.join(args['test_path'], d)) for d in os.listdir(args['test_path']))
		if has_subdirs:
			# run the evaluator (device already selected earlier)
			device = torch.device('cuda') if args['cuda'] else torch.device('cpu')
			evaluate_testsets_root(args, model_temp, device)
			# close logger and exit the function early (we already saved logs)

			#printe averages
			csv_path = os.path.join(args['save_path'], 'psnr_results.csv')
			print_averages_from_csv(csv_path)

			close_logger(logger)
			return

	

	with torch.no_grad():
		# process data
		seq, _, _ = open_sequence(args['test_path'],\
									args['gray'],\
									expand_if_needed=False,\
									max_num_fr=args['max_num_fr_per_seq'])
		seq = torch.from_numpy(seq).to(device)
		#print("seq shape =", seq.shape) #DEBUG LINE
		seq_time = time.time()

		# Add noise
		noise = torch.empty_like(seq).normal_(mean=0, std=args['noise_sigma']).to(device)
		seqn = seq + noise
		noisestd = torch.FloatTensor([args['noise_sigma']]).to(device)

		#print("channels =", seq.shape[1]) #DEBUG LINE

		denframes = denoise_seq_fastdvdnet(seq=seqn,\
										noise_std=noisestd,\
										temp_psz=NUM_IN_FR_EXT,\
										model_temporal=model_temp)

	# Compute PSNR and log it
	stop_time = time.time()
	psnr = batch_psnr(denframes, seq, 1.)
	psnr_noisy = batch_psnr(seqn.squeeze(), seq, 1.)
	loadtime = (seq_time - start_time)
	runtime = (stop_time - seq_time)
	seq_length = seq.size()[0]
	logger.info("Finished denoising {}".format(args['test_path']))
	logger.info("\tDenoised {} frames in {:.3f}s, loaded seq in {:.3f}s".\
				 format(seq_length, runtime, loadtime))
	logger.info("\tPSNR noisy {:.4f}dB, PSNR result {:.4f}dB".format(psnr_noisy, psnr))

	# Save outputs
	if not args['dont_save_results']:
		# Save sequence
		#save_out_seq(seqn, denframes, args['save_path'], \
					   #int(args['noise_sigma']*255), args['suffix'], args['save_noisy'])
		
		# For 14-bit images, save sequence
		save_out_seq(seqn, denframes, args['save_path'], \
					   int(args['noise_sigma']*255), args['suffix'], args['save_noisy'])

	# close logger
	close_logger(logger)

if __name__ == "__main__":
	# Parse arguments
	parser = argparse.ArgumentParser(description="Denoise a sequence with FastDVDnet")
	parser.add_argument("--model_file", type=str,\
						default="./model.pth", \
						help='path to model of the pretrained denoiser')
	parser.add_argument("--test_path", type=str, default="./data/rgb/Kodak24", \
						help='path to sequence to denoise')
	parser.add_argument("--suffix", type=str, default="", help='suffix to add to output name')
	parser.add_argument("--max_num_fr_per_seq", type=int, default=85, \
						help='max number of frames to load per sequence')
	parser.add_argument("--noise_sigma", type=float, default=25, help='noise level used on test set')
	parser.add_argument("--dont_save_results", action='store_true', help="don't save output images")
	parser.add_argument("--save_noisy", action='store_true', help="save noisy frames")
	parser.add_argument("--no_gpu", action='store_true', help="run model on CPU")
	parser.add_argument("--save_path", type=str, default='./results', \
						 help='where to save outputs as png')
	parser.add_argument("--gray", action='store_true',\
						help='perform denoising of grayscale images instead of RGB')

	argspar = parser.parse_args()
	# Normalize noises ot [0, 1]
	#argspar.noise_sigma /= 255.

	# For 14-bit, normalize noises ot [0, 1]
	argspar.noise_sigma /= 255.

	# use CUDA?
	argspar.cuda = not argspar.no_gpu and torch.cuda.is_available()

	print("\n### Testing FastDVDnet model ###")
	print("> Parameters:")
	for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
		print('\t{}: {}'.format(p, v))
	print('\n')

	test_fastdvdnet(**vars(argspar))
