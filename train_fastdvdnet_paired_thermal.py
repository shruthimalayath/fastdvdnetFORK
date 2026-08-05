"""
Trains a FastDVDnet model.
"""
import time
import os, sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from models import FastDVDnet
from dataset_val_paired_thermal import PairedValDataset
#from dataloaders_paired import train_dali_loader
from dataset_paired_thermal import PairedThermalDataset
from torch.utils.data import DataLoader
from utils_thermal import svd_orthogonalization, close_logger, init_logging, normalize_augment_pair
from train_common_paired_thermal import resume_training, lr_scheduler, log_train_psnr, \
					validate_and_log, save_model_checkpoint

#logging imports
import time
import atexit



def main(**args):

	#Load dataset NOT with dali, but with custom dataset class for paired thermal images
	r"""Performs the main training loop
	"""
	print('> Loading datasets ...')

	dataset_val = PairedValDataset(
		 noisy_root=args['val_noisy_dir'],
		 clean_root=args['val_clean_dir']
	) 

	dataset_train = PairedThermalDataset(
		noisy_root=args['train_noisy_dir'],
		clean_root=args['train_clean_dir'],
		patch_size=args['patch_size'],
		temp_patch_size=args['temp_patch_size'],
		epoch_size=args['max_number_patches'],
		preload = True,
		progress = True
	)

	loader_train = DataLoader(
		dataset_train,
		batch_size=args['batch_size'],
		shuffle=True,
		num_workers=8,
		pin_memory=True
	)


	num_minibatches = int(args['max_number_patches']//args['batch_size'])
	ctrl_fr_idx = (args['temp_patch_size'] - 1) // 2
	print("\t# of training samples: %d\n" % int(args['max_number_patches']))

	# Init loggers
	writer, logger = init_logging(args)

	# Define GPU devices #
	device_ids = [1]
	torch.backends.cudnn.benchmark = True # CUDNN optimization

	# Create model
	#model = FastDVDnet()
	#model = nn.DataParallel(model, device_ids=device_ids).cuda() 

	device_ids = list(range(torch.cuda.device_count()))
	device0 = f'cuda:{device_ids[0]}' if device_ids else 'cuda:0'

	model = FastDVDnet()
	if len(device_ids) > 1:
		# wrap first, then move to output device (recommended)
		model = nn.DataParallel(model, device_ids=device_ids).to(device0)
	else:
		model = model.to(device0)
	

	# Define loss
	criterion = nn.MSELoss(reduction='sum')
	criterion.cuda()

	# Optimizer
	optimizer = optim.Adam(model.parameters(), lr=args['lr'])

	# Resume training or start anew
	start_epoch, training_params = resume_training(args, model, optimizer)

	# Training
	start_time = time.time()
	#print("DEBUG: torch.cuda.device_count()", torch.cuda.device_count())  
	#print("DEBUG: device_ids variable:", device_ids) 
	#print("DEBUG: model.device_ids:", getattr(model, "device_ids", None))
	for epoch in range(start_epoch, args['epochs']):
		#print(f"DEBUG: start epoch {epoch}")
		# Set learning rate
		current_lr, reset_orthog = lr_scheduler(epoch, args)
		if reset_orthog:
			training_params['no_orthog'] = True

		# set learning rate in optimizer
		for param_group in optimizer.param_groups:
			param_group["lr"] = current_lr
		#print('\nlearning rate %f' % current_lr)

		# train

		for i, data in enumerate(loader_train, 0):
			#print(f"DEBUG: start batch {i}") 
			# Pre-training step
			model.train()
			# When optimizer = optim.Optimizer(net.parameters()) we only zero the optim's grads
			optimizer.zero_grad()

			#NEW: 2 frames: noisy and clean, from the paired dataset, per sample noise STD computed from the residual, dual aug&norm
			# get paired noisy/clean sequences
			noisy_seq, clean_seq = data

			# normalize + apply SAME augmentation to noisy and clean
			imgn_train, gt_train = normalize_augment_pair(noisy_seq, clean_seq, ctrl_fr_idx)
			N, _, H, W = imgn_train.size()
			#print("DEBUG: imgn_train.shape", getattr(imgn_train, "shape", None))
			#print("DEBUG: gt_train.shape", getattr(gt_train, "shape", None))

			# move to GPU
			imgn_train = imgn_train.cuda(non_blocking=True)
			gt_train = gt_train.cuda(non_blocking=True)

			# center noisy frame
			noisy_central = imgn_train[:,3*ctrl_fr_idx:3*ctrl_fr_idx+3,:,:]

			# noise_map = residual
			#start_ch = ctrl_fr_idx * 3
			#noisy_center = imgn_train[:, start_ch:start_ch+3, :, :]   # [B,3,H,W]
			# If original data was grayscale repeated to 3 channels, extract single channel:
			#noisy_center_single = noisy_center[:, :1, :, :]          # [B,1,H,W]
			#gt_single = gt_train[:, :1, :, :]
			#residual = noisy_center_single - gt_single
			#noise_map = residual.abs()    # shape [B,1,H,W]
			#noise_map = noise_map.clamp(min=1e-6).to(imgn_train.dtype).to(imgn_train.device)


			#hard coded noise map
			CONST_SIGMA = 25.0
			# normalize to [0,1] (training data is normalized to [0,1])
			sigma_norm = CONST_SIGMA / 255.0
			# N, _, H, W are already available above
			# create per-sample, per-pixel constant noise map: shape [B,1,H,W]
			noise_map = torch.full((N, 1, H, W), float(sigma_norm),
								   dtype=imgn_train.dtype, device=imgn_train.device)
			# clamp tiny values for safety
			noise_map = noise_map.clamp(min=1e-6)
			# --- end hard-coded block ---

			# Evaluate model and optimize it
			out_train = model(imgn_train, noise_map)

			# Compute loss
			loss = criterion(gt_train, out_train) / (N*2)
			loss.backward()
			optimizer.step()

			# Results
			if training_params['step'] % args['save_every'] == 0:
				# Apply regularization by orthogonalizing filters
				if not training_params['no_orthog']:
					model.apply(svd_orthogonalization)

				# Compute training PSNR
				log_train_psnr(out_train, \
								gt_train, \
								loss, \
								writer, \
								epoch, \
								i, \
								num_minibatches, \
								training_params)
			# update step counter
			training_params['step'] += 1

		# Call to model.eval() to correctly set the BN layers before inference
		model.eval()

		# Validation and log images
		validate_and_log(
			model_temp=model,
			dataset_val=dataset_val,
			temp_psz=args['temp_patch_size'],
			writer=writer,
			epoch=epoch,
			lr=current_lr,
			logger=logger,
			trainimg=imgn_train
		)

		# save model and checkpoint
		training_params['start_epoch'] = epoch + 1
		save_model_checkpoint(model, args, optimizer, training_params, epoch)

	# Print elapsed time
	elapsed_time = time.time() - start_time
	print('Elapsed time {}'.format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))

	# Close logger file
	close_logger(logger)

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description="Train the denoiser")

	#Training parameters
	parser.add_argument("--batch_size", type=int, default=64, 	\
					 help="Training batch size")
	parser.add_argument("--epochs", "--e", type=int, default=80, \
					 help="Number of total training epochs")
	parser.add_argument("--resume_training", "--r", action='store_true',\
						help="resume training from a previous checkpoint")
	parser.add_argument("--milestone", nargs=2, type=int, default=[50, 60], \
						help="When to decay learning rate; should be lower than 'epochs'")
	parser.add_argument("--lr", type=float, default=1e-3, \
					 help="Initial learning rate")
	parser.add_argument("--no_orthog", action='store_true',\
						help="Don't perform orthogonalization as regularization")
	parser.add_argument("--save_every", type=int, default=10,\
						help="Number of training steps to log psnr and perform \
						orthogonalization")
	parser.add_argument("--save_every_epochs", type=int, default=5,\
						help="Number of training epochs to save state")

	#No longer needed because of dynamic noise maps
	#parser.add_argument("--noise_ival", nargs=2, type=int, default=[5, 55], help="Noise training interval")
	#parser.add_argument("--val_noiseL", type=float, default=25, help='noise level used on validation set')  

	# Preprocessing parameters
	parser.add_argument("--patch_size", "--p", type=int, default=96, help="Patch size")
	parser.add_argument("--temp_patch_size", "--tp", type=int, default=5, help="Temporal patch size")
	parser.add_argument("--max_number_patches", "--m", type=int, default=384000, help="Maximum number of patches")
	
	# Dirs
	parser.add_argument("--log_dir", type=str, default="logs", help='path of log files')
	#parser.add_argument("--trainset_dir", type=str, default=None, help='path of trainset')

	#Paths to paired dataset
	parser.add_argument("--train_noisy_dir", type=str, required=True, help = "path to the directory containing noisy training images")
	parser.add_argument("--train_clean_dir", type=str, required=True, help = "path to the directory containing clean training images")

	parser.add_argument("--val_noisy_dir", type=str, default=None, help='path to the directory containing noisy validation images')
	parser.add_argument("--val_clean_dir", type=str, default=None, help='path to the directory containing clean validation images')
	argspar = parser.parse_args()

	#for log logging
	log_dir = argspar.log_dir
	os.makedirs(log_dir, exist_ok=True)
	logfile = os.path.join(log_dir, f"train_{time.strftime('%Y%m%d-%H%M%S')}.log")

	# open file line-buffered
	f = open(logfile, 'a', buffering=1, encoding='utf-8')

	class Tee:
		def __init__(self, *streams):
			self.streams = streams

		def write(self, s):
			for st in self.streams:
				try:
					st.write(s)
				except Exception:
					pass

		def flush(self):
			for st in self.streams:
				try:
					st.flush()
				except Exception:
					pass

		def fileno(self):
			# Return a fileno from one of the wrapped streams, prefer the file (last stream)
			for st in reversed(self.streams):
				if hasattr(st, "fileno"):
					try:
						return st.fileno()
					except Exception:
						continue
			# If no underlying stream exposes fileno, raise to match file-like behavior
			raise OSError("Tee: no fileno available")

	def _close_log():
		try:
			f.write(f"\n=== Training log ended: {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
			f.flush()
		except Exception:
			pass
		try:
			f.close()
		except Exception:
			pass

	# replace stdout/stderr with a tee to terminal + file
	sys.stdout = Tee(sys.__stdout__, f)
	sys.stderr = sys.stdout

	# write a header and ensure file is closed on exit
	f.write(f"=== Training log started: {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
	f.flush()
	#atexit.register(lambda: (f.write(f"\n=== Training log ended: {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n"), f.close()))
	atexit.register(_close_log)

	#for terminal logging
	print("\n### Training FastDVDnet denoiser model ###")
	print("> Parameters:")
	for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
		print('\t{}: {}'.format(p, v))
	print('\n')

	main(**vars(argspar))