"""
Trains a FastDVDnet model.
"""
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from models import FastDVDnet
from dataset_val_paired import PairedValDataset
#from dataloaders_paired import train_dali_loader
from dataset_paired import PairedThermalDataset
from torch.utils.data import DataLoader
from utils import svd_orthogonalization, close_logger, init_logging, normalize_augment
from train_common_paired import resume_training, lr_scheduler, log_train_psnr, \
					validate_and_log, save_model_checkpoint



# Differences from original train_fastdvdnet.py: 
# 1. The dataset is loaded from a paired dataset of noisy and clean thermal images, instead of using a DALI loader.
# 2. The noise map is generated dynamically based on the validation noise level, instead of using a fixed value.
# 3. The training loop is modified to handle the paired dataset, with the noisy and clean sequences being loaded separately and processed accordingly.
# 4. The validation and logging functions are modified to handle the paired dataset, with the noisy and clean sequences being passed separately to the model for evaluation.


def main(**args):

	#previous change: Load dataset NOT with dali, with custom dataset class for paired thermal images
	#--------------------------------------------------------------------------------------------------
	#dataset_val = PairedValDataset( noisy_root=args['val_noisy_dir'], clean_root=args['val_clean_dir'])
	#loader_train = train_dali_loader(batch_size=args['batch_size'],\#file_root=args['trainset_dir'],\#sequence_length=args['temp_patch_size'],\#crop_size=args['patch_size'],\#epoch_size=args['max_number_patches'],\#random_shuffle=True,\#temp_stride=3)
	#dataset_train = PairedThermalDataset(#noisy_root=args['train_noisy_dir'],#clean_root=args['train_clean_dir'],patch_size=args['patch_size'],#temp_patch_size=args['temp_patch_size'],#epoch_size=args['max_number_patches'])
	#loader_train = DataLoader(#dataset_train,#batch_size=args['batch_size'],#shuffle=True,#num_workers=4,#pin_memory=True#)
	#loader_train = DataLoader(#dataset_train,#batch_size=args['batch_size'],#shuffle=True,#num_workers=4,#pin_memory=True#)
	#loader_train = train_dali_loader(
		#batch_size=args['batch_size'],
		#noisy_root=args['train_noisy_dir'],
		#clean_root=args['train_clean_dir'],
		#sequence_length=args['temp_patch_size'],
		#crop_size=args['patch_size'],
		#epoch_size=args['max_number_patches'],
		#random_shuffle=False,
		#temp_stride=3
	#)
	#-------------------------------------------------------------------------------------------------

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
		epoch_size=args['max_number_patches']
	)


	num_minibatches = int(args['max_number_patches']//args['batch_size'])
	ctrl_fr_idx = (args['temp_patch_size'] - 1) // 2
	print("\t# of training samples: %d\n" % int(args['max_number_patches']))

	# Init loggers
	writer, logger = init_logging(args)

	# Define GPU devices
	device_ids = [0]
	torch.backends.cudnn.benchmark = True # CUDNN optimization

	# Create model
	model = FastDVDnet()
	model = nn.DataParallel(model, device_ids=device_ids).cuda()

	# Define loss
	criterion = nn.MSELoss(reduction='sum')
	criterion.cuda()

	# Optimizer
	optimizer = optim.Adam(model.parameters(), lr=args['lr'])

	# Resume training or start anew
	start_epoch, training_params = resume_training(args, model, optimizer)

	# Training
	start_time = time.time()
	for epoch in range(start_epoch, args['epochs']):
		# Set learning rate
		current_lr, reset_orthog = lr_scheduler(epoch, args)
		if reset_orthog:
			training_params['no_orthog'] = True

		# set learning rate in optimizer
		for param_group in optimizer.param_groups:
			param_group["lr"] = current_lr
		print('\nlearning rate %f' % current_lr)

		# train

		for i, data in enumerate(loader_train, 0):

			# Pre-training step
			model.train()

			# When optimizer = optim.Optimizer(net.parameters()) we only zero the optim's grads
			optimizer.zero_grad()

			#OLD
			# convert inp to [N, num_frames*C. H, W] in  [0., 1.] from [N, num_frames, C. H, W] in [0., 255.]
			# extract ground truth (central frame)
			#img_train, gt_train = normalize_augment(data[0]['data'], ctrl_fr_idx)
			#N, _, H, W = img_train.size()
			# std dev of each sequence
			#stdn = torch.empty((N, 1, 1, 1)).cuda().uniform_(args['noise_ival'][0], to=args['noise_ival'][1])
			# draw noise samples from std dev tensor
			#noise = torch.zeros_like(img_train)
			#noise = torch.normal(mean=noise, std=stdn.expand_as(noise))
			#define noisy input
			#imgn_train = img_train + noise

			#NEW: 2 frames: noisy and clean, from the paired dataset; 
			#per sample noise standard deviation computed from the residual

			noisy_seq= data[0]["noisy"]
			clean_seq= data[1]["clean"]

			N = noisy_seq.shape[0]

			imgn_train = noisy_seq.view(N, -1, noisy_seq.shape[-2],noisy_seq.shape[-1])
			clean_seq = clean_seq.view(N, -1, clean_seq.shape[-2], clean_seq.shape[-1])

			H = imgn_train.shape[-2]
			W = imgn_train.shape[-1]

			gt_train = clean_seq[:, 6:9, :, :]
			# Send tensors to GPU
			gt_train = gt_train.cuda(non_blocking=True)
			imgn_train = imgn_train.cuda(non_blocking=True)

			noisy_central = imgn_train[:, 6:9, :, :]
			residual = noisy_central - gt_train					
			sigma = residual.view(N, -1).std(dim=1, unbiased=False)
			sigma = sigma.clamp(min=1e-6).view(N, 1, 1, 1)

			noise_map = sigma.expand(N, 1, H, W)

			#old noise maps
			#noise = noise.cuda(non_blocking=True)
			#noise_map = stdn.expand((N, 1, H, W)).cuda(non_blocking=True) # one channel per image



			#temporary fix: FastDVDnet expects a noise map, but we don't have one for the paired dataset. 
			#So we will create a dummy noise map with a fixed value of 30/16383.0 
			#noise_map = torch.full(
				#(N, 1, H, W),
				#30.0 / 16383.0,
				#device=imgn_train.device
			#)

			#use val noise (old)
			#noise_map = torch.full((N, 1, H, W), args['val_noiseL'],   device=imgn_train.device)

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
						model_temp=model, \
						dataset_val=dataset_val, \
						valnoisestd=args['val_noiseL'], \
						temp_psz=args['temp_patch_size'], \
						writer=writer, \
						epoch=epoch, \
						lr=current_lr, \
						logger=logger, \
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
	parser.add_argument("--max_number_patches", "--m", type=int, default=256000, help="Maximum number of patches")
	
	
	# Dirs
	parser.add_argument("--log_dir", type=str, default="logs", help='path of log files')
	#parser.add_argument("--trainset_dir", type=str, default=None, help='path of trainset')

	#Paths to paired dataset
	parser.add_argument("--train_noisy_dir", type=str, required=True, help = "path to the directory containing noisy training images")
	parser.add_argument("--train_clean_dir", type=str, required=True, help = "path to the directory containing clean training images")

	parser.add_argument("--val_noisy_dir", type=str, default=None, help='path to the directory containing noisy validation images')
	parser.add_argument("--val_clean_dir", type=str, default=None, help='path to the directory containing clean validation images')
	argspar = parser.parse_args()

	# For 8-bit images, normalize noise between [0, 1]
	#argspar.val_noiseL /= 255.                     
	#argspar.noise_ival[0] /= 255.
	#argspar.noise_ival[1] /= 255.

	#For 16-bit images, normalize noise between [0, 1] -- not needed anymore, because we will compute the noise level dynamically based on the validation set
	#argspar.val_noiseL /= 16383.                     
	#argspar.noise_ival[0] /= 16383.
	#argspar.noise_ival[1] /= 16383.

	print("\n### Training FastDVDnet denoiser model ###")
	print("> Parameters:")
	for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
		print('\t{}: {}'.format(p, v))
	print('\n')

	main(**vars(argspar))