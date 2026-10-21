"""
Trains a FastDVDnet model with MSE fidelity plus explicit row/column
residual-bias penalties (targets temporally stable stripe / fixed-pattern noise).

Loss: MSE + row_loss_weight * Lrow + col_loss_weight * Lcol
    Lrow = mean(| mean_over_columns(pred - target) |)
    Lcol = mean(| mean_over_rows(pred - target) |)
"""
import time
import os, sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from models import FastDVDnet
from dataset_val_paired_thermal import PairedValDataset
from dataset_paired_thermal import PairedThermalDataset
from torch.utils.data import DataLoader
from utils_thermal import svd_orthogonalization, close_logger, init_logging, \
						normalize_augment_pair, normalize_augment_pair_fixed, \
						get_const_noise_sigma_norm, mse_row_col_loss
from train_res_np_common import resume_training, lr_scheduler, log_train_psnr, \
						validate_and_log, save_model_checkpoint

import atexit
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


def main(**args):
    
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group(backend="nccl")
    dist.barrier()
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    print(f"Rank {rank} using GPU {local_rank}")


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
    
	train_sampler = DistributedSampler(
        dataset_train,
        num_replicas=world_size,
        rank=rank,
        shuffle=True             #check
    )

    loader_train = DataLoader(
        dataset_train,
        batch_size=args['batch_size'],
        sampler=train_sampler,
        num_workers=8,
        pin_memory=True
    )

	num_minibatches = len(loader_train)
	ctrl_fr_idx = (args['temp_patch_size'] - 1) // 2
	print("\t# of training samples: %d\n" % int(args['max_number_patches']))

	#writer, logger = init_logging(args)
    if rank == 0:
        writer, logger = init_logging(args)
    else:
        writer, logger = None, None
	
    torch.backends.cudnn.benchmark = True

	#device_ids = list(range(torch.cuda.device_count()))
	#device0 = f'cuda:{device_ids[0]}' if device_ids else 'cuda:0'

	#model = FastDVDnet()
	#if len(device_ids) > 1:
		#model = nn.DataParallel(model, device_ids=device_ids).to(device0)
	#else:
		#model = model.to(device0)
    
    model = FastDVDnet().to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

	optimizer = optim.Adam(model.parameters(), lr=args['lr'])

	start_epoch, training_params = resume_training(
        args,
        model.module,
        optimizer,
        local_rank
    )

	start_time = time.time()
	for epoch in range(start_epoch, args['epochs']):
        train_sampler.set_epoch(epoch)
		epoch_start = time.time()
		current_lr, reset_orthog = lr_scheduler(epoch, args)
		if reset_orthog:
			training_params['no_orthog'] = True

		for param_group in optimizer.param_groups:
			param_group["lr"] = current_lr

		epoch_loss = 0.0
		epoch_batches = 0
		for i, data in enumerate(loader_train, 0):
			model.train()
			optimizer.zero_grad()

			noisy_seq, clean_seq = data

			# --- normalization (toggle via --norm_mode) ---
			if args['norm_mode'] == 'minmax':
				imgn_train, gt_train = normalize_augment_pair(noisy_seq, clean_seq, ctrl_fr_idx)
			elif args['norm_mode'] == 'fixed':
				imgn_train, gt_train = normalize_augment_pair_fixed(
					noisy_seq, clean_seq, ctrl_fr_idx, max_val=args['max_val'])
			else:
				raise ValueError(f"Unknown norm_mode: {args['norm_mode']}")

			N, _, H, W = imgn_train.size()

            imgn_train = imgn_train.to(device, non_blocking=True)
            gt_train = gt_train.to(device, non_blocking=True)

			# --- noise map (toggle via --noise_map_mode) ---
			if args['noise_map_mode'] == 'residual':
				ctrl_ch = ctrl_fr_idx * 3
				noisy_center = imgn_train[:, ctrl_ch:ctrl_ch+3, :, :]      # [B,3,H,W]
				noisy_center_single = noisy_center[:, :1, :, :]            # [B,1,H,W]
				gt_single = gt_train[:, :1, :, :]                          # [B,1,H,W]
				residual = noisy_center_single - gt_single
				noise_map = residual.abs().clamp(min=1e-6).to(imgn_train.dtype).to(imgn_train.device)
			elif args['noise_map_mode'] == 'constant':
				sigma_norm = get_const_noise_sigma_norm(
					args['norm_mode'], args['const_sigma'], args['max_val'])
				noise_map = torch.full((N, 1, H, W), float(sigma_norm),
									   dtype=imgn_train.dtype, device=imgn_train.device)
				noise_map = noise_map.clamp(min=1e-6)
			else:
				raise ValueError(f"Unknown noise_map_mode: {args['noise_map_mode']}")

			out_train = model(imgn_train, noise_map)

			# MSE stays the base objective; row/col terms attack stripe bias only.
			# All terms share one scale so the weights stay interpretable as a
			# fraction of per-pixel MSE. 'legacy' reproduces FastDVDnet's
			# sum/(N*2) gradient magnitude so the usual LR still applies.
			if args['loss_scale_mode'] == 'legacy':
				loss_scale = out_train[0].numel() / 2.0
			else:
				loss_scale = 1.0

			loss, loss_parts = mse_row_col_loss(
				out_train,
				gt_train,
				row_weight=args['row_loss_weight'],
				col_weight=args['col_loss_weight'],
			)
			loss = loss * loss_scale

			loss.backward()
			optimizer.step()
			epoch_loss += loss.item()
			epoch_batches += 1

			if training_params['step'] % args['save_every'] == 0:

            if not training_params['no_orthog'\]:
                model.module.apply(svd_orthogonalization)

            if rank == 0:
                writer.add_scalar('Loss/mse', loss_parts['mse'].item(), training_params['step'])
                writer.add_scalar('Loss/row', loss_parts['row'].item(), training_params['step'])
                writer.add_scalar('Loss/col', loss_parts['col'].item(), training_params['step'])
                writer.add_scalar('Loss/total', loss.item(), training_params['step'])

                log_train_psnr(
                    out_train,
                    gt_train,
                    loss,
                    writer,
                    epoch,
                    i,
                    num_minibatches,
                    training_params
                )

			training_params['step'] += 1

		if rank == 0 and epoch_batches > 0:
			avg_loss = epoch_loss / epoch_batches
			writer.add_scalar('Loss/train', avg_loss, epoch)
			writer.flush()
			print(f"[epoch {epoch+1}] avg train loss: {avg_loss:.6f}")

		epoch_elapsed = time.time() - epoch_start
		overall_elapsed = time.time() - start_time
		num_completed_epochs = epoch - start_epoch + 1
		avg_epoch_time = overall_elapsed / max(num_completed_epochs, 1)
		remaining_epochs = max(args['epochs'] - (epoch + 1), 0)
		eta_seconds = remaining_epochs * avg_epoch_time
		eta_hms = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
		#print(f"[epoch {epoch+1}/{args['epochs']}] elapsed: {epoch_elapsed:.1f}s | avg_epoch: {avg_epoch_time:.1f}s | ETA: {eta_hms}")
        if rank == 0:
            print(
                f"[epoch {epoch+1}/{args['epochs']}] "
                f"elapsed: {epoch_elapsed:.1f}s | "
                f"avg_epoch: {avg_epoch_time:.1f}s | "
                f"ETA: {eta_hms}"
            )

		dist.barrier()

        if rank == 0:

            model.eval()

            validate_and_log(
                model_temp=model.module,
                dataset_val=dataset_val,
                temp_psz=args['temp_patch_size'],
                writer=writer,
                epoch=epoch,
                lr=current_lr,
                logger=logger,
                trainimg=imgn_train,
                norm_mode=args['norm_mode'],
                noise_map_mode=args['noise_map_mode'],
                const_sigma=args['const_sigma'],
                max_val=args['max_val'],
            )

        dist.barrier()


		training_params['start_epoch'] = epoch + 1
		if rank == 0:
            save_model_checkpoint(
                model.module,
                args,
                optimizer,
                training_params,
                epoch
            )

	elapsed_time = time.time() - start_time
	print('Elapsed time {}'.format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))

	if rank == 0:
        close_logger(logger)
    
    dist.destroy_process_group()

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description="Train the denoiser with MSE + row/col stripe penalties")

	parser.add_argument("--batch_size", type=int, default=64)
	parser.add_argument("--epochs", "--e", type=int, default=80)
	parser.add_argument("--resume_training", "--r", action='store_true')
	parser.add_argument("--milestone", nargs=2, type=int, default=[50, 60])
	parser.add_argument("--lr", type=float, default=1e-3)
	parser.add_argument("--no_orthog", action='store_true')
	parser.add_argument("--save_every", type=int, default=10)
	parser.add_argument("--save_every_epochs", type=int, default=1)

	parser.add_argument("--row_loss_weight", type=float, default=0.2,
						help="Weight for the row-wise residual bias loss")
	parser.add_argument("--col_loss_weight", type=float, default=0.2,
						help="Weight for the column-wise residual bias loss")
	parser.add_argument("--loss_scale_mode", type=str, choices=["legacy", "mean"], default="legacy",
						help="'legacy' matches FastDVDnet's sum/(N*2) gradient magnitude; 'mean' uses raw per-pixel means")

	parser.add_argument("--patch_size", "--p", type=int, default=96)
	parser.add_argument("--temp_patch_size", "--tp", type=int, default=5)
	parser.add_argument("--max_number_patches", "--m", type=int, default=384000)

	parser.add_argument("--log_dir", type=str, default="logs")

	parser.add_argument("--train_noisy_dir", type=str, required=True)
	parser.add_argument("--train_clean_dir", type=str, required=True)
	parser.add_argument("--val_noisy_dir", type=str, default=None)
	parser.add_argument("--val_clean_dir", type=str, default=None)

	# --- ablation toggles ---
	parser.add_argument("--norm_mode", type=str, default="minmax",
						 choices=["minmax", "fixed"],
						 help="'minmax' = per-clip adaptive; 'fixed' = divide by max_val")
	parser.add_argument("--noise_map_mode", type=str, default="constant",
						 choices=["residual", "constant"],
						 help="'residual' = |noisy-clean| center frame; 'constant' = flat map")
	parser.add_argument("--const_sigma", type=float, default=25.0,
						 help="Raw-unit sigma used when --noise_map_mode constant")
	parser.add_argument("--max_val", type=float, default=65535.0,
						 help="Divisor for --norm_mode fixed / scale ref for constant sigma")

	argspar = parser.parse_args()

	log_dir = argspar.log_dir
	os.makedirs(log_dir, exist_ok=True)
	logfile = os.path.join(log_dir, f"train_{time.strftime('%Y%m%d-%H%M%S')}.log")

	f = open(logfile, 'a', buffering=1, encoding='utf-8')

	class Tee:
		def __init__(self, *streams):
			self.streams = streams
		def write(self, s):
			for st in self.streams:
				try: st.write(s)
				except Exception: pass
		def flush(self):
			for st in self.streams:
				try: st.flush()
				except Exception: pass
		def fileno(self):
			for st in reversed(self.streams):
				if hasattr(st, "fileno"):
					try: return st.fileno()
					except Exception: continue
			raise OSError("Tee: no fileno available")

	def _close_log():
		try:
			f.write(f"\n=== Training log ended: {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
			f.flush()
		except Exception:
			pass
		try: f.close()
		except Exception: pass

	sys.stdout = Tee(sys.__stdout__, f)
	sys.stderr = sys.stdout

	f.write(f"=== Training log started: {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
	f.flush()
	atexit.register(_close_log)

	print("\n### Training FastDVDnet denoiser (MSE + row/col stripe loss) ###")
	print("> Parameters:")
	for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
		print('\t{}: {}'.format(p, v))
	print('\n')

	main(**vars(argspar))
