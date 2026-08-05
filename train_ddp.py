"""
Trains a FastDVDnet model (DDP-enabled).
"""
import time
import os
import sys
import argparse
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from models import FastDVDnet
from dataset_val_ddp import PairedValDataset
from dataset_ddp import PairedThermalDataset
from utils_ddp import svd_orthogonalization, close_logger, init_logging, normalize_augment_pair
from train_common_ddp import resume_training, lr_scheduler, log_train_psnr, \
                    validate_and_log, save_model_checkpoint

import atexit

"""
Trains a FastDVDnet model (DDP-enabled).
"""

def _maybe_setup_stdout_log(log_dir, is_main):
    """Create a tee to duplicate stdout/stderr to a file, only on main process."""
    if not is_main:
        return None
    os.makedirs(log_dir, exist_ok=True)
    logfile = os.path.join(log_dir, f"train_{time.strftime('%Y%m%d-%H%M%S')}.log")
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
            for st in reversed(self.streams):
                if hasattr(st, "fileno"):
                    try:
                        return st.fileno()
                    except Exception:
                        continue
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

    sys.stdout = Tee(sys.__stdout__, f)
    sys.stderr = sys.stdout

    f.write(f"=== Training log started: {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    f.flush()
    atexit.register(_close_log)
    return f

def main(**args):
    # Distributed setup (torchrun / torch.distributed.run sets LOCAL_RANK, WORLD_SIZE)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_distributed = world_size > 1

    if is_distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    is_main = (not is_distributed) or (dist.get_rank() == 0)

    # seed per-process for reproducibility (optional)
    base_seed = args.get('seed', 42)
    seed = base_seed + (dist.get_rank() if is_distributed else 0)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print('> Loading datasets ...')

    # validation dataset (kept on CPU, validation run only on main rank by default)
    dataset_val = PairedValDataset(
        noisy_root=args['val_noisy_dir'],
        clean_root=args['val_clean_dir']
    )

    # For distributed runs, avoid preloading entire dataset in each process (memory duplication)
    #preload_flag = False if is_distributed else True

    dataset_train = PairedThermalDataset(
        noisy_root=args['train_noisy_dir'],
        clean_root=args['train_clean_dir'],
        patch_size=args['patch_size'],
        temp_patch_size=args['temp_patch_size'],
        epoch_size=args['max_number_patches'],
    )

    # Sampler / DataLoader
    if is_distributed:
        train_sampler = DistributedSampler(dataset_train, num_replicas=world_size, rank=dist.get_rank(), shuffle=True)
        loader_train = DataLoader(
            dataset_train,
            batch_size=args['batch_size'],   # batch_size is per-process (per-GPU) when using torchrun
            sampler=train_sampler,
            num_workers=args['num_workers'],
            pin_memory=args['pin_memory'],
            drop_last=True
        )
    else:
        train_sampler = None
        loader_train = DataLoader(
            dataset_train,
            batch_size=args['batch_size'],
            shuffle=True,
            num_workers=args['num_workers'],
            pin_memory=args['pin_memory']
        )

    num_minibatches = int(args['max_number_patches'] // args['batch_size'])
    ctrl_fr_idx = (args['temp_patch_size'] - 1) // 2
    if is_main:
        print("\t# of training samples: %d\n" % int(args['max_number_patches']))

    # Init logging / Tee only on main
    tee_file = _maybe_setup_stdout_log(args['log_dir'], is_main)
    if is_main:
        writer, logger = init_logging(args)
    else:
        writer, logger = None, None

    # Devices and model
    torch.backends.cudnn.benchmark = True  # CUDNN optimization

    # Create model (single-module, on local device)
    model = FastDVDnet().to(device)

    # Define loss and optimizer
    criterion = nn.MSELoss(reduction='sum').to(device)
    optimizer = optim.Adam(model.parameters(), lr=args['lr'])

    # Resume training or start anew (operate on module)
    start_epoch, training_params = resume_training(args, model, optimizer)

    # Wrap with DDP if distributed
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    # Training
    start_time = time.time()
    if is_main:
        print("DEBUG: torch.cuda.device_count()", torch.cuda.device_count())
        print("DEBUG: model device:", device)
    for epoch in range(start_epoch, args['epochs']):
        if is_distributed:
            train_sampler.set_epoch(epoch)

        # Set learning rate
        current_lr, reset_orthog = lr_scheduler(epoch, args)
        if reset_orthog:
            training_params['no_orthog'] = True

        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        if is_main:
            print('\nlearning rate %f' % current_lr)

        # train
        for i, data in enumerate(loader_train, 0):
            if is_main and (i % 50 == 0):
                print(f"DEBUG: start batch {i}")

            model.train()
            optimizer.zero_grad()

            noisy_seq, clean_seq = data

            # normalize + apply SAME augmentation to noisy and clean (still CPU tensors)
            imgn_train, gt_train = normalize_augment_pair(noisy_seq, clean_seq, ctrl_fr_idx)
            N, _, H, W = imgn_train.size()

            # Move to local device (each process must move its own tensors)
            imgn_train = imgn_train.to(device, non_blocking=True)
            gt_train = gt_train.to(device, non_blocking=True)

            # center noisy frame
            noisy_central = imgn_train[:, 3 * ctrl_fr_idx:3 * ctrl_fr_idx + 3, :, :]

            # noise_map = residual
            start_ch = ctrl_fr_idx * 3
            noisy_center = imgn_train[:, start_ch:start_ch + 3, :, :]   # [B,3,H,W]
            noisy_center_single = noisy_center[:, :1, :, :]             # [B,1,H,W]

            gt_single = gt_train[:, :1, :, :]

            residual = noisy_center_single - gt_single
            noise_map = residual.abs()    # shape [B,1,H,W]
            noise_map = noise_map.clamp(min=1e-6).to(imgn_train.dtype).to(device)

            # Evaluate model and optimize it
            out_train = model(imgn_train, noise_map)

            # Compute loss
            loss = criterion(gt_train, out_train) / (N * 2)
            loss.backward()
            optimizer.step()

            # Results (only main process logs and applies orthogonalization)
            if training_params['step'] % args['save_every'] == 0:
                if not training_params['no_orthog']:
                    if hasattr(model, "module"):
                        model.module.apply(svd_orthogonalization)
                    else:
                        model.apply(svd_orthogonalization)

                if is_main and writer is not None:
                    log_train_psnr(out_train, gt_train, loss, writer, epoch, i, num_minibatches, training_params)

            training_params['step'] += 1

        # eval
        if hasattr(model, "eval"):
            model.eval()

        # Validation and log images (run only on main process to avoid duplication)
        if is_main:
            validate_and_log(
                model_temp=(model.module if hasattr(model, "module") else model),
                dataset_val=dataset_val,
                temp_psz=args['temp_patch_size'],
                writer=writer,
                epoch=epoch,
                lr=current_lr,
                logger=logger,
                trainimg=imgn_train
            )

        # save model and checkpoint (only main)
        training_params['start_epoch'] = epoch + 1
        if is_main:
            save_model_checkpoint((model.module if hasattr(model, "module") else model), args, optimizer, training_params, epoch)

    # Print elapsed time (only main)
    if is_main:
        elapsed_time = time.time() - start_time
        print('Elapsed time {}'.format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))
        close_logger(logger)

    # cleanup distributed
    if is_distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train the denoiser")

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Training batch size (per-process when using DDP)")
    parser.add_argument("--epochs", "--e", type=int, default=80,
                        help="Number of total training epochs")
    parser.add_argument("--resume_training", "--r", action='store_true',
                        help="resume training from a previous checkpoint")
    parser.add_argument("--milestone", nargs=2, type=int, default=[50, 60],
                        help="When to decay learning rate; should be lower than 'epochs'")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Initial learning rate")
    parser.add_argument("--no_orthog", action='store_true',
                        help="Don't perform orthogonalization as regularization")
    parser.add_argument("--save_every", type=int, default=10,
                        help="Number of training steps to log psnr and perform orthogonalization")
    parser.add_argument("--save_every_epochs", type=int, default=5,
                        help="Number of training epochs to save state")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader num_workers")
    parser.add_argument("--pin_memory", action='store_true', help="Use pin_memory in DataLoader")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Preprocessing parameters
    parser.add_argument("--patch_size", "--p", type=int, default=96, help="Patch size")
    parser.add_argument("--temp_patch_size", "--tp", type=int, default=5, help="Temporal patch size")
    parser.add_argument("--max_number_patches", "--m", type=int, default=384000, help="Maximum number of patches")

    # Dirs
    parser.add_argument("--log_dir", type=str, default="logs", help='path of log files')

    # Paths to paired dataset
    parser.add_argument("--train_noisy_dir", type=str, required=True, help="path to noisy training images")
    parser.add_argument("--train_clean_dir", type=str, required=True, help="path to clean training images")
    parser.add_argument("--val_noisy_dir", type=str, default=None, help='path to noisy validation images')
    parser.add_argument("--val_clean_dir", type=str, default=None, help='path to clean validation images')

    argspar = parser.parse_args()

    # NOTE: logging (file tee) is now handled inside main() only on the main rank (rank 0)
    main(**vars(argspar))