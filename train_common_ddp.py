"""
Different common functions for training the models.
"""
import os
import time
import torch
import torchvision.utils as tutils
from utils_ddp import batch_psnr
from fastdvdnet import denoise_seq_fastdvdnet

# Resume training unchanged in behavior; caller should pass the *module* (unwrapped) model
def resume_training(argdict, model, optimizer):
    """ Resumes previous training or starts anew
    """
    if argdict['resume_training']:
        resumef = os.path.join(argdict['log_dir'], 'ckpt.pth')
        if os.path.isfile(resumef):
            # allow CPU map if needed
            checkpoint = torch.load(resumef, map_location='cpu')
            print("> Resuming previous training")
            # load into model (caller should pass an unwrapped module)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            new_epoch = argdict['epochs']
            new_milestone = argdict['milestone']
            current_lr = argdict['lr']
            argdict = checkpoint['args']
            training_params = checkpoint['training_params']
            start_epoch = training_params['start_epoch']
            argdict['epochs'] = new_epoch
            argdict['milestone'] = new_milestone
            argdict['lr'] = current_lr
            print("=> loaded checkpoint '{}' (epoch {})" \
                  .format(resumef, start_epoch))
            print("=> loaded parameters :")
            print("==> checkpoint['optimizer']['param_groups']")
            print("\t{}".format(checkpoint['optimizer']['param_groups']))
            print("==> checkpoint['training_params']")
            for k in checkpoint['training_params']:
                print("\t{}, {}".format(k, checkpoint['training_params'][k]))
            argpri = checkpoint['args']
            print("==> checkpoint['args']")
            for k in argpri:
                print("\t{}, {}".format(k, argpri[k]))

            argdict['resume_training'] = False
        else:
            raise Exception("Couldn't resume training with checkpoint {}".\
                   format(resumef))
    else:
        start_epoch = 0
        training_params = {}
        training_params['step'] = 0
        training_params['current_lr'] = 0
        training_params['no_orthog'] = argdict['no_orthog']

    return start_epoch, training_params

def lr_scheduler(epoch, argdict):
    """Returns the learning rate value depending on the actual epoch number"""
    reset_orthog = False
    if epoch > argdict['milestone'][1]:
        current_lr = argdict['lr'] / 1000.
        reset_orthog = True
    elif epoch > argdict['milestone'][0]:
        current_lr = argdict['lr'] / 10.
    else:
        current_lr = argdict['lr']
    return current_lr, reset_orthog

def log_train_psnr(result, imsource, loss, writer, epoch, idx, num_minibatches, training_params):
    '''Logs train loss and (optionally) PSNR'''
    if writer is not None:
        writer.add_scalar('loss', loss.item(), training_params['step'])
    print("[epoch {}][{}/{}] loss: {:1.4f}".\
          format(epoch+1, idx+1, num_minibatches, loss.item()))

def save_model_checkpoint(model, argdict, optimizer, train_pars, epoch):
    """Stores the model parameters under log_dir/net.pth and ckpt.pth
    model should be an nn.Module (unwrapped module.module when DDP-wrapped)"""
    torch.save(model.state_dict(), os.path.join(argdict['log_dir'], 'net.pth'))
    save_dict = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'training_params': train_pars,
        'args': argdict
    }
    torch.save(save_dict, os.path.join(argdict['log_dir'], 'ckpt.pth'))

    if epoch % argdict['save_every_epochs'] == 0:
        torch.save(save_dict, os.path.join(argdict['log_dir'], 'ckpt_e{}.pth'.format(epoch+1)))
    del save_dict

# validate_and_log updated to use model_temp device and not call .cuda()
def validate_and_log(model_temp, dataset_val, temp_psz, writer,
                     epoch, lr, logger, trainimg):
    """Validation step after each epoch.
    model_temp: unwrapped module (nn.Module) that is on the correct device.
    dataset_val: an iterable (Dataset). This function will move tensors to model_temp device.
    """
    device = None
    try:
        # get device from model parameters if available
        dev = next(model_temp.parameters()).device
        device = dev
    except StopIteration:
        # fallback
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    t1 = time.time()
    psnr_val = 0.0
    count = 0

    with torch.no_grad():
        for noisy_val, clean_val in dataset_val:
            noisy_val = noisy_val.float().to(device, non_blocking=True)
            clean_val = clean_val.float().to(device, non_blocking=True)

            # Normalize to [0, 1]
            noisy_val = noisy_val / 65535.0
            clean_val = clean_val / 65535.0

            # Use the center frame of the validation sequence
            ctrl_fr_idx = noisy_val.size(0) // 2
            noisy_central = noisy_val[ctrl_fr_idx]
            clean_central = clean_val[ctrl_fr_idx]

            # Compute per-pixel residual and noise map
            noisy_center_single = noisy_central[:1, :, :]   # (1, H, W)
            clean_center_single = clean_central[:1, :, :]   # (1, H, W)
            residual = (noisy_center_single - clean_center_single).abs()  # (1, H, W)

            # make into shape (1, 1, H, W)
            noise_map = residual.unsqueeze(0).clamp(min=1e-6).to(device, dtype=noisy_val.dtype)

            # Denoise the full sequence
            out_val = denoise_seq_fastdvdnet(
                seq=noisy_val,
                noise_std=noise_map,
                temp_psz=temp_psz,
                model_temporal=model_temp,
            )

            psnr_val += batch_psnr(out_val.cpu(), clean_val.cpu(), 1.)
            count += 1

    if count > 0:
        psnr_val = psnr_val / count
    else:
        psnr_val = 0.0
    t2 = time.time()

    print("\n[epoch %d] PSNR_val: %.4f, on %.2f sec" % (epoch + 1, psnr_val, (t2 - t1)))
    if writer is not None:
        writer.add_scalar('PSNR on validation data', psnr_val, epoch)
        writer.add_scalar('Learning rate', lr, epoch)

    # Log images from the last validation batch (if writer available)
    try:
        if writer is None:
            return

        idx = 0
        if epoch == 0:
            _, _, Ht, Wt = trainimg.size()
            img = tutils.make_grid(
                trainimg.view(-1, 3, Ht, Wt),
                nrow=8, normalize=True, scale_each=True
            )
            writer.add_image('Training patches', img, epoch)

            img_clean = tutils.make_grid(
                clean_val.data[idx].clamp(0., 1.).cpu(),
                nrow=2, normalize=False, scale_each=False
            )
            img_noisy = tutils.make_grid(
                noisy_val.data[idx].clamp(0., 1.).cpu(),
                nrow=2, normalize=False, scale_each=False
            )
            writer.add_image('Clean validation image {}'.format(idx), img_clean, epoch)
            writer.add_image('Noisy validation image {}'.format(idx), img_noisy, epoch)

        irecon = tutils.make_grid(
            out_val.data[idx].clamp(0., 1.).cpu(),
            nrow=2, normalize=False, scale_each=False
        )
        writer.add_image('Reconstructed validation image {}'.format(idx), irecon, epoch)

    except Exception as e:
        if logger is not None:
            logger.error("validate_and_log_temporal(): Couldn't log results, {}".format(e))