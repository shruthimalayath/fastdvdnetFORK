"""
Different utilities such as orthogonalization of weights, initialization of
loggers, etc
"""
import os
import subprocess
import glob
import logging
from random import choices # requires Python >= 3.6
import numpy as np
import cv2
import torch
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from tensorboardX import SummaryWriter

IMAGETYPES = ('*.bmp', '*.png', '*.jpg', '*.jpeg', '*.tif') # Supported image types


#Normalization & augmentation for pairs: ensures same transformations are applied to both
def normalize_augment_pair(noisy_seq, clean_seq, ctrl_fr_idx):
    """
    Args:
        noisy_seq: [N, num_frames, C, H, W]
        clean_seq: [N, num_frames, C, H, W]

    Returns:
        imgn_train: [N, num_frames*C, H, W]
        gt_train:   [N, 3, H, W]
    """

    def get_transform():
        aug_list = [
            lambda x: x,
            lambda x: torch.flip(x, dims=[2]),
            lambda x: torch.rot90(x, k=1, dims=[2, 3]),
            lambda x: torch.flip(torch.rot90(x, k=1, dims=[2, 3]), dims=[2]),
            lambda x: torch.rot90(x, k=2, dims=[2, 3]),
            lambda x: torch.flip(torch.rot90(x, k=2, dims=[2, 3]), dims=[2]),
            lambda x: torch.rot90(x, k=3, dims=[2, 3]),
            lambda x: torch.flip(torch.rot90(x, k=3, dims=[2, 3]), dims=[2]),
        ]

        w_aug = [32, 12, 12, 12, 12, 12, 12, 12]
        return choices(aug_list, w_aug)[0]

    # normalize both noisy and clean using the same min/max values per sample
    noisy_seq, clean_seq, minv, maxv = minmax_normalize_pair(noisy_seq, clean_seq)

    noisy_seq = noisy_seq.reshape(noisy_seq.size(0), -1, noisy_seq.size(-2), noisy_seq.size(-1))
    clean_seq = clean_seq.reshape(clean_seq.size(0), -1, clean_seq.size(-2), clean_seq.size(-1))

    # flatten temporal dimension: issue continguous vs non contiguous tensor
    #noisy_seq = noisy_seq.view(noisy_seq.size(0),-1,noisy_seq.size(-2),noisy_seq.size(-1))
    #clean_seq = clean_seq.view(clean_seq.size(0),-1,clean_seq.size(-2),clean_seq.size(-1))

    # choose one augmentation
    transf = get_transform()

    # apply same augmentation to both
    noisy_seq = transf(noisy_seq)
    clean_seq = transf(clean_seq)

    # extract clean center frame
    gt_train = clean_seq[ :,3*ctrl_fr_idx:3*ctrl_fr_idx+3,:,:]

    return noisy_seq, gt_train

def normalize_augment(datain, ctrl_fr_idx):
	'''Normalizes and augments an input patch of dim [N, num_frames, C. H, W] in [0., 255.] to \
		[N, num_frames*C. H, W] in  [0., 1.]. It also returns the central frame of the temporal \
		patch as a ground truth.
	'''
	def transform(sample):
		# define transformations
		do_nothing = lambda x: x
		do_nothing.__name__ = 'do_nothing'
		flipud = lambda x: torch.flip(x, dims=[2])
		flipud.__name__ = 'flipup'
		rot90 = lambda x: torch.rot90(x, k=1, dims=[2, 3])
		rot90.__name__ = 'rot90'
		rot90_flipud = lambda x: torch.flip(torch.rot90(x, k=1, dims=[2, 3]), dims=[2])
		rot90_flipud.__name__ = 'rot90_flipud'
		rot180 = lambda x: torch.rot90(x, k=2, dims=[2, 3])
		rot180.__name__ = 'rot180'
		rot180_flipud = lambda x: torch.flip(torch.rot90(x, k=2, dims=[2, 3]), dims=[2])
		rot180_flipud.__name__ = 'rot180_flipud'
		rot270 = lambda x: torch.rot90(x, k=3, dims=[2, 3])
		rot270.__name__ = 'rot270'
		rot270_flipud = lambda x: torch.flip(torch.rot90(x, k=3, dims=[2, 3]), dims=[2])
		rot270_flipud.__name__ = 'rot270_flipud'
		add_csnt = lambda x: x + torch.normal(mean=torch.zeros(x.size()[0], 1, 1, 1), \
								 std=(5/65535.)).expand_as(x).to(x.device)
		add_csnt.__name__ = 'add_csnt'

		# define transformations and their frequency, then pick one.
		aug_list = [do_nothing, flipud, rot90, rot90_flipud, \
					rot180, rot180_flipud, rot270, rot270_flipud, add_csnt]
		w_aug = [32, 12, 12, 12, 12, 12, 12, 12, 12] # one fourth chances to do_nothing
		transf = choices(aug_list, w_aug)

		# transform all images in array
		return transf[0](sample)

	img_train = datain

	# convert to [N, num_frames*C. H, W] in  [0., 1.] from [N, num_frames, C. H, W] in [0., 65535.]
	img_train = img_train.view(img_train.size()[0], -1, \
							   img_train.size()[-2], img_train.size()[-1]) / 65535.
	
	#augment
	img_train = transform(img_train)

	# extract ground truth (central frame)
	gt_train = img_train[:, 3*ctrl_fr_idx:3*ctrl_fr_idx+3, :, :]
	return img_train, gt_train

def init_logging(argdict):
	"""Initilizes the logging and the SummaryWriter modules
	"""
	if not os.path.exists(argdict['log_dir']):
		os.makedirs(argdict['log_dir'])
	writer = SummaryWriter(argdict['log_dir'])
	logger = init_logger(argdict['log_dir'], argdict)
	return writer, logger

def get_imagenames(seq_dir, pattern=None):
	""" Get ordered list of filenames
	"""
	files = []
	for typ in IMAGETYPES:
		files.extend(glob.glob(os.path.join(seq_dir, typ)))

	# filter filenames
	if not pattern is None:
		ffiltered = []
		ffiltered = [f for f in files if pattern in os.path.split(f)[-1]]
		files = ffiltered
		del ffiltered

	# sort filenames alphabetically
	files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
	return files

def open_sequence(seq_dir, gray_mode, expand_if_needed=False, max_num_fr=100):
	r""" Opens a sequence of images and expands it to even sizes if necesary
	Args:
		fpath: string, path to image sequence
		gray_mode: boolean, True indicating if images is to be open are in grayscale mode
		expand_if_needed: if True, the spatial dimensions will be expanded if
			size is odd
		expand_axis0: if True, output will have a fourth dimension
		max_num_fr: maximum number of frames to load
	Returns:
		seq: array of dims [num_frames, C, H, W], C=1 grayscale or C=3 RGB, H and W are even.
			The image gets normalized gets normalized to the range [0, 1].
		expanded_h: True if original dim H was odd and image got expanded in this dimension.
		expanded_w: True if original dim W was odd and image got expanded in this dimension.
	"""
	# Get ordered list of filenames
	files = get_imagenames(seq_dir)

	seq_list = []
	print("\tOpen sequence in folder: ", seq_dir)
	for fpath in files[0:max_num_fr]:

		img, expanded_h, expanded_w = open_image(fpath,\
												   gray_mode=gray_mode,\
												   expand_if_needed=expand_if_needed,\
												   expand_axis0=False)
		seq_list.append(img)
	seq = np.stack(seq_list, axis=0)
	return seq, expanded_h, expanded_w

def open_image(fpath, gray_mode, expand_if_needed=False, expand_axis0=True, normalize_data=True):
	r""" Opens an image and expands it if necesary
	Args:
		fpath: string, path of image file
		gray_mode: boolean, True indicating if image is to be open
			in grayscale mode
		expand_if_needed: if True, the spatial dimensions will be expanded if
			size is odd
		expand_axis0: if True, output will have a fourth dimension
	Returns:
		img: image of dims NxCxHxW, N=1, C=1 grayscale or C=3 RGB, H and W are even.
			if expand_axis0=False, the output will have a shape CxHxW.
			The image gets normalized gets normalized to the range [0, 1].
		expanded_h: True if original dim H was odd and image got expanded in this dimension.
		expanded_w: True if original dim W was odd and image got expanded in this dimension.
	"""
	if not gray_mode:
		# Open image as a CxHxW torch.Tensor
		img = cv2.imread(fpath)
		# from HxWxC to CxHxW, RGB image
		img = (cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).transpose(2, 0, 1)
	else:
		# from HxWxC to CxHxW grayscale image (C=1)
		#img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
		img = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
		img = np.stack([img, img, img], axis=0)
		#img = np.expand_dims(img, 0)

	if expand_axis0:
		img = np.expand_dims(img, 0)

	# Handle odd sizes
	expanded_h = False
	expanded_w = False
	sh_im = img.shape
	if expand_if_needed:
		if sh_im[-2]%2 == 1:
			expanded_h = True
			if expand_axis0:
				img = np.concatenate((img, \
					img[:, :, -1, :][:, :, np.newaxis, :]), axis=2)
			else:
				img = np.concatenate((img, \
					img[:, -1, :][:, np.newaxis, :]), axis=1)


		if sh_im[-1]%2 == 1:
			expanded_w = True
			if expand_axis0:
				img = np.concatenate((img, \
					img[:, :, :, -1][:, :, :, np.newaxis]), axis=3)
			else:
				img = np.concatenate((img, \
					img[:, :, -1][:, :, np.newaxis]), axis=2)

	if normalize_data:
		img = normalize(img)
	print("single image shape =", img.shape)
	return img, expanded_h, expanded_w

def batch_psnr(img, imclean, data_range):
	r"""
	Computes the PSNR along the batch dimension (not pixel-wise)

	Args:
		img: a `torch.Tensor` containing the restored image
		imclean: a `torch.Tensor` containing the reference image
		data_range: The data range of the input image (distance between
			minimum and maximum possible values). By default, this is estimated
			from the image data-type.
	"""
	img_cpu = img.data.cpu().numpy().astype(np.float32)
	imgclean = imclean.data.cpu().numpy().astype(np.float32)
	psnr = 0
	for i in range(img_cpu.shape[0]):
		psnr += compare_psnr(imgclean[i, :, :, :], img_cpu[i, :, :, :], \
					   data_range=data_range)
	return psnr/img_cpu.shape[0]

def variable_to_cv2_image(invar, conv_rgb_to_bgr=True, orig_min=None, orig_max=None, eps=1e-6):
    """Converts a torch tensor to an OpenCV image.

    Args:
        invar: a torch.Tensor with values in [0, 1] (unless orig_min/orig_max are provided)
        conv_rgb_to_bgr: boolean. If True, convert output image from RGB to BGR color space
        orig_min: optional scalar or tensor with the original minimum value used for min-max normalization
        orig_max: optional scalar or tensor with the original maximum value used for min-max normalization
        eps: not used for denormalize here; kept for compatibility
    Returns:
        a HxWxC uint16 image
    """
    x = invar.data.cpu()
    # If orig_min/orig_max are provided, denormalize first. We expect orig_min/orig_max
    # to be in the same units as the original raw values (e.g., uint16 sensor units).
    if orig_min is not None and orig_max is not None:
        x = minmax_denormalize(x, orig_min, orig_max, eps=eps)
        # At this point x should already be in raw units (e.g. ~0..65535). Do NOT multiply further.
        x = x.numpy().astype(np.float32)
        size4 = x.ndim == 4
        if size4:
            nchannels = x.shape[1]
        else:
            nchannels = x.shape[0]

        if nchannels == 1:
            if size4:
                res = x[0, 0, :]
            else:
                res = x[0, :]
            res = np.clip(res, 0, 65535).astype(np.uint16)
        elif nchannels == 3:
            if size4:
                res = x[0]
            else:
                res = x
            res = res.transpose(1, 2, 0)
            res = np.clip(res, 0, 65535).astype(np.uint16)
            if conv_rgb_to_bgr:
                res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
        else:
            raise Exception('Number of color channels not supported')
        return res
    else:
        # Expect input is normalized [0,1]: multiply to uint16 range
        # only assert in that case
        assert torch.max(invar) <= 1.0 + 1e-6
        x = invar.data.cpu().float().numpy().astype(np.float32)
        size4 = x.ndim == 4
        if size4:
            nchannels = x.shape[1]
        else:
            nchannels = x.shape[0]

        if nchannels == 1:
            if size4:
                res = x[0, 0, :]
            else:
                res = x[0, :]
            res = (res * 65535.).clip(0, 65535).astype(np.uint16)
                
        elif nchannels == 3:
            if size4:
                res_float = x[0]   # shape (C,H,W)
            else:
                res_float = x      # shape (C,H,W)

            # Convert to HWC float for inspection
            res_float_hwc = res_float.transpose(1, 2, 0)  # (H, W, 3)

            # If all three channels are nearly identical, treat as grayscale and save single channel
            # Use a tolerance appropriate to your data: use a small tol on normalized floats
            tol = 1e-6
            ch0 = res_float_hwc[:, :, 0]
            ch1 = res_float_hwc[:, :, 1]
            ch2 = res_float_hwc[:, :, 2]
            if np.allclose(ch0, ch1, atol=tol) and np.allclose(ch0, ch2, atol=tol):
                # Single-channel output (take first channel), convert to uint16
                res = np.clip(ch0, 0, 65535).astype(np.float32)
                res = np.rint(res).astype(np.uint16)
                return res
            else:
                # Color image: scale/clamp and convert to uint16 then RGB->BGR for cv2
                res = (res_float_hwc).clip(0, 65535).astype(np.float32)
                res = np.rint(res).astype(np.uint16)
                if conv_rgb_to_bgr:
                    res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
                return res
        else:
            raise Exception('Number of color channels not supported')
        return res

def get_git_revision_short_hash():
	r"""Returns the current Git commit.
	"""
	return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip()

def init_logger(log_dir, argdict):
	r"""Initializes a logging.Logger to save all the running parameters to a
	log file

	Args:
		log_dir: path in which to save log.txt
		argdict: dictionary of parameters to be logged
	"""
	from os.path import join

	logger = logging.getLogger(__name__)
	logger.setLevel(level=logging.INFO)
	fh = logging.FileHandler(join(log_dir, 'log.txt'), mode='w+')
	formatter = logging.Formatter('%(asctime)s - %(message)s')
	fh.setFormatter(formatter)
	logger.addHandler(fh)
	try:
		logger.info("Commit: {}".format(get_git_revision_short_hash()))
	except Exception as e:
		logger.error("Couldn't get commit number: {}".format(e))
	logger.info("Arguments: ")
	for k in argdict.keys():
		logger.info("\t{}: {}".format(k, argdict[k]))

	return logger

def init_logger_test(result_dir):
	r"""Initializes a logging.Logger in order to log the results after testing
	a model

	Args:
		result_dir: path to the folder with the denoising results
	"""
	from os.path import join

	logger = logging.getLogger('testlog')
	logger.setLevel(level=logging.INFO)
	fh = logging.FileHandler(join(result_dir, 'log.txt'), mode='w+')
	formatter = logging.Formatter('%(asctime)s - %(message)s')
	fh.setFormatter(formatter)
	logger.addHandler(fh)

	return logger

def close_logger(logger):
	'''Closes the logger instance
	'''
	x = list(logger.handlers)
	for i in x:
		logger.removeHandler(i)
		i.flush()
		i.close()

def normalize(data, eps=1e-6):
	r"""Normalizes an image to the range [0, 1] using min-max scaling.

	Args:
		data: a numpy array or torch tensor
		eps: small epsilon to avoid division by zero
	Returns:
		float32 tensor or numpy array in [0, 1]
	"""
	return np.float32(minmax_normalize(data, eps=eps))


def minmax_normalize_pair(noisy_seq, clean_seq, eps=1e-6, dims=None):
	"""Normalize a noisy/clean pair using the same per-sample min/max range."""
	if torch.is_tensor(noisy_seq) and torch.is_tensor(clean_seq):
		if dims is None:
			if noisy_seq.ndim == 5:
				dims = (1, 2, 3, 4)
			elif noisy_seq.ndim == 4:
				dims = (1, 2, 3)
			elif noisy_seq.ndim == 3:
				dims = (1, 2)
			else:
				dims = tuple(range(1, noisy_seq.ndim))
		combined = torch.cat((noisy_seq, clean_seq), dim=1)
		minv = combined.amin(dim=dims, keepdim=True)
		maxv = combined.amax(dim=dims, keepdim=True)
		noisy_norm = (noisy_seq.float() - minv) / (maxv - minv + eps)
		clean_norm = (clean_seq.float() - minv) / (maxv - minv + eps)
		return noisy_norm, clean_norm, minv, maxv
	else:
		noisy_seq = np.array(noisy_seq, dtype=np.float32)
		clean_seq = np.array(clean_seq, dtype=np.float32)
		if dims is None:
			if noisy_seq.ndim == 5:
				dims = (1, 2, 3, 4)
			elif noisy_seq.ndim == 4:
				dims = (1, 2, 3)
			elif noisy_seq.ndim == 3:
				dims = (1, 2)
			else:
				dims = tuple(range(1, noisy_seq.ndim))
		combined = np.concatenate((noisy_seq, clean_seq), axis=1)
		minv = combined.min(axis=dims, keepdims=True)
		maxv = combined.max(axis=dims, keepdims=True)
		noisy_norm = (noisy_seq - minv) / (maxv - minv + eps)
		clean_norm = (clean_seq - minv) / (maxv - minv + eps)
		return noisy_norm, clean_norm, minv, maxv


def minmax_denormalize(x, orig_min, orig_max, eps=1e-6):
    """Undo min-max normalization using stored min/max values."""
    if torch.is_tensor(x):
        x = x.float()
        if not torch.is_tensor(orig_min):
            orig_min = torch.tensor(orig_min, dtype=x.dtype, device=x.device)
        else:
            orig_min = orig_min.to(x.device)
        if not torch.is_tensor(orig_max):
            orig_max = torch.tensor(orig_max, dtype=x.dtype, device=x.device)
        else:
            orig_max = orig_max.to(x.device)

        scale = orig_max - orig_min
        # avoid division-by-zero / zero-scale by setting zero scales to 1.0
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        return x * scale + orig_min
    else:
        x = np.array(x, dtype=np.float32)
        orig_min = np.array(orig_min, dtype=np.float32)
        orig_max = np.array(orig_max, dtype=np.float32)
        scale = orig_max - orig_min
        scale = np.where(scale == 0, 1.0, scale)
        return x * scale + orig_min


def svd_orthogonalization(lyr):
	r"""Applies regularization to the training by performing the
	orthogonalization technique described in the paper "An Analysis and Implementation of
	the FFDNet Image Denoising Method." Tassano et al. (2019).
	For each Conv layer in the model, the method replaces the matrix whose columns
	are the filters of the layer by new filters which are orthogonal to each other.
	This is achieved by setting the singular values of a SVD decomposition to 1.

	This function is to be called by the torch.nn.Module.apply() method,
	which applies svd_orthogonalization() to every layer of the model.
	"""
	classname = lyr.__class__.__name__
	if classname.find('Conv') != -1:
		weights = lyr.weight.data.clone()
		c_out, c_in, f1, f2 = weights.size()
		dtype = lyr.weight.data.type()

		# Reshape filters to columns
		# From (c_out, c_in, f1, f2)  to (f1*f2*c_in, c_out)
		weights = weights.permute(2, 3, 1, 0).contiguous().view(f1*f2*c_in, c_out)

		try:
			# SVD decomposition and orthogonalization
			mat_u, _, mat_v = torch.svd(weights)
			weights = torch.mm(mat_u, mat_v.t())

			lyr.weight.data = weights.view(f1, f2, c_in, c_out).permute(3, 2, 0, 1).contiguous().type(dtype)
		except:
			pass
	else:
		pass

def remove_dataparallel_wrapper(state_dict):
	r"""Converts a DataParallel model to a normal one by removing the "module."
	wrapper in the module dictionary


	Args:
		state_dict: a torch.nn.DataParallel state dictionary
	"""
	from collections import OrderedDict

	new_state_dict = OrderedDict()
	for k, v in state_dict.items():
		name = k[7:] # remove 'module.' of DataParallel
		new_state_dict[name] = v

	return new_state_dict


def minmax_normalize(x, eps=1e-6, dims=None, return_stats=False):
    """Apply min-max normalization across the spatial/channel dimensions."""
    if torch.is_tensor(x):
        x = x.float()
        if dims is None:
            if x.ndim == 5:
                dims = (1, 2, 3, 4)
            elif x.ndim == 4:
                dims = (1, 2, 3)
            elif x.ndim == 3:
                dims = (1, 2)
            else:
                dims = tuple(range(1, x.ndim))
        minv = x.amin(dim=dims, keepdim=True)
        maxv = x.amax(dim=dims, keepdim=True)
        normalized = (x - minv) / (maxv - minv + eps)
        return (normalized, minv, maxv) if return_stats else normalized
    else:
        x = np.array(x, dtype=np.float32)
        if dims is None:
            if x.ndim == 5:
                dims = (1, 2, 3, 4)
            elif x.ndim == 4:
                dims = (1, 2, 3)
            elif x.ndim == 3:
                dims = (1, 2)
            else:
                dims = tuple(range(1, x.ndim))
        minv = x.min(axis=dims, keepdims=True)
        maxv = x.max(axis=dims, keepdims=True)
        normalized = (x - minv) / (maxv - minv + eps)
        return (normalized, minv, maxv) if return_stats else normalized