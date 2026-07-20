'''
Paired DALI video loader

Loads:
    noisy video sequences
    clean video sequences

Applies:
    same temporal window (assuming matching file order and no shuffle)
    same spatial crop

Returns:
    noisy : [N, F, C, H, W]
    clean : [N, F, C, H, W]
'''
# When first testing check
    # does VideoReader keep 16 bit precision?

import os

from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin import pytorch
import nvidia.dali.ops as ops
import nvidia.dali.types as types


class PairedVideoReaderPipeline(Pipeline):
    ''' Pipeline for reading H264 videos based on NVIDIA DALI.
	Returns a batch of sequences of `sequence_length` frames of shape [N, F, C, H, W]
	(N being the batch size and F the number of frames). Frames are RGB uint8.
	Args:
		batch_size: (int)
				Size of the batches
		sequence_length: (int)
				Frames to load per sequence.
		num_threads: (int)
				Number of threads.
		device_id: (int)
				GPU device ID where to load the sequences.
		files: (str or list of str)
				File names of the video files to load.
		crop_size: (int)
				Size of the crops. The crops are in the same location in all frames in the sequence
		random_shuffle: (bool, optional, default=True)
				Whether to randomly shuffle data.
		step: (int, optional, default=-1)
				Frame interval between each sequence (if `step` < 0, `step` is set to `sequence_length`).
	'''
    def __init__(self,batch_size, sequence_length, num_threads, device_id, noisy_files, clean_files, crop_size, step=-1,):

        super(PairedVideoReaderPipeline, self).__init__(batch_size, num_threads, device_id, seed=12)

        # Noisy reader
        self.noisy_reader = ops.VideoReader(
            device="gpu",
            filenames=noisy_files,
            sequence_length=sequence_length, 
            normalized=False,
            random_shuffle=False,
            image_type=types.DALIImageType.RGB, 
            dtype=types.DALIDataType.FLOAT, 
            step=step, 
            initial_fill=16
        )

        # Clean reader
        self.clean_reader = ops.VideoReader(
            device="gpu",
            filenames=clean_files,
            sequence_length=sequence_length,
            normalized=False,
            random_shuffle=False,
            image_type=types.DALIImageType.RGB,
            dtype=types.DALIDataType.FLOAT,
            step=step,
            initial_fill=16,
        )

        # Shared crop op
        self.crop = ops.CropMirrorNormalize(
            device="gpu",
            crop_w=crop_size,
            crop_h=crop_size,
            output_layout="FCHW",
            dtype=types.DALIDataType.FLOAT,
        )

        # random crop coordinates
        self.uniform = ops.Uniform(range=(0.0, 1.0))

    def define_graph(self):
        noisy_seq = self.noisy_reader(name="NoisyReader")
        clean_seq = self.clean_reader(name="CleanReader")

        # Generate crop ONCE
        crop_x = self.uniform()
        crop_y = self.uniform()

        # Apply SAME crop to noisy
        noisy_crop = self.crop(noisy_seq, crop_pos_x=crop_x, crop_pos_y=crop_y)

        # Apply SAME crop to clean
        clean_crop = self.crop(clean_seq,crop_pos_x=crop_x,crop_pos_y=crop_y,)
        return noisy_crop, clean_crop


class train_dali_loader():
    '''Sequence dataloader.
	Args:
		batch_size: (int)
			Size of the batches
		file_root: (str)
			Path to directory with video sequences
		sequence_length: (int)
			Frames to load per sequence
		crop_size: (int)
			Size of the crops. The crops are in the same location in all frames in the sequence
		epoch_size: (int, optional, default=-1)
			Size of the epoch. If epoch_size <= 0, epoch_size will default to the size of VideoReaderPipeline
		random_shuffle (bool, optional, default=True)
			Whether to randomly shuffle data.
		temp_stride: (int, optional, default=-1)
			Frame interval between each sequence
			(if `temp_stride` < 0, `temp_stride` is set to `sequence_length`).
	'''
    def __init__(
        self,
        batch_size,
        noisy_root,
        clean_root,
        sequence_length,
        crop_size,
        epoch_size=-1,
        temp_stride=-1,
    ):

    
        # Build file lists
        noisy_files = sorted(
            [
                os.path.join(noisy_root, f)
                for f in os.listdir(noisy_root)
            ]
        )

        clean_files = sorted(
            [
                os.path.join(clean_root, f)
                for f in os.listdir(clean_root)
            ]
        )

        if len(noisy_files) != len(clean_files):
            raise RuntimeError(
                "Error: Different number of noisy and clean videos"
            )

        self.pipeline = PairedVideoReaderPipeline(
            batch_size=batch_size,
            sequence_length=sequence_length,
            num_threads=2,
            device_id=0,
            noisy_files=noisy_files,
            clean_files=clean_files,
            crop_size=crop_size,
            step=temp_stride,
        )

        self.pipeline.build()

        if epoch_size <= 0:
            self.epoch_size = self.pipeline.epoch_size("NoisyReader")
        else:
            self.epoch_size = epoch_size

            
        self.dali_iterator = pytorch.DALIGenericIterator(
                    pipelines=self.pipeline,
                    output_map=["noisy", "clean"],
                    size=self.epoch_size,
                    auto_reset=True,
                )

    def __len__(self):
        return self.epoch_size

    def __iter__(self):
        return self.dali_iterator.__iter__()


