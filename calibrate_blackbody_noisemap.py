#!/usr/bin/env python3
"""
Run this ONCE on your 800-frame static blackbody sequence to compute and
cache the fixed-pattern noise map. Reuse the saved .npy across all test/
inference runs -- don't recompute it every time.

Usage:
  python calibrate_blackbody_noisemap.py --blackbody_dir /path/to/blackbody_frames --save_path calib/blackbody_noise_map.npy
"""
import argparse
from utils_thermal import compute_blackbody_noise_map

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--blackbody_dir", type=str, required=True,
                        help="dir of raw .tif frames of a static blackbody")
    parser.add_argument("--save_path", type=str, default="calib/blackbody_noise_map_2.npy")
    parser.add_argument("--max_num_fr", type=int, default=800)
    parser.add_argument("--no_abs", action='store_true',
                        help="keep the signed FPN pattern instead of taking abs()")
    parser.add_argument("--no_normalize", action='store_true',
                    help="skip the global [0,1] min-max normalization")
    args = parser.parse_args()

    import os
    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)

    compute_blackbody_noise_map(
      blackbody_dir=args.blackbody_dir,
      max_num_fr=args.max_num_fr,
      use_abs=not args.no_abs,
      normalize=not args.no_normalize,
      save_path=args.save_path,
    )