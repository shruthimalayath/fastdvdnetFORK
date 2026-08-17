#!/usr/bin/env python3
"""
Diagnostic: how much does raw min/max actually jitter frame-to-frame in your
thermal sequences?

This directly tests the assumption behind "min/max shouldn't change much
across a 5-frame window at high frame rate" -- by measuring it on real data,
rather than reasoning about it. Since min/max are order statistics (driven
by single extreme pixels, not overall scene content), they can jitter even
when the scene itself is stable.

For each sequence it computes, per frame:
  - per-frame min/max (what validate_and_log's minmax_normalize_pair
    actually uses, since it treats each frame as an independent batch
    element when given a [F,C,H,W] tensor)
  - windowed (joint, size=--window_size) min/max centered on that frame
    (what training's normalize_augment_pair uses -- one min/max shared
    across the whole temporal patch)

And reports frame-to-frame drift (delta) for both, so you can see directly
whether per-frame min/max is stable or jittery on your actual sensor/scenes,
and how much smoother the windowed version is by comparison.

Usage:
  python diagnostic_minmax_drift.py --data_dir /path/to/thermal_val/thermal_val_noisy --save_path results/minmax_drift
"""
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils_thermal import load_raw_sequence


def compute_per_frame_minmax(seq_raw):
    """seq_raw: [F,C,H,W] raw float32. Returns per-frame min, max arrays [F]."""
    F = seq_raw.shape[0]
    mins = np.array([seq_raw[t].min() for t in range(F)])
    maxs = np.array([seq_raw[t].max() for t in range(F)])
    return mins, maxs


def compute_windowed_minmax(seq_raw, window_size):
    """Joint min/max over a centered sliding window, per frame index.
    Mirrors training's normalize_augment_pair scope (joint across the
    whole temporal patch), using simple edge-clamped indexing.
    """
    F = seq_raw.shape[0]
    radius = (window_size - 1) // 2
    mins = np.zeros(F)
    maxs = np.zeros(F)
    for t in range(F):
        lo = max(0, t - radius)
        hi = min(F, t + radius + 1)
        window = seq_raw[lo:hi]
        mins[t] = window.min()
        maxs[t] = window.max()
    return mins, maxs


def summarize_drift(name, mins, maxs, log_lines):
    if len(mins) < 2:
        return
    d_min = np.diff(mins)
    d_max = np.diff(maxs)
    range_vals = maxs - mins
    line = (f"  [{name}] "
            f"min drift: mean|Δ|={np.mean(np.abs(d_min)):.2f} max|Δ|={np.max(np.abs(d_min)):.2f} | "
            f"max drift: mean|Δ|={np.mean(np.abs(d_max)):.2f} max|Δ|={np.max(np.abs(d_max)):.2f} | "
            f"typical range (max-min): {np.mean(range_vals):.1f}")
    print(line)
    log_lines.append(line)

    # express drift as a % of the typical range, since raw-unit drift alone
    # doesn't tell you how much it actually matters for normalization
    pct_min = 100 * np.mean(np.abs(d_min)) / (np.mean(range_vals) + 1e-6)
    pct_max = 100 * np.mean(np.abs(d_max)) / (np.mean(range_vals) + 1e-6)
    line2 = (f"       -> as % of typical range: min drift {pct_min:.2f}%, max drift {pct_max:.2f}%")
    print(line2)
    log_lines.append(line2)


def plot_sequence(seq_name, frame_mins, frame_maxs, win_mins, win_maxs, save_path):
    F = len(frame_mins)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(frame_mins, label='per-frame min', color='tab:blue')
    axes[0].plot(frame_maxs, label='per-frame max', color='tab:red')
    axes[0].set_title(f'{seq_name}: per-frame min/max (what validate_and_log uses)')
    axes[0].set_ylabel('raw pixel value')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(win_mins, label='windowed min', color='tab:blue', linestyle='--')
    axes[1].plot(win_maxs, label='windowed max', color='tab:red', linestyle='--')
    axes[1].set_title(f'{seq_name}: windowed (joint 5-frame) min/max (what training uses)')
    axes[1].set_xlabel('frame index')
    axes[1].set_ylabel('raw pixel value')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join(save_path, f'{seq_name}_minmax_drift.png')
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def main(args):
    os.makedirs(args.save_path, exist_ok=True)
    log_lines = []

    seq_names = sorted([d for d in os.listdir(args.data_dir)
                        if os.path.isdir(os.path.join(args.data_dir, d))])
    if args.max_sequences is not None:
        seq_names = seq_names[:args.max_sequences]

    if len(seq_names) == 0:
        raise RuntimeError(f"No sequence subfolders found in {args.data_dir}")

    print(f"Found {len(seq_names)} sequences, checking up to {args.max_sequences or 'all'}\n")

    all_frame_min_drift = []
    all_frame_max_drift = []
    all_win_min_drift = []
    all_win_max_drift = []

    for seq_name in seq_names:
        seq_dir = os.path.join(args.data_dir, seq_name)
        seq_raw = load_raw_sequence(seq_dir, max_num_fr=args.max_num_fr_per_seq)
        if seq_raw is None:
            print(f"  [{seq_name}] SKIPPED: no .tif frames found")
            continue

        frame_mins, frame_maxs = compute_per_frame_minmax(seq_raw)
        win_mins, win_maxs = compute_windowed_minmax(seq_raw, args.window_size)

        print(f"[{seq_name}] {seq_raw.shape[0]} frames")
        summarize_drift("per-frame", frame_mins, frame_maxs, log_lines)
        summarize_drift(f"windowed(w={args.window_size})", win_mins, win_maxs, log_lines)
        print()

        if len(frame_mins) > 1:
            all_frame_min_drift.extend(np.abs(np.diff(frame_mins)).tolist())
            all_frame_max_drift.extend(np.abs(np.diff(frame_maxs)).tolist())
            all_win_min_drift.extend(np.abs(np.diff(win_mins)).tolist())
            all_win_max_drift.extend(np.abs(np.diff(win_maxs)).tolist())

        if args.plot:
            out_path = plot_sequence(seq_name, frame_mins, frame_maxs, win_mins, win_maxs, args.save_path)
            print(f"  saved plot: {out_path}\n")

    if all_frame_min_drift:
        print("=== Overall summary across all sequences ===")
        print(f"  Per-frame min drift:  mean|Δ|={np.mean(all_frame_min_drift):.2f}, "
              f"p95={np.percentile(all_frame_min_drift, 95):.2f}, max={np.max(all_frame_min_drift):.2f}")
        print(f"  Per-frame max drift:  mean|Δ|={np.mean(all_frame_max_drift):.2f}, "
              f"p95={np.percentile(all_frame_max_drift, 95):.2f}, max={np.max(all_frame_max_drift):.2f}")
        print(f"  Windowed min drift:   mean|Δ|={np.mean(all_win_min_drift):.2f}, "
              f"p95={np.percentile(all_win_min_drift, 95):.2f}, max={np.max(all_win_min_drift):.2f}")
        print(f"  Windowed max drift:   mean|Δ|={np.mean(all_win_max_drift):.2f}, "
              f"p95={np.percentile(all_win_max_drift, 95):.2f}, max={np.max(all_win_max_drift):.2f}")
        print("\nInterpretation guide:")
        print("  - If per-frame drift is small relative to typical range (see per-sequence")
        print("    '% of typical range' lines above), your intuition holds and PSNR_val's")
        print("    per-frame normalization is a close approximation of per-clip -- safe to")
        print("    trust more than we assumed.")
        print("  - If per-frame drift is large or has high outliers (check p95 vs mean --")
        print("    a big gap means a few frames/pixels are driving most of the jitter),")
        print("    that's likely hot/dead pixels or sensor recalibration events, and the")
        print("    train/val normalization mismatch is a real concern worth fixing.")

    with open(os.path.join(args.save_path, 'summary.log'), 'w') as f:
        f.write('\n'.join(log_lines))

    print(f"\nDone. Plots and summary.log saved to {args.save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check min/max drift across thermal sequences")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="parent dir containing sequence subfolders of .tif frames")
    parser.add_argument("--save_path", type=str, default="results/minmax_drift")
    parser.add_argument("--window_size", type=int, default=5,
                        help="temporal window size to compare against (match --temp_patch_size)")
    parser.add_argument("--max_num_fr_per_seq", type=int, default=200)
    parser.add_argument("--max_sequences", type=int, default=None,
                        help="limit number of sequences checked, for a quick look")
    parser.add_argument("--plot", action='store_true', default=True,
                        help="save a min/max-over-time plot per sequence (default: on)")
    parser.add_argument("--no_plot", dest='plot', action='store_false')

    args = parser.parse_args()
    main(args)