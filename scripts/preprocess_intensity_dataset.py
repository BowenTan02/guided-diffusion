#!/usr/bin/env python3
"""
Preprocess RGB frames -> normalized log-flux 256x256 .npy images for
unconditional 2D diffusion fine-tuning, matching the per-video pipeline
used in generate_flux_dataset.py so the spatial prior shares the same
intensity domain as the 1D temporal prior.

For each video subfolder:
  Pass A  (full streaming):
      I_mean = mean of grayscale(linear_RGB(sRGB(frame))) over ALL frames/pixels
      a      = target_ppp / I_mean                       (per-video scaling)
  Pass B  (full streaming):
      flux   = a * gray + d                              (d = 7.74e-4)
      track per-video flux_min, flux_max
  Pass C  (chosen frames only):
      flux_scaled = (flux - flux_min) / (flux_max - flux_min) * 10000
      log_flux    = log(flux_scaled + 1e-6)
      random 256x256 crop  ->  cache to a temp .npy

After all videos:
  Either reuse the 1D dataset's global log_flux_min/max (--stats_json from
  log_flux_dataset_stats.json) or compute them from this dataset.
  Normalize each cached crop to [-1, 1] and write final .npy + stats.json.

Why streaming the full video (not 32-frame probes):
  The 1D temporal prior was trained with I_mean computed over the full
  (interpolated) video. Estimating 'a' from 32 frames produces a different
  scaling per video, which shifts the log-flux distribution by an unknown
  per-video offset and breaks the shared normalization that joint
  spatial+temporal guidance assumes.

Usage:
  python scripts/preprocess_intensity_dataset.py \
      --videos_dir /path/to/parent_with_video_subfolders \
      --output_dir data/my_intensity \
      --dataset_size 20000 \
      --crop_size 256 \
      --target_ppp 1.0 \
      --stats_json /path/to/log_flux_dataset_stats.json \
      --seed 0
"""
import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


D_SPURIOUS = 7.74e-4
EPS = 1e-6
FLUX_SCALE_MAX = 10000.0
IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def srgb_to_linear(srgb):
    return np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)


def rgb_to_gray_linear(rgb_uint8):
    srgb = rgb_uint8.astype(np.float32) / 255.0
    lin = srgb_to_linear(srgb)
    return (0.2989 * lin[..., 0] + 0.5870 * lin[..., 1] + 0.1140 * lin[..., 2]).astype(np.float32)


def list_videos(parent):
    return sorted([p for p in parent.iterdir() if p.is_dir()])


def list_frames(video_dir):
    files = [p for p in video_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
    return sorted(files, key=lambda p: p.name)


def allocate_per_video(n_total, available):
    """Even split capped by each video's frame budget; remainder spread randomly."""
    n_videos = len(available)
    base = [min(n_total // n_videos, available[i]) for i in range(n_videos)]
    remaining = n_total - sum(base)
    order = list(range(n_videos))
    random.shuffle(order)
    i = 0
    while remaining > 0:
        room = [j for j in order if base[j] < available[j]]
        if not room:
            break
        j = room[i % len(room)]
        base[j] += 1
        remaining -= 1
        i += 1
    return base


def stream_I_mean(frames):
    """Exact mean of grayscale-linear over every pixel of every frame."""
    total = 0.0
    count = 0
    for fp in frames:
        img = np.array(Image.open(fp).convert("RGB"))
        gray = rgb_to_gray_linear(img)
        total += float(gray.sum(dtype=np.float64))
        count += gray.size
    return total / count if count > 0 else 0.0


def stream_flux_minmax(frames, a):
    fmin = math.inf
    fmax = -math.inf
    for fp in frames:
        img = np.array(Image.open(fp).convert("RGB"))
        gray = rgb_to_gray_linear(img)
        flux = a * gray + D_SPURIOUS
        fmin = min(fmin, float(flux.min()))
        fmax = max(fmax, float(flux.max()))
    return fmin, fmax


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos_dir", required=True, type=str)
    ap.add_argument("--output_dir", required=True, type=str)
    ap.add_argument("--dataset_size", type=int, required=True)
    ap.add_argument("--crop_size", type=int, default=256)
    ap.add_argument("--target_ppp", type=float, default=1.0)
    ap.add_argument("--stats_json", type=str, default="",
                    help="Path to log_flux_dataset_stats.json from the 1D temporal "
                         "dataset. If provided, uses its log_flux_min/max as the "
                         "global normalization range (recommended for joint "
                         "spatial+temporal priors). Otherwise computes them from "
                         "the crops produced here.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np_rng = np.random.default_rng(args.seed)

    videos_dir = Path(args.videos_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "_raw_logflux"
    raw_dir.mkdir(exist_ok=True)

    videos = list_videos(videos_dir)
    if not videos:
        raise SystemExit(f"No video subfolders found under {videos_dir}")

    frames_per_video = [list_frames(v) for v in videos]
    avail = [len(f) for f in frames_per_video]
    total_avail = sum(avail)
    target = min(args.dataset_size, total_avail)
    if total_avail < args.dataset_size:
        print(f"WARNING: only {total_avail} frames across {len(videos)} videos; "
              f"requested {args.dataset_size}.")

    per_video = allocate_per_video(target, avail)
    print(f"Found {len(videos)} videos, {total_avail} frames total. "
          f"Sampling {target} crops; per-video budget min/max = "
          f"{min(per_video)}/{max(per_video)}.")

    # ---- Per-video pipeline + crop caching ----
    g_min = math.inf
    g_max = -math.inf
    written = 0

    for vi, (vdir, frames, k) in enumerate(zip(videos, frames_per_video, per_video)):
        if k == 0 or len(frames) == 0:
            continue

        # Pass A: exact I_mean over all frames/pixels of this video.
        I_mean = stream_I_mean(tqdm(frames, desc=f"[{vi+1}/{len(videos)}] {vdir.name} A:I_mean", leave=False))
        a = args.target_ppp / I_mean if I_mean > 0 else 1.0

        # Pass B: exact flux min/max across the video.
        f_min, f_max = stream_flux_minmax(
            tqdm(frames, desc=f"[{vi+1}/{len(videos)}] {vdir.name} B:flux_minmax", leave=False), a
        )
        f_range = f_max - f_min
        if f_range <= 0:
            print(f"  {vdir.name}: degenerate flux range; skipping.")
            continue

        # Choose k frames uniformly from this video.
        replace = k > len(frames)
        chosen = sorted(np_rng.choice(len(frames), size=k, replace=replace).tolist())

        # Pass C: load only chosen frames, apply pipeline, random-crop, cache.
        for fi in tqdm(chosen, desc=f"[{vi+1}/{len(videos)}] {vdir.name} C:crops", leave=False):
            img = np.array(Image.open(frames[fi]).convert("RGB"))
            H, W = img.shape[:2]
            cs = args.crop_size
            if H < cs or W < cs:
                continue
            cy = random.randint(0, H - cs)
            cx = random.randint(0, W - cs)
            crop = img[cy:cy + cs, cx:cx + cs]

            gray = rgb_to_gray_linear(crop)
            flux = a * gray + D_SPURIOUS
            flux_scaled = (flux - f_min) / f_range * FLUX_SCALE_MAX
            log_flux = np.log(flux_scaled + EPS).astype(np.float32)

            g_min = min(g_min, float(log_flux.min()))
            g_max = max(g_max, float(log_flux.max()))

            np.save(raw_dir / f"v{vi:04d}_f{int(fi):06d}_{written:08d}.npy", log_flux)
            written += 1

    if written == 0:
        raise SystemExit("No crops were produced; check inputs / crop_size.")

    # ---- Global normalization ----
    if args.stats_json:
        with open(args.stats_json) as f:
            stats_in = json.load(f)
        norm_min = float(stats_in["log_flux_min"])
        norm_max = float(stats_in["log_flux_max"])
        source = f"stats_json={args.stats_json}"
        # Sanity check: warn if our crops fall well outside the temporal dataset's range.
        if g_min < norm_min - 0.5 or g_max > norm_max + 0.5:
            print(f"WARNING: spatial log-flux range [{g_min:.3f}, {g_max:.3f}] "
                  f"extends beyond temporal range [{norm_min:.3f}, {norm_max:.3f}]. "
                  f"Values will exceed [-1, 1] after normalization (no clipping).")
    else:
        norm_min, norm_max = g_min, g_max
        source = "computed from this dataset"

    norm_range = norm_max - norm_min
    if norm_range <= 0:
        raise SystemExit(f"Degenerate normalization range: [{norm_min}, {norm_max}]")

    print(f"Spatial log-flux range observed: [{g_min:.4f}, {g_max:.4f}]")
    print(f"Normalizing with [{norm_min:.4f}, {norm_max:.4f}] -> [-1, 1]  ({source})")

    # ---- Pass: normalize and emit final .npy ----
    raw_files = sorted(raw_dir.iterdir())
    for p in tqdm(raw_files, desc="Normalize"):
        x = np.load(p)
        x = (2.0 * (x - norm_min) / norm_range - 1.0).astype(np.float32)
        np.save(out_dir / p.name, x)
        p.unlink()
    raw_dir.rmdir()

    stats = {
        "log_flux_min": norm_min,
        "log_flux_max": norm_max,
        "log_flux_min_observed": g_min,
        "log_flux_max_observed": g_max,
        "n_samples": written,
        "image_size": args.crop_size,
        "target_ppp": args.target_ppp,
        "d_spurious": D_SPURIOUS,
        "flux_scale_max": FLUX_SCALE_MAX,
        "normalization": "global_minmax_to_[-1,1]",
        "normalization_source": source,
        "denorm": "log_flux_scaled = (x + 1) / 2 * (max - min) + min;  "
                  "flux_scaled = exp(log_flux_scaled) - 1e-6",
    }
    with open(out_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Wrote {written} normalized .npy crops + stats.json to {out_dir}")


if __name__ == "__main__":
    main()
