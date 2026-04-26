#!/usr/bin/env python3
"""
Preprocess RGB frames -> normalized log-flux 256x256 .npy images for
unconditional 2D diffusion fine-tuning, matching the per-video pipeline
used in generate_flux_dataset.py so the spatial prior shares the same
intensity domain as the 1D temporal prior.

NO temporal interpolation is performed. Each input frame is treated as-is.

Per-video math (single streaming pass):
    gray   = 0.2989*lin(R) + 0.5870*lin(G) + 0.1140*lin(B)
             with lin = sRGB-piecewise-gamma applied via a 256-entry LUT.
    a      = target_ppp / mean(gray over all frames/pixels)
    flux   = a * gray + d                 (d = 7.74e-4)
    Because flux is monotonic in gray:
        flux_min = a * gray_min + d
        flux_max = a * gray_max + d
    flux_scaled = (flux - flux_min) / (flux_max - flux_min) * 10000
    log_flux    = log(flux_scaled + 1e-6)

After all videos, normalize globally to [-1, 1]. If --stats_json is given,
reuse the 1D temporal dataset's log_flux_min/max so the two priors share
the exact same intensity domain.

Usage:
python generate_2D_dataset.py --videos_dir /u/scratch1/tan583/projects/phantom_simulation/separated_videos/ --output_dir /u/scratch1/tan583/guided-diffusion/data/ --dataset_size 30000 --crop_size 256 --target_ppp 1.0 --seed 0
"""
import argparse
import json
import math
import os
import random
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


D_SPURIOUS = 7.74e-4
EPS = 1e-6
FLUX_SCALE_MAX = 10000.0
IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


# --- sRGB -> linear LUT, one for each channel weighted by grayscale coeff ---
# After this, gray = LUT_R[r] + LUT_G[g] + LUT_B[b] for uint8 inputs.
def _build_lut():
    x = np.arange(256, dtype=np.float64) / 255.0
    lin = np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    return (
        (0.2989 * lin).astype(np.float32),
        (0.5870 * lin).astype(np.float32),
        (0.1140 * lin).astype(np.float32),
    )


_LUT_R, _LUT_G, _LUT_B = _build_lut()


def gray_from_bgr(bgr_uint8):
    """cv2 returns BGR; index per-channel LUTs accordingly."""
    return _LUT_B[bgr_uint8[..., 0]] + _LUT_G[bgr_uint8[..., 1]] + _LUT_R[bgr_uint8[..., 2]]


def list_videos(parent):
    return sorted([p for p in parent.iterdir() if p.is_dir()])


def list_frames(video_dir):
    files = [p for p in video_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
    return sorted(files, key=lambda p: p.name)


def allocate_per_video(n_total, available):
    n_videos = len(available)
    base = [min(n_total // n_videos, available[i]) for i in range(n_videos)]
    remaining = n_total - sum(base)
    order = list(range(n_videos))
    random.shuffle(order)
    while remaining > 0:
        room = [j for j in order if base[j] < available[j]]
        if not room:
            break
        for j in room:
            if remaining == 0:
                break
            base[j] += 1
            remaining -= 1
    return base


def process_video(task):
    """
    Single pass over a video: stream all frames once, accumulate gray statistics,
    cache the chosen frames' gray crops (full crop already taken), then at the
    end apply per-video scaling + log to the cached crops.

    Returns: list of (filename, log_flux_crop float32) plus per-video info.
    """
    (video_idx, video_dir, frames, k, crop_size, target_ppp, seed, per_video_rescale) = task

    rng = random.Random(seed + video_idx)
    np_rng = np.random.default_rng(seed + video_idx)

    if k == 0 or len(frames) == 0:
        return [], None

    if k > len(frames):
        chosen = sorted(np_rng.choice(len(frames), size=k, replace=True).tolist())
    else:
        chosen = sorted(np_rng.choice(len(frames), size=k, replace=False).tolist())
    chosen_set = set(chosen)
    # For replacement: precompute which crop offsets to use for each occurrence.
    # Multiplicity per frame index -> we need a separate (cy, cx) per occurrence.
    occurrences = {}
    for fi in chosen:
        occurrences.setdefault(fi, 0)
        occurrences[fi] += 1

    # Streaming accumulators
    g_sum = 0.0
    g_count = 0
    g_min = math.inf
    g_max = -math.inf

    cached = []  # list of (frame_idx, occurrence_idx, gray_crop float32)

    for fi, fp in enumerate(frames):
        bgr = cv2.imread(str(fp), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        gray = gray_from_bgr(bgr)  # float32 [H, W]

        g_sum += float(gray.sum(dtype=np.float64))
        g_count += gray.size
        gmin_f = float(gray.min())
        gmax_f = float(gray.max())
        if gmin_f < g_min:
            g_min = gmin_f
        if gmax_f > g_max:
            g_max = gmax_f

        if fi in chosen_set:
            H, W = gray.shape
            if H < crop_size or W < crop_size:
                continue
            for occ in range(occurrences[fi]):
                cy = rng.randint(0, H - crop_size)
                cx = rng.randint(0, W - crop_size)
                cached.append(
                    (fi, occ, gray[cy:cy + crop_size, cx:cx + crop_size].copy())
                )

    if g_count == 0:
        return [], None

    I_mean = g_sum / g_count
    a = target_ppp / I_mean if I_mean > 0 else 1.0
    f_min = a * g_min + D_SPURIOUS
    f_max = a * g_max + D_SPURIOUS
    f_range = f_max - f_min
    if f_range <= 0:
        return [], None

    out = []
    local_min = math.inf
    local_max = -math.inf
    for fi, occ, gray_crop in cached:
        flux = a * gray_crop + D_SPURIOUS
        if per_video_rescale:
            flux_for_log = np.maximum((flux - f_min) / f_range * FLUX_SCALE_MAX, 0.0)
        else:
            flux_for_log = flux
        log_flux = np.log(flux_for_log + EPS).astype(np.float32)
        local_min = min(local_min, float(log_flux.min()))
        local_max = max(local_max, float(log_flux.max()))
        name = f"v{video_idx:04d}_f{fi:06d}_o{occ}.npy"
        out.append((name, log_flux))

    return out, (local_min, local_max)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos_dir", required=True, type=str)
    ap.add_argument("--output_dir", required=True, type=str)
    ap.add_argument("--dataset_size", type=int, required=True)
    ap.add_argument("--crop_size", type=int, default=256)
    ap.add_argument("--target_ppp", type=float, default=1.0)
    ap.add_argument("--stats_json", type=str, default="",
                    help="Only used when --norm_mode=shared. Path to the 1D "
                         "temporal dataset's log_flux_dataset_stats.json.")
    ap.add_argument("--norm_mode", type=str, default="dataset_global",
                    choices=["dataset_global", "per_image", "shared"],
                    help="dataset_global (RECOMMENDED): percentile-based global "
                         "normalization of log-flux on this dataset using "
                         "(--norm_low_pct, --norm_high_pct), clipped to [-1,1]. "
                         "Preserves cross-crop brightness, robust to outliers, "
                         "well-spread distribution. "
                         "per_image: each crop normalized to its own min/max "
                         "(throws away absolute brightness; per-crop min/max "
                         "saved in per_image_stats.npz). "
                         "shared: reuse --stats_json from the 1D temporal dataset.")
    ap.add_argument("--num_workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per_video_rescale", action="store_true",
                    help="Apply the per-video [0,10000] flux min-max rescale "
                         "(matches the 1D temporal pipeline). For 2D crops this "
                         "biases the log distribution toward +1 and is NOT "
                         "recommended. Default: off.")
    ap.add_argument("--norm_low_pct", type=float, default=0.5,
                    help="Lower percentile of global log_flux used as the "
                         "normalization minimum (default 0.5). Robust to "
                         "outlier dark pixels.")
    ap.add_argument("--norm_high_pct", type=float, default=99.5,
                    help="Upper percentile of global log_flux used as the "
                         "normalization maximum (default 99.5).")
    args = ap.parse_args()

    random.seed(args.seed)
    np_rng = np.random.default_rng(args.seed)

    def _resolve(raw, label):
        # Strip whitespace, expand ~ and $VARS, then resolve to absolute path.
        s = (raw or "").strip()
        s = os.path.expandvars(os.path.expanduser(s))
        p = Path(s).resolve()
        return s, p

    def _diagnose_missing(label, raw, resolved):
        # Show exactly what we tried and why it might not match.
        print(f"\n[{label}] could not be opened.")
        print(f"  raw arg (repr):     {raw!r}")
        print(f"  resolved absolute:  {resolved}")
        print(f"  os.path.exists:     {os.path.exists(resolved)}")
        parent = resolved.parent
        print(f"  parent exists:      {parent.exists()}  ({parent})")
        if parent.exists():
            try:
                kids = sorted(os.listdir(parent))
                print(f"  parent contains ({len(kids)} entries):")
                for k in kids[:50]:
                    print(f"    {k!r}")
                if len(kids) > 50:
                    print(f"    ... ({len(kids) - 50} more)")
            except PermissionError as e:
                print(f"  parent listing failed: {e}")
        print("  cwd:                ", os.getcwd())
        print("  hint: trailing/leading whitespace, smart quotes from copy-paste, "
              "or unexpanded ~/$VAR in the argument all produce this error.\n")

    videos_raw, videos_dir = _resolve(args.videos_dir, "--videos_dir")
    out_raw, out_dir = _resolve(args.output_dir, "--output_dir")

    if not videos_dir.is_dir():
        _diagnose_missing("--videos_dir", videos_raw, videos_dir)
        raise SystemExit("--videos_dir is not a directory.")
    # --stats_json is OPTIONAL. If it's missing, unreadable, or malformed we
    # WARN and fall back to computing the normalization range from this dataset.
    # We never let stats_json failures block the production of normalized crops.
    if args.stats_json:
        stats_raw, sp = _resolve(args.stats_json, "--stats_json")
        ok = False
        if not sp.is_file():
            print(f"\nWARNING: --stats_json not found at {sp}.")
            print(f"  raw arg (repr): {stats_raw!r}")
            print(f"  Falling back to computing min/max from this dataset.\n")
        else:
            try:
                with open(sp) as f:
                    _probe = json.load(f)
                float(_probe["log_flux_min"]); float(_probe["log_flux_max"])
                args.stats_json = str(sp)
                ok = True
            except Exception as e:
                print(f"\nWARNING: --stats_json at {sp} is unreadable or missing "
                      f"keys ({e}). Falling back to computing min/max from this dataset.\n")
        if not ok:
            args.stats_json = ""

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
          f"{min(per_video)}/{max(per_video)}. Workers: {args.num_workers}.")

    tasks = [
        (vi, videos[vi], frames_per_video[vi], per_video[vi],
         args.crop_size, args.target_ppp, args.seed, args.per_video_rescale)
        for vi in range(len(videos))
    ]

    g_min = math.inf
    g_max = -math.inf
    written = 0

    if args.num_workers <= 1:
        it = (process_video(t) for t in tasks)
    else:
        pool = Pool(args.num_workers)
        it = pool.imap_unordered(process_video, tasks, chunksize=1)

    for crops, rng_minmax in tqdm(it, total=len(tasks), desc="Videos"):
        if rng_minmax is not None:
            g_min = min(g_min, rng_minmax[0])
            g_max = max(g_max, rng_minmax[1])
        for name, arr in crops:
            np.save(raw_dir / name, arr)
            written += 1

    if args.num_workers > 1:
        pool.close()
        pool.join()

    if written == 0:
        raise SystemExit("No crops were produced; check inputs / crop_size.")

    raw_files = sorted(raw_dir.iterdir())

    if args.norm_mode == "per_image":
        # Each crop normalized to [-1, 1] by its own min/max. This gives the
        # diffusion model data with proper dynamic range. Per-crop (min, max)
        # is recorded so absolute log-flux can be recovered at inference time.
        norm_min, norm_max, source = None, None, "per_image"
        per_image_keys = []
        per_image_min = []
        per_image_max = []
    elif args.norm_mode == "shared":
        with open(args.stats_json) as f:
            stats_in = json.load(f)
        norm_min = float(stats_in["log_flux_min"])
        norm_max = float(stats_in["log_flux_max"])
        source = f"stats_json={args.stats_json}"
    else:  # dataset_global
        sample_pool = []
        target_pool = 5_000_000
        per_file = max(1, target_pool // max(1, len(raw_files)))
        rng_pct = np.random.default_rng(args.seed + 1)
        for p in raw_files:
            x = np.load(p).ravel()
            if x.size > per_file:
                idx = rng_pct.integers(0, x.size, size=per_file)
                sample_pool.append(x[idx])
            else:
                sample_pool.append(x)
        sample = np.concatenate(sample_pool)
        norm_min = float(np.percentile(sample, args.norm_low_pct))
        norm_max = float(np.percentile(sample, args.norm_high_pct))
        print(f"Percentile normalization: p{args.norm_low_pct}={norm_min:.4f}, "
              f"p{args.norm_high_pct}={norm_max:.4f}  "
              f"(sample size {sample.size:,})")
        source = (f"percentile p{args.norm_low_pct}/p{args.norm_high_pct} "
                  f"on {sample.size:,} pixels")

    if args.norm_mode != "per_image":
        norm_range = norm_max - norm_min
        if norm_range <= 0:
            raise SystemExit(f"Degenerate normalization range: [{norm_min}, {norm_max}]")
        print(f"Spatial log-flux range observed: [{g_min:.4f}, {g_max:.4f}]")
        print(f"Normalizing with [{norm_min:.4f}, {norm_max:.4f}] -> [-1, 1]  ({source})")

    # Write a partial stats.json now so a partial run still leaves metadata.
    stats_pre = {
        "norm_mode": args.norm_mode,
        "log_flux_min": norm_min,
        "log_flux_max": norm_max,
        "log_flux_min_observed": g_min,
        "log_flux_max_observed": g_max,
        "image_size": args.crop_size,
        "target_ppp": args.target_ppp,
        "d_spurious": D_SPURIOUS,
        "flux_scale_max": FLUX_SCALE_MAX,
        "normalization_source": source,
        "denorm": (
            "per_image: per_image_stats.npz holds (min, max) for each filename; "
            "log_flux_scaled = (x + 1)/2 * (max - min) + min;  "
            "flux_scaled = exp(log_flux_scaled) - 1e-6"
            if args.norm_mode == "per_image"
            else "log_flux_scaled = (x + 1)/2 * (max - min) + min;  "
                 "flux_scaled = exp(log_flux_scaled) - 1e-6"
        ),
    }
    with open(out_dir / "stats.json", "w") as f:
        json.dump(stats_pre, f, indent=2)
    print(f"Pre-normalize stats.json written to {out_dir / 'stats.json'}")

    bulk_min = math.inf
    bulk_max = -math.inf
    bulk_sum = 0.0
    bulk_sqsum = 0.0
    bulk_n = 0
    for p in tqdm(raw_files, desc="Normalize"):
        x = np.load(p)
        if args.norm_mode == "per_image":
            xmin = float(x.min())
            xmax = float(x.max())
            xrange = xmax - xmin
            if xrange <= 0:
                # Degenerate constant crop: emit zeros and record the constant.
                y = np.zeros_like(x, dtype=np.float32)
            else:
                y = (2.0 * (x - xmin) / xrange - 1.0).astype(np.float32)
            per_image_keys.append(p.name)
            per_image_min.append(xmin)
            per_image_max.append(xmax)
            x = y
        else:
            x = (2.0 * (x - norm_min) / norm_range - 1.0).astype(np.float32)
            np.clip(x, -1.0, 1.0, out=x)
        np.save(out_dir / p.name, x)
        p.unlink()
        bulk_min = min(bulk_min, float(x.min()))
        bulk_max = max(bulk_max, float(x.max()))
        bulk_sum += float(x.sum(dtype=np.float64))
        bulk_sqsum += float((x.astype(np.float64) ** 2).sum())
        bulk_n += x.size
    raw_dir.rmdir()

    if args.norm_mode == "per_image":
        np.savez(
            out_dir / "per_image_stats.npz",
            filename=np.array(per_image_keys),
            log_flux_min=np.array(per_image_min, dtype=np.float64),
            log_flux_max=np.array(per_image_max, dtype=np.float64),
        )
        print(f"Wrote per_image_stats.npz with {len(per_image_keys)} entries.")

    bulk_mean = bulk_sum / bulk_n
    bulk_std = math.sqrt(max(0.0, bulk_sqsum / bulk_n - bulk_mean ** 2))
    print(f"Final normalized stats: min={bulk_min:.3f}, max={bulk_max:.3f}, "
          f"mean={bulk_mean:.3f}, std={bulk_std:.3f}")

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
