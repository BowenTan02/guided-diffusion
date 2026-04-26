"""
Convert a pretrained 3-channel guided-diffusion UNet checkpoint into a
1-channel checkpoint suitable for fine-tuning on grayscale images.

What it does:
  - input_blocks.0.0.weight: average across the 3 input channels -> 1 channel.
  - out.2.weight / out.2.bias: collapse RGB triplets to a single channel.
    For learn_sigma=True, the original output has 6 channels split as
    [eps_R, eps_G, eps_B, var_R, var_G, var_B]; we average each triplet
    to produce 2 output channels [eps, var].
  - Drops `label_emb.weight` so the result loads cleanly into an
    unconditional model (class_cond=False).

Usage:
    python scripts/convert_checkpoint_to_1ch.py \
        --src models/256x256_diffusion.pt \
        --dst models/256x256_diffusion_1ch.pt \
        --learn_sigma True \
        --drop_label_emb True
"""

import argparse
import torch


def convert(src, dst, learn_sigma, drop_label_emb):
    sd = torch.load(src, map_location="cpu")

    # --- input conv ---
    k = "input_blocks.0.0.weight"
    w = sd[k]  # [C, 3, 3, 3]
    assert w.shape[1] == 3, f"expected 3 input channels, got {tuple(w.shape)}"
    sd[k] = w.mean(dim=1, keepdim=True)  # [C, 1, 3, 3]
    print(f"{k}: {tuple(w.shape)} -> {tuple(sd[k].shape)}")

    # --- output conv weight ---
    k = "out.2.weight"
    w = sd[k]  # [out, C, 3, 3] where out = 6 (learn_sigma) or 3
    if learn_sigma:
        assert w.shape[0] == 6, f"expected 6 out channels, got {tuple(w.shape)}"
        # split into [eps_R, eps_G, eps_B] and [var_R, var_G, var_B], avg each.
        eps = w[:3].mean(dim=0, keepdim=True)
        var = w[3:].mean(dim=0, keepdim=True)
        sd[k] = torch.cat([eps, var], dim=0)  # [2, C, 3, 3]
    else:
        assert w.shape[0] == 3
        sd[k] = w.mean(dim=0, keepdim=True)  # [1, C, 3, 3]
    print(f"out.2.weight: {tuple(w.shape)} -> {tuple(sd[k].shape)}")

    # --- output conv bias ---
    k = "out.2.bias"
    b = sd[k]
    if learn_sigma:
        sd[k] = torch.stack([b[:3].mean(), b[3:].mean()])
    else:
        sd[k] = b.mean(keepdim=True)
    print(f"out.2.bias: {tuple(b.shape)} -> {tuple(sd[k].shape)}")

    # --- drop class embedding for unconditional fine-tuning ---
    if drop_label_emb:
        for k in list(sd.keys()):
            if k.startswith("label_emb"):
                print(f"dropping {k} {tuple(sd[k].shape)}")
                del sd[k]

    torch.save(sd, dst)
    print(f"wrote {dst}")


def str2bool(v):
    return str(v).lower() in ("yes", "true", "t", "y", "1")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    p.add_argument("--learn_sigma", type=str2bool, default=True)
    p.add_argument("--drop_label_emb", type=str2bool, default=True)
    args = p.parse_args()
    convert(args.src, args.dst, args.learn_sigma, args.drop_label_emb)
