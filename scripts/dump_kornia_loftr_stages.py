#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def add_reference_path(ref_dir: Path) -> None:
    if str(ref_dir) not in sys.path:
        sys.path.insert(0, str(ref_dir))


def tensor_stats(tensor, sample_len: int = 8) -> dict[str, object]:
    import torch

    flat = tensor.detach().cpu().reshape(-1).float()
    sample = flat[:sample_len].tolist()
    return {
        "shape": list(tensor.shape),
        "mean": float(flat.mean().item()),
        "std": float(flat.std(unbiased=True).item()),
        "min": float(flat.min().item()),
        "max": float(flat.max().item()),
        "abs_mean": float(flat.abs().mean().item()),
        "l2_norm": float(torch.linalg.vector_norm(flat).item()),
        "sample": [float(v) for v in sample],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dir", default="/tmp/kornia-python-0.8.2")
    parser.add_argument("--output", required=True)
    parser.add_argument("--resize-width", type=int, default=960)
    parser.add_argument("--resize-height", type=int, default=540)
    parser.add_argument("left")
    parser.add_argument("right")
    args = parser.parse_args()

    ref_dir = Path(args.reference_dir)
    add_reference_path(ref_dir)

    import torch
    import kornia
    import kornia.feature as KF
    from kornia.geometry import resize

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    def load_grayscale(path: str) -> torch.Tensor:
        image = kornia.io.load_image(path, kornia.io.ImageLoadType.RGB32, device=device).unsqueeze(0)
        image = resize(image, (args.resize_height, args.resize_width), align_corners=False)
        return image.mean(dim=1, keepdim=True)

    image0 = load_grayscale(args.left)
    image1 = load_grayscale(args.right)

    matcher = KF.LoFTR(pretrained="outdoor").to(device).eval()

    with torch.inference_mode():
        data = {
            "image0": image0,
            "image1": image1,
            "bs": image0.size(0),
            "hw0_i": image0.shape[2:],
            "hw1_i": image1.shape[2:],
        }
        if data["hw0_i"] == data["hw1_i"]:
            feats_c, feats_f = matcher.backbone(torch.cat([image0, image1], dim=0))
            feat_c0, feat_c1 = feats_c.split(data["bs"])
            feat_f0, feat_f1 = feats_f.split(data["bs"])
        else:
            (feat_c0, feat_f0), (feat_c1, feat_f1) = matcher.backbone(image0), matcher.backbone(image1)

        data.update(
            {
                "hw0_c": feat_c0.shape[2:],
                "hw1_c": feat_c1.shape[2:],
                "hw0_f": feat_f0.shape[2:],
                "hw1_f": feat_f1.shape[2:],
            }
        )
        feat_c0_pos = matcher.pos_encoding(feat_c0).permute(0, 2, 3, 1).reshape(feat_c0.shape[0], -1, feat_c0.shape[1])
        feat_c1_pos = matcher.pos_encoding(feat_c1).permute(0, 2, 3, 1).reshape(feat_c1.shape[0], -1, feat_c1.shape[1])
        feat_c0_coarse, feat_c1_coarse = matcher.loftr_coarse(feat_c0_pos, feat_c1_pos, None, None)

        matcher.coarse_matching(feat_c0_coarse, feat_c1_coarse, data, mask_c0=None, mask_c1=None)
        conf_matrix = data["conf_matrix"]
        mutual = (conf_matrix == conf_matrix.max(dim=2, keepdim=True)[0]) & (
            conf_matrix == conf_matrix.max(dim=1, keepdim=True)[0]
        )

    payload = {
        "image0": tensor_stats(image0),
        "image1": tensor_stats(image1),
        "feat_c0_backbone": tensor_stats(feat_c0),
        "feat_c1_backbone": tensor_stats(feat_c1),
        "feat_f0_backbone": tensor_stats(feat_f0),
        "feat_f1_backbone": tensor_stats(feat_f1),
        "feat_c0_pos": tensor_stats(feat_c0_pos),
        "feat_c1_pos": tensor_stats(feat_c1_pos),
        "feat_c0_coarse": tensor_stats(feat_c0_coarse),
        "feat_c1_coarse": tensor_stats(feat_c1_coarse),
        "coarse": {
            "conf_matrix": tensor_stats(conf_matrix),
            "threshold_count": int((conf_matrix > matcher.coarse_matching.thr).sum().item()),
            "mutual_count": int(mutual.sum().item()),
            "match_count": int(data["mconf"].shape[0]),
            "confidence_mean": float(data["mconf"].mean().item()) if data["mconf"].numel() > 0 else 0.0,
            "confidence_max": float(data["mconf"].max().item()) if data["mconf"].numel() > 0 else 0.0,
        },
    }

    output.write_text(json.dumps(payload, indent=2))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
