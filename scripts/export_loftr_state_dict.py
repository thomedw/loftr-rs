#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch
from kornia.feature import LoFTR


def load_state_dict(checkpoint_path: str | None, pretrained: str | None):
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        normalized = {}
        for key, value in state_dict.items():
            if key.startswith("matcher."):
                normalized[key.replace("matcher.", "", 1)] = value
            else:
                normalized[key] = value
        return normalized

    model = LoFTR(pretrained=pretrained)
    return model.state_dict()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="Output .pt or .safetensors path")
    parser.add_argument(
        "--pretrained",
        default=None,
        choices=[None, "outdoor", "indoor", "indoor_new"],
        help="Kornia LoFTR pretrained preset to instantiate directly",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Existing Kornia checkpoint path; if set, this wins over --pretrained",
    )
    args = parser.parse_args()

    state_dict = load_state_dict(args.checkpoint, args.pretrained)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.suffix == ".safetensors":
        from safetensors.torch import save_file

        save_file(state_dict, str(out))
    else:
        torch.save(state_dict, str(out))

    print(f"saved {len(state_dict)} tensors to {out}")


if __name__ == "__main__":
    main()
