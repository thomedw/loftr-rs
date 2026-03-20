#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file


def normalize_tensors(data: dict[str, object]) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for key, value in data.items():
        if not isinstance(value, torch.Tensor):
            continue
        if key == "loftr_outdoorr_tentatives1":
            continue
        tensors[key] = value.detach().cpu().contiguous()
    return tensors


def build_metadata(data: dict[str, object], tensors: dict[str, torch.Tensor]) -> dict[str, object]:
    string_entries = {
        key: value
        for key, value in data.items()
        if isinstance(value, str)
    }
    return {
        "tensor_names": sorted(tensors),
        "tensor_shapes": {key: list(value.shape) for key, value in tensors.items()},
        "tensor_dtypes": {
            key: str(value.dtype).removeprefix("torch.")
            for key, value in tensors.items()
        },
        "string_entries": string_entries,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input Kornia .pt fixture file")
    parser.add_argument("--output", required=True, help="Output safetensors file")
    parser.add_argument("--metadata", required=True, help="Output JSON metadata file")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    metadata_path = Path(args.metadata)

    data = torch.load(input_path, map_location="cpu", weights_only=False)
    if not isinstance(data, dict):
        raise TypeError(f"expected a dict in {input_path}, got {type(data)!r}")

    tensors = normalize_tensors(data)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(output_path))

    metadata = build_metadata(data, tensors)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(f"saved {len(tensors)} tensors to {output_path}")
    print(f"saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()
