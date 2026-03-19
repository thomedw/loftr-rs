#!/usr/bin/env python3

from __future__ import annotations

import argparse
import collections
import io
import math
import pickle
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class StorageType:
    dtype: np.dtype


@dataclass(frozen=True)
class StorageRef:
    storage_type: StorageType
    key: str
    location: str
    size: int


@dataclass(frozen=True)
class TensorRef:
    storage: StorageRef
    storage_offset: int
    size: tuple[int, ...]
    stride: tuple[int, ...]


FLOAT_STORAGE = StorageType(np.dtype(np.float32))
DOUBLE_STORAGE = StorageType(np.dtype(np.float64))


def rebuild_tensor_v2(
    storage: StorageRef,
    storage_offset: int,
    size: tuple[int, ...],
    stride: tuple[int, ...],
    _requires_grad: bool,
    _backward_hooks: object,
) -> TensorRef:
    return TensorRef(storage, storage_offset, tuple(size), tuple(stride))


class FixtureUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> object:
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return rebuild_tensor_v2
        if module == "torch" and name == "FloatStorage":
            return FLOAT_STORAGE
        if module == "torch" and name == "DoubleStorage":
            return DOUBLE_STORAGE
        if module == "collections" and name == "OrderedDict":
            return collections.OrderedDict
        raise pickle.UnpicklingError(f"unsupported global: {module}.{name}")

    def persistent_load(self, pid: object) -> StorageRef:
        kind, storage_type, key, location, size = pid
        if kind != "storage":
            raise pickle.UnpicklingError(f"unsupported persistent kind: {kind}")
        return StorageRef(storage_type, key, location, size)


def load_fixture(path: Path) -> dict[str, object]:
    with zipfile.ZipFile(path) as archive:
        payload = FixtureUnpickler(io.BytesIO(archive.read("archive/data.pkl"))).load()
        return {
            key: materialize_tensor(archive, value) if isinstance(value, TensorRef) else value
            for key, value in payload.items()
        }


def materialize_tensor(archive: zipfile.ZipFile, tensor: TensorRef) -> np.ndarray:
    raw = archive.read(f"archive/data/{tensor.storage.key}")
    flat = np.frombuffer(raw, dtype=tensor.storage.storage_type.dtype)
    length = math.prod(tensor.size)
    return flat[tensor.storage_offset : tensor.storage_offset + length].reshape(tensor.size)


def to_grayscale_image(tensor: np.ndarray) -> Image.Image:
    if tensor.ndim == 4:
        tensor = tensor[0, 0]
    image = np.clip(tensor, 0.0, 1.0)
    image = (image * 255.0).round().astype(np.uint8)
    return Image.fromarray(image, mode="L")


def export_images(fixture: dict[str, object], output_dir: Path, prefix: str) -> None:
    image0 = to_grayscale_image(np.asarray(fixture["image0"]))
    image1 = to_grayscale_image(np.asarray(fixture["image1"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    image0.save(output_dir / f"{prefix}-image0.png")
    image1.save(output_dir / f"{prefix}-image1.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export PNG inputs from a Kornia LoFTR fixture.")
    parser.add_argument("fixture", type=Path, help="Path to the Kornia homography fixture .pt file")
    parser.add_argument("output_dir", type=Path, help="Directory to write the extracted images into")
    parser.add_argument("--prefix", default="loftr-demo", help="Output filename prefix")
    args = parser.parse_args()

    fixture = load_fixture(args.fixture)
    export_images(fixture, args.output_dir, prefix=args.prefix)


if __name__ == "__main__":
    main()
