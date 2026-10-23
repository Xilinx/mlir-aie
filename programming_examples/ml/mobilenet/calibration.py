# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""The images the brevitas generators calibrate on, fetched when regenerating.

COCO 2017, all license 7 ("No known copyright restrictions",
http://flickr.com/commons/usage/). The first is the input the tracked golden data
comes from. Only the generators import this; tests read the tracked data.
"""

import hashlib
import os
import urllib.request
from pathlib import Path

import torch  # pyright: ignore[reportMissingImports]
from torchvision.datasets.folder import (  # pyright: ignore[reportMissingImports]
    default_loader,
)

CALIBRATION_IMAGES = (
    # COCO id 438047, steam locomotive
    (
        "http://images.cocodataset.org/train2017/000000438047.jpg",
        "813cf4ceafbd37f71828b0b8d9c05f65df898a45cce096a4e6d068637e3bcb27",
    ),
    # COCO id 110445, airliner on a runway
    (
        "http://images.cocodataset.org/train2017/000000110445.jpg",
        "75e7b9f732f3ea7345080438a6fd7936a450e8852250e4514db3f3c970413096",
    ),
    # COCO id 225040, warplane in flight
    (
        "http://images.cocodataset.org/train2017/000000225040.jpg",
        "3417d3cf8fed2697c71d404e29caf3aaf3633ab65ff0c33d0e5020f230ccb709",
    ),
    # COCO id 274219, portrait
    (
        "http://images.cocodataset.org/val2017/000000274219.jpg",
        "a2b2e8d425f64a51b566261300bbb1f528ab1967aee4f3283b9d214b37cb1c9d",
    ),
)


def calibration_image_paths() -> list[Path]:
    """Return the calibration images, downloading any not yet cached.

    The cache is ``$XDG_CACHE_HOME/mlir-aie/mobilenet-calib`` (default
    ``~/.cache``). Every file is checked against its pinned sha256 on every use.

    Raises:
        RuntimeError: An image cannot be downloaded, or its content does not
            match the pinned sha256.
    """
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    cache = cache / "mlir-aie" / "mobilenet-calib"
    cache.mkdir(parents=True, exist_ok=True)
    paths = []
    for url, sha256 in CALIBRATION_IMAGES:
        path = cache / url.rsplit("/", 1)[1]
        if not path.exists():
            part = path.with_suffix(".part")
            try:
                urllib.request.urlretrieve(url, part)
            except OSError as e:
                raise RuntimeError(
                    f"cannot download calibration image {url}: {e}"
                ) from e
            part.replace(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != sha256:
            raise RuntimeError(
                f"calibration image {path} (from {url}) has sha256 {digest}, "
                f"expected {sha256}; delete it to download it again"
            )
        paths.append(path)
    return paths


class CalibrationImages(torch.utils.data.Dataset):
    """The calibration images as ``(image, 0)`` samples, ``transform`` applied."""

    def __init__(self, transform):
        self.paths = calibration_image_paths()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        return self.transform(default_loader(self.paths[index])), 0


def calibration_loader(transform):
    """Return one batch holding every calibration image, in order."""
    images = CalibrationImages(transform)
    return torch.utils.data.DataLoader(images, batch_size=len(images), shuffle=False)
