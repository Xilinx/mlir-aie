#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""The host's only step: fit an image into the 640x640 square the model reads."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2  # pyright: ignore[reportMissingImports]
import numpy as np

IMAGE_SIZE = 640
PAD_VALUE = 114


@dataclass(frozen=True)
class Fit:
    """Where letterbox puts an image: its original size, its scale and the padding before it.

    Attributes:
        height: The original image's height.
        width: Its width.
        ratio: The scale both axes take.
        left: Columns of padding left of the scaled image.
        top: Rows of padding above it.
    """

    height: int
    width: int
    ratio: float
    left: int
    top: int


def letterbox(path: Path, size: int = IMAGE_SIZE) -> tuple[np.ndarray, Fit]:
    """Scale to fit and centre on grey, as Ultralytics' LetterBox does.

    Args:
        path: The image file.
        size: The square's side.

    Returns:
        The RGB `(size, size, 3)` uint8 image, and where it was put.

    Raises:
        FileNotFoundError: OpenCV can't read `path`.
    """
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(path)
    height, width = image.shape[:2]
    ratio = min(size / height, size / width)
    new_width, new_height = round(width * ratio), round(height * ratio)
    if (new_width, new_height) != (width, height):
        image = cv2.resize(
            image, (new_width, new_height), interpolation=cv2.INTER_LINEAR
        )
    dw, dh = (size - new_width) / 2, (size - new_height) / 2
    top, left = round(dh - 0.1), round(dw - 0.1)
    image = cv2.copyMakeBorder(
        image,
        top,
        round(dh + 0.1),
        left,
        round(dw + 0.1),
        cv2.BORDER_CONSTANT,
        value=(PAD_VALUE,) * 3,
    )
    return np.ascontiguousarray(image[..., ::-1]), Fit(height, width, ratio, left, top)
