#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""The host's only step: fit an image into the 640x640 square the model reads."""

from pathlib import Path

import cv2  # pyright: ignore[reportMissingImports]
import numpy as np

IMAGE_SIZE = 640
PAD_VALUE = 114


def letterbox(path: Path, size: int = IMAGE_SIZE) -> np.ndarray:
    """Scale to fit, centre on grey and return RGB `(size, size, 3)` uint8, as Ultralytics' LetterBox does."""
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
    image = cv2.copyMakeBorder(
        image,
        round(dh - 0.1),
        round(dh + 0.1),
        round(dw - 0.1),
        round(dw + 0.1),
        cv2.BORDER_CONSTANT,
        value=(PAD_VALUE,) * 3,
    )
    return np.ascontiguousarray(image[..., ::-1])
