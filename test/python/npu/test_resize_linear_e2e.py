# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %run_on_npu1_xrt% %pytest %s
# RUN: %run_on_npu2_xrt% %pytest %s
# RUN: %run_on_npu2_hrx% %pytest %s
# REQUIRES: xrt_python_bindings || hrx_python_bindings
"""Resize images with resize_linear's four entry points, and match resize_linear_ref exactly.

The sizes are chosen for the places a core can go wrong: a downscale that
skips source rows, an upscale that reuses them and clamps at both edges,
a width whose last elements take the scalar formula, an identity, and
sizes the kernel refuses, which emit zero rows while the streams stay in
step. One build resizes several images in turn, as a per-call size runs
it, so no state outlives its image. Where OpenCV is installed, the
reference is held to cv2.resize itself on its unoptimized path.
"""

import aie.iron as iron
import numpy as np
import pytest
from aie.iron.kernels import vision
from resize_linear_design import resize_rows

# Each case: the build's row bytes and its images (rows, columns, out_rows,
# out_columns).
CASES = {
    # A COCO photo letterboxed to 416 by YOLOX: down 0.65 both ways, source
    # rows skipped.
    "letterbox": (1920, 1248, [(480, 640, 312, 416)]),
    # Up, rows reused and both edges clamped; 3 * 53 = 159 elements, the
    # last 7 by the scalar formula.
    "upscale": (192, 192, [(37, 41, 61, 53)]),
    "identity": (192, 192, [(16, 64, 16, 64)]),
    # Past the build's output rows: refused, every row zeros.
    "refused": (192, 192, [(16, 64, 16, 65)]),
    # One build, three images in turn: refused between two resized.
    "in_turn": (
        768,
        768,
        [(100, 200, 70, 150), (30, 40, 30, 300), (90, 256, 120, 33)],
    ),
}


@pytest.mark.parametrize("case", list(CASES))
def test_device_rows_match_resize_linear_ref(case):
    in_bytes, out_bytes, sizes = CASES[case]
    # Random past each image's pixels too, which no output reads.
    images = [
        np.random.default_rng(k).integers(0, 256, (h, in_bytes), np.uint8)
        for k, (h, _, _, _) in enumerate(sizes)
    ]
    expected = np.concatenate(
        [
            vision.resize_linear_ref(h, w, oh, ow, image, out_bytes=out_bytes)
            for image, (h, w, oh, ow) in zip(images, sizes)
        ]
    )
    out = iron.full((expected.size,), 7, dtype=np.uint8)
    resize_rows(
        iron.tensor(np.concatenate([i.reshape(-1) for i in images]), dtype=np.uint8),
        out,
        sizes=tuple(sizes),
        in_bytes=in_bytes,
        out_bytes=out_bytes,
    )
    np.testing.assert_array_equal(out.numpy().reshape(expected.shape), expected)


@pytest.mark.parametrize("case", ["letterbox", "upscale", "identity", "in_turn"])
def test_the_reference_is_opencv_s_unoptimized_path(case):
    """cv2.resize with INTER_LINEAR, unoptimized, on one thread."""
    cv2 = pytest.importorskip("cv2")
    in_bytes, out_bytes, sizes = CASES[case]
    cv2.setNumThreads(1)
    optimized = cv2.useOptimized()
    cv2.setUseOptimized(False)
    try:
        for k, (h, w, oh, ow) in enumerate(sizes):
            if 3 * ow > out_bytes:
                continue
            image = np.random.default_rng(k).integers(0, 256, (h, in_bytes), np.uint8)
            pixels = np.ascontiguousarray(image[:, : 3 * w].reshape(h, w, 3))
            want = cv2.resize(pixels, (ow, oh), interpolation=cv2.INTER_LINEAR)
            got = vision.resize_linear_ref(h, w, oh, ow, image, out_bytes=out_bytes)
            np.testing.assert_array_equal(got[:, : 3 * ow].reshape(oh, ow, 3), want)
    finally:
        cv2.setUseOptimized(optimized)
