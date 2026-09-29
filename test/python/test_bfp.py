# test_bfp.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %pytest %s
"""aie.utils.bfp: the bfp16ebs8 host codec and the mmul tile shuffle (no NPU)."""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest
from aie.utils import bfp
from ml_dtypes import bfloat16

_HELPER = (
    Path(__file__).resolve().parents[2]
    / "programming_examples"
    / "ml"
    / "block_datatypes"
    / "helper.h"
)


def test_dtype_metadata_uses_native_scalar_sizes():
    for dtype in (np.int8, np.int32, np.float32, bfloat16):
        assert bfp.itemsize(dtype) == np.dtype(dtype).itemsize
        assert bfp.dtype_name(dtype) == np.dtype(dtype).name
        assert bfp.values_per_elem(dtype) == 1
    assert bfp.itemsize(bfp.v8bfp16ebs8) == 9
    assert bfp.values_per_elem(bfp.v8bfp16ebs8) == 8


def test_decode_accepts_strided_and_empty_storage():
    values = np.arange(-8, 8, dtype=np.float32).reshape(2, 8)
    encoded = bfp.encode(values)
    strided = np.zeros((2, 18), dtype=np.uint8)
    strided[:, ::2] = encoded
    np.testing.assert_array_equal(bfp.decode(strided[:, ::2]), values)
    assert bfp.decode(bfp.encode(np.empty((2, 0), np.float32))).shape == (2, 0)


def test_layout_and_small_integers_are_exact():
    # 8 values -> 9 bytes: shared exponent first, then one mantissa each.
    x = np.array([[1, 2, 3, 4, 5, 6, 7, 8]], np.float32)
    enc = bfp.encode(x)
    assert enc.shape == (1, 9) and enc[0, 0] == 127 + 3  # exponent of 8
    assert np.array_equal(bfp.decode(enc), x)
    # Integers within 7 magnitude bits of the block's largest round-trip.
    rng = np.random.default_rng(0)
    ints = rng.integers(-64, 64, (32, 256)).astype(np.float32)
    assert np.array_equal(bfp.quantize(ints), ints)
    # Smaller values in a block lose the bits below the shared exponent.
    y = np.array([[64.0, 0.75, 0, 0, 0, 0, 0, 0]], np.float32)
    assert bfp.quantize(y)[0, 1] == 0.0  # 1 LSB is 64 / 64 = 1.0
    z = np.array([[64.0, -0.75, 0, 0, 0, 0, 0, 0]], np.float32)
    assert bfp.quantize(z)[0, 1] == -1.0  # truncation toward -inf
    q = bfp.quantize(rng.standard_normal((16, 64)).astype(np.float32))
    assert q.shape == (16, 64) and q.dtype == np.float32


def test_conv_even_carry_is_one_sided():
    # 1.999 * 64 rounds to +128, which does not fit, so the block's exponent
    # goes up and 0.01 (0.64 LSB) is lost at the coarser step.
    up = np.array([[1.999, 0.5, -0.25, 0.01, 0, 0, 0, 0]], np.float32)
    np.testing.assert_array_equal(
        bfp.quantize(up, rounding="conv_even"), [[2.0, 0.5, -0.25, 0, 0, 0, 0, 0]]
    )
    # -1.999 * 64 rounds to -128, which fits: the exponent stays and 0.01
    # keeps its 1 LSB of 2**-6.
    down = -up
    np.testing.assert_array_equal(
        bfp.quantize(down, rounding="conv_even"),
        [[-2.0, -0.5, 0.25, -(2.0**-6), 0, 0, 0, 0]],
    )


# amd/IRON's f32_to_bfp16ebs8 (iron/operators/flm/packing.py) on the input
# below, pasted from one run. The blocks cover ties to even (106.5 and 94.5
# LSBs), a value 32 binades below its block (0 or -1), +128 saturating to 127
# and -128 fitting.
_IRON_CONV_EVEN_IN = np.array(
    [
        [1.0, 106.5 / 64, 94.5 / 64, -1.5, 0.3, -0.7, 1e-12, -1e-12],
        [1.999, 0.5, -0.25, 0.1, -0.1, 0.01, -0.01, 0.0],
        [-1.999, 0.5, 0.25, -0.1, 0.1, -0.01, 0.01, 0.0],
        [-106.5 / 64, -93.5 / 64, 3.0, -2.75, 0.123, -0.456, 0.789, -0.001],
    ],
    np.float32,
).reshape(1, 32)
_IRON_CONV_EVEN_BYTES = [
    [127, 64, 106, 94, 160, 19, 211, 0, 255],
    [127, 127, 32, 240, 6, 250, 1, 255, 0],
    [127, 128, 32, 16, 250, 6, 255, 1, 0],
    [128, 203, 209, 96, 168, 4, 241, 25, 0],
]


def test_encode_conv_even_matches_iron_packer():
    enc = bfp.encode(_IRON_CONV_EVEN_IN, rounding="conv_even")
    np.testing.assert_array_equal(enc, np.reshape(_IRON_CONV_EVEN_BYTES, (1, 36)))
    # floor is still the default, and truncates toward -inf instead: -6.4
    # and -53.25 LSBs become -7 and -54.
    floor = bfp.encode(_IRON_CONV_EVEN_IN).view(np.int8)
    assert floor[0, 14] == -7 and floor[0, 28] == -54


def test_encode_rejects_bad_shapes_and_non_finite():
    with pytest.raises(ValueError):
        bfp.encode(np.ones((2, 12), np.float32))
    with pytest.raises(ValueError):
        bfp.encode(np.array([np.inf] + [0.0] * 7, np.float32))
    with pytest.raises(ValueError):
        bfp.encode(np.ones((1, 8), np.float32), rounding="ceil")
    with pytest.raises(ValueError):
        bfp.decode(np.zeros(10, np.uint8))
    with pytest.raises(ValueError):
        bfp.shuffle(np.zeros((16, 36), np.uint8), 32, 16, 12, 8)


def test_shuffle_makes_subtiles_contiguous_and_inverts():
    H, W, th, tw = 32, 64, 16, 32
    rng = np.random.default_rng(1)
    enc = bfp.encode(rng.standard_normal((H, W)).astype(np.float32))
    sh = bfp.shuffle(enc, W, H, tw, th)
    assert sh.shape == enc.shape
    # Read as the DMA delivers it, a tile (16 rows x 36 bytes) row by row:
    # its first 72 bytes are the top-left 8 rows x 9 bytes of the original,
    # the next 72 the sub-tile to its right, and so on in raster order.
    tile = sh[:th, :36].ravel()
    assert np.array_equal(tile[:72], enc[:8, :9].ravel())
    assert np.array_equal(tile[72:144], enc[:8, 9:18].ravel())
    assert np.array_equal(tile[-72:], enc[8:16, 27:36].ravel())
    # The second tile (columns 32..63, bytes 36..71) stays in place.
    tile1 = sh[:th, 36:72].ravel()
    assert np.array_equal(tile1[:72], enc[:8, 36:45].ravel())
    assert np.array_equal(bfp.shuffle(sh, W, H, tw, th, unshuffle=True), enc)
    # A flat input is accepted and the tile height may span the whole matrix.
    assert np.array_equal(bfp.shuffle(enc.ravel(), W, H, tw, H), sh)


@pytest.mark.skipif(
    shutil.which("g++") is None or not _HELPER.exists(),
    reason="needs g++ and programming_examples/ml/block_datatypes/helper.h",
)
def test_codec_matches_the_cpp_helper_bit_for_bit():
    src = r"""
    #include "helper.h"
    #include <fstream>
    int main(int argc, char **argv) {
      int n = atoi(argv[2]), W = atoi(argv[3]), H = atoi(argv[4]);
      int tw = atoi(argv[5]), th = atoi(argv[6]);
      std::vector<float> in(n);
      std::ifstream(argv[1], std::ios::binary).read((char *)in.data(), n * 4);
      auto enc = floatToBfp16(8, n, in.data(), 0);
      std::ofstream("enc.u8", std::ios::binary).write((char *)enc.data(), enc.size());
      auto sh = shuffleMatrixForBfp16ebs8(W, H, tw, th, enc);
      std::ofstream("shuf.u8", std::ios::binary).write((char *)sh.data(), sh.size());
      auto dec = bfp16ebs8ToFloat(enc.size(), enc.data(), 0);
      std::ofstream("dec.f32", std::ios::binary).write((char *)dec.data(), dec.size() * 4);
      return 0;
    }
    """
    with tempfile.TemporaryDirectory() as d:
        shutil.copy(_HELPER, d)
        Path(d, "driver.cpp").write_text(src)
        subprocess.run(
            ["g++", "-O1", "-std=c++17", "-o", "driver", "driver.cpp"],
            cwd=d,
            check=True,
        )
        rng = np.random.default_rng(2)
        H, W, th, tw = 64, 256, 32, 64
        # Values within 30 binades of each other: past 2**31 the header's
        # integer shift in bfp16ebs8ToFloat overflows, where decode follows
        # the arithmetic instead.
        x = rng.standard_normal((H, W)) * 2.0 ** rng.uniform(-14, 14, (H, W))
        x = x.astype(np.float32)
        x.tofile(os.path.join(d, "in.f32"))
        subprocess.run(
            ["./driver", "in.f32", str(x.size), str(W), str(H), str(tw), str(th)],
            cwd=d,
            check=True,
        )
        enc = bfp.encode(x)
        assert np.array_equal(enc.ravel(), np.fromfile(Path(d, "enc.u8"), np.uint8))
        assert np.array_equal(
            bfp.shuffle(enc, W, H, tw, th).ravel(),
            np.fromfile(Path(d, "shuf.u8"), np.uint8),
        )
        assert np.array_equal(
            bfp.decode(enc).ravel(), np.fromfile(Path(d, "dec.f32"), np.float32)
        )
