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

_HELPER = (
    Path(__file__).resolve().parents[2]
    / "programming_examples"
    / "ml"
    / "block_datatypes"
    / "helper.h"
)


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


def test_encode_rejects_bad_shapes_and_non_finite():
    with pytest.raises(ValueError):
        bfp.encode(np.ones((2, 12), np.float32))
    with pytest.raises(ValueError):
        bfp.encode(np.array([np.inf] + [0.0] * 7, np.float32))
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
