# test_transform_stack_size.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %pytest %s
"""Library designs leave the core stack to aiecc unless the caller fixes it."""

import os
import re
import subprocess
import sys

import numpy as np
import pytest
from ml_dtypes import bfloat16

from aie.iron import In, Out, kernels
from aie.iron.algorithms import kernel_design as kd
from aie.iron.algorithms._transform import transform, transform_parallel
from aie.iron.kernel import ExternalFunction
from aie.iron.device import NPU1Col1
from aie.utils import config
from aie.utils.hostruntime import set_current_device

try:
    config.peano_cxx_path()
    HAS_PEANO = True
except RuntimeError:
    HAS_PEANO = False
needs_peano = pytest.mark.skipif(
    not HAS_PEANO, reason="measures frames from compiled kernels"
)


@pytest.fixture
def npu1_device():
    with set_current_device(NPU1Col1()):
        yield


@pytest.mark.parametrize("num_channels", [1, 2])
def test_transform_parallel_leaves_the_stack_to_aiecc(npu1_device, num_channels):
    module = transform_parallel(
        kernels.gelu(tile_size=1024),
        np.ndarray[(4096,), np.dtype[bfloat16]],
        tile_size=1024,
        num_channels=num_channels,
    )
    assert "stack_size" not in str(module)


def test_transform_leaves_the_stack_to_aiecc(npu1_device):
    module = transform(
        kernels.gelu(tile_size=1024),
        np.ndarray[(4096,), np.dtype[bfloat16]],
        tile_size=1024,
    )
    assert "stack_size" not in str(module)


def test_kernel_design_leaves_the_stack_to_aiecc(npu1_device):
    mlir = str(kd.KernelDesign(kernels.gelu(tile_size=1024)).as_mlir())
    assert "stack_size" not in mlir


@needs_peano
def test_kernel_design_budgets_the_kernel_frames(npu2_device):
    # IRON's N128/CT_K32 flm tile: two sets of tiles fit beside the default
    # stack but not beside its 32 KiB stack accumulator.
    tile = dict(
        dim_m=64,
        band_m=64,
        dim_k=32,
        dim_n=128,
        chunk_k=32,
        out_chunk=512,
        bfp16_b=True,
    )
    fn = kernels.fused_mm(**tile)
    design = kd.KernelDesign(fn, calls=4)
    params = fn.param_values(design.sample_inputs())
    mlir = str(design.design(params=params).as_mlir())
    depths = re.findall(r"aie\.objectfifo @out0\([^)]*?(\d+) : i32\)", mlir)
    assert depths == ["1"]


TABLE_WORDS, TILE_WORDS = 5000, 3000
TABLE = np.zeros(TABLE_WORDS, np.int32)
TABLE[:3] = (1, 2, 3)


def table_add():
    """A tile plus a 20 KB constant table: 24 KB of tiles each way."""
    tile = np.ndarray[(TILE_WORDS,), np.dtype[np.int32]]
    return ExternalFunction(
        "table_add",
        source_string=f"""#include <stdint.h>
static const int32_t table[{TABLE_WORDS}] = {{1, 2, 3}};
extern "C" void table_add(const int32_t *in, int32_t *out) {{
  for (int i = 0; i < {TILE_WORDS}; i++)
    out[i] = in[i] + table[in[i] & 4095];
}}""",
        arg_types=[tile, tile],
        contract=kernels.KernelContract(
            roles=(In, Out), reference=lambda x: x + TABLE[x & 4095]
        ),
    )


@needs_peano
def test_kernel_design_budgets_the_kernels_static_data(npu2_device):
    # Two sets of tiles fit beside the stack (49 KB of 64), not beside the
    # table as well (69 KB): one set does (45 KB).
    mlir = str(kd.KernelDesign(table_add()).as_mlir())
    depths = re.findall(r"aie\.objectfifo @out0\([^)]*?(\d+) : i32\)", mlir)
    assert depths == ["1"]


@needs_peano
@pytest.mark.parametrize("use_cache", [True, False])
def test_kernel_design_reads_frames_through_the_design_cache(tmp_path, use_cache):
    # NPU_CACHE_HOME is read at import, so generate in a fresh interpreter.
    script = (
        "from aie.iron import kernels\n"
        "from aie.iron.algorithms import kernel_design as kd\n"
        "from aie.iron.device import NPU2Col1\n"
        "from aie.utils.hostruntime import set_current_device\n"
        "set_current_device(NPU2Col1())\n"
        "kd.KernelDesign(kernels.gelu(tile_size=1024)).design()"
        f".specialize(use_cache={use_cache}).as_mlir()\n"
    )
    subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "NPU_CACHE_HOME": str(tmp_path)},
        check=True,
    )
    assert any((tmp_path / "objects").rglob("*.o")) == use_cache


def test_kernel_design_keeps_an_explicit_stack(npu1_device):
    design = kd.KernelDesign(kernels.gelu(tile_size=1024), stack_bytes=4096)
    mlir = str(design.as_mlir())
    assert re.findall(r"stack_size = (\d+) : i32", mlir) == ["4096"]
